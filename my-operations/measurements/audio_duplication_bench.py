#!/usr/bin/env python3
"""DOWOD: symulacja akustyczna wykonywala sie DWA RAZY na render (274 vs 139 ms).

Pytanie: `phase3_echolocation()` wolalo `runSimulation()` jawnie, a
`sim.get_sensor_observations()` dla sensora AUDIO wchodzi w
`Sensor._get_audio_observation()` (habitat-sim/src_python/habitat_sim/simulator.py:763-777),
ktore wola ja PONOWNIE i to jej wynik zwraca. Ile to kosztuje i czy da sie pominac
renderowanie wizualne bez ruszania sekwencji RNG?

Metoda: cztery warianty tego samego renderu, mediana z N powtorzen. Dodatkowo
CZESC 1 sprawdza rownowaznosc sekwencji RNG: E1 wykazal, ze swiezy Simulator
odtwarza identyczna sekwencje, wiec jesli wariant B robi te same wywolania audio
co A, IR musi byc identyczne CO DO BITU miedzy dwiema swiezymi instancjami.

Wynik (2026-07-28, office_1, 25 renderow):
    A  2 symulacje + RGB + depth   274.1 ms   <- stan zastany
    B  2 symulacje, bez wizualnych 277.4 ms   <- sekwencja RNG ZACHOWANA (bit-exact)
    C  1 symulacja, bez wizualnych 139.3 ms   <- dokladnie 2x szybciej
    D  same RGB + depth              0.2 ms   <- wizualne sa DARMOWE (700x taniej)

Raport: RAPORT_SESJI_2026-07-26_29.md §2.2 | Dokument: GENERATOR_PARAMS.md §4.3

Uruchomienie:
    conda activate habitat
    python my-operations/measurements/audio_duplication_bench.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion
import habitat_sim

from echo_core import audio
from echo_core.params import INDIRECT_RAY_COUNT, SENSOR_HEIGHT, THREAD_COUNT
from echo_core.paths import MATERIAL_CONFIG, OUT_ROOT, scene_mesh
from echo_core.scenes import load_scene_locations


class _Args:
    def __init__(self, scene):
        self.scene = str(scene_mesh(scene))
        self.sensor_height = SENSOR_HEIGHT
        self.material_config = str(MATERIAL_CONFIG)
        self.out_dir = str(OUT_ROOT / "_bench_scratch")
        self.indirect_ray_count = INDIRECT_RAY_COUNT
        self.thread_count = THREAD_COUNT
        self.gpu_device_id = 0


def _prep(sim, pos, ang, mc):
    """Dokladnie to, co robi phase3_echolocation PRZED get_sensor_observations()."""
    st = habitat_sim.AgentState()
    st.position = pos
    st.rotation = habitat_sim.utils.common.quat_from_angle_axis(
        np.deg2rad(ang), np.array([0., 1., 0.]))
    sim.get_agent(0).set_state(st)
    a = sim.get_agent(0)._sensors["audio_sensor"]
    a.setAudioMaterialsJSON(mc)
    lp = np.array(a.node.absolute_translation)
    a.setAudioSourceTransform(lp)          # KRYTYCZNE — _get_audio_observation() tego NIE robi
    a.setAudioListenerTransform(lp, quaternion.as_float_array(st.rotation))
    return a


def var_A(sim, pos, ang, mc):     # 2 symulacje + RGB + depth
    obs, _, _ = audio.phase3_echolocation(sim, pos, ang, mc, run_simulation=True)
    return np.array(obs["audio_sensor"])


def var_B(sim, pos, ang, mc):     # 2 symulacje, ZERO wizualnych
    a = _prep(sim, pos, ang, mc)
    a.runSimulation(sim)
    return np.array(sim._Simulator__sensors[0]["audio_sensor"].get_observation())


def var_C(sim, pos, ang, mc):     # 1 symulacja (sciezka produkcyjna)
    obs, _, _ = audio.phase3_echolocation(sim, pos, ang, mc, run_simulation=False)
    return np.array(obs["audio_sensor"])


def visual_only(sim):
    w = sim._Simulator__sensors[0]
    return {u: (w[u].draw_observation(), w[u].get_observation())[1] for u in ("rgb", "depth")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="office_1")
    ap.add_argument("--loc", type=int, default=None, help="loc_id (domyslnie srodkowa)")
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    ids, positions = load_scene_locations(args.scene)
    loc = args.loc if args.loc is not None else ids[len(ids) // 2]
    pos, mc = positions[loc], str(MATERIAL_CONFIG)

    # --- CZESC 1: czy wariant B odtwarza sekwencje RNG wariantu A? ---
    seqs = {}
    for name, fn in (("A", var_A), ("B", var_B)):
        sim = audio.build_simulator(_Args(args.scene))
        try:
            seqs[name] = [fn(sim, pos, float(a), mc) for a in (0., 10., 20., 30., 40.)]
        finally:
            sim.close()
    ident = all(np.array_equal(a, b) for a, b in zip(seqs["A"], seqs["B"]))
    print(f"\n=== CZESC 1: sekwencja RNG przy pominieciu renderowania wizualnego ===")
    for i, (a, b) in enumerate(zip(seqs["A"], seqs["B"])):
        print(f"  render {i}: {a.shape} vs {b.shape}  bit-identyczne={np.array_equal(a, b)}")
    print(f"  WERDYKT: sekwencja {'ZACHOWANA' if ident else 'ZMIENIONA'}")

    # --- CZESC 2: czasy ---
    sim = audio.build_simulator(_Args(args.scene))
    res = {}
    try:
        for name, fn in (("A  2xaudio + RGB + depth", var_A),
                         ("B  2xaudio, bez wizualnych", var_B),
                         ("C  1xaudio, bez wizualnych", var_C)):
            for _ in range(3):
                fn(sim, pos, 0.0, mc)                    # rozgrzewka pomiaru
            ts = []
            for _ in range(args.n):
                t = time.perf_counter(); fn(sim, pos, 0.0, mc); ts.append(time.perf_counter() - t)
            res[name] = float(np.median(ts))
        for _ in range(3):
            visual_only(sim)
        ts = []
        for _ in range(args.n):
            t = time.perf_counter(); visual_only(sim); ts.append(time.perf_counter() - t)
        res["D  tylko RGB + depth"] = float(np.median(ts))
    finally:
        sim.close()

    print(f"\n=== CZESC 2: mediana czasu renderu ({args.scene}/{loc}, n={args.n}) ===")
    base = res["A  2xaudio + RGB + depth"]
    for k, v in res.items():
        print(f"  {k:<32}{v*1000:8.1f} ms   {base/v:5.2f}x wzgl. A")


if __name__ == "__main__":
    main()
