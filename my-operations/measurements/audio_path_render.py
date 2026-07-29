#!/usr/bin/env python3
"""KROK 1/2: renderuje obie sciezki audio i ZAPISUJE surowe spektrogramy.

Rozdzielenie renderowania od analizy jest celowe — statystyke mozna przeliczyc
dowolna liczbe razy bez ponownego dotykania GPU (i bez zuzywania czasu karty przy
kazdej poprawce metody).

M=40 zamiast 20, zeby uzyskac REPLIKATY: kazdy kat dzielimy na dwie ROZLACZNE
polowki po 20 renderow, co daje niezalezne estymaty tej samej wielkosci.
Niepewnosc bierzemy z rozrzutu replikatow, nie z zalozen o rozkladzie.

Sciezek NIE przeplatamy w jednej instancji Simulatora — komplet jedna, potem
komplet druga, zeby nie mieszac sekwencji RNG.

PRODUKUJE: outputs/measurements/paths_specs.npz (~95 MiB, gitignored)
NASTEPNY KROK: audio_path_analyse.py

Raport: RAPORT_SESJI_2026-07-26_29.md §2.2 | Dokument: GENERATOR_PARAMS.md §4.3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
MEAS_OUT = Path(__file__).resolve().parents[2] / "outputs/measurements"
MEAS_OUT.mkdir(parents=True, exist_ok=True)

from echo_core import audio, spectrogram
from echo_core.params import (INDIRECT_RAY_COUNT, N_MAX, N_MIN, N_PROBE, SENSOR_HEIGHT,
                              SIGNAL_10DEG, TARGET_SNR, THREAD_COUNT, WARMUP_DISCARD)
from echo_core.paths import (CHIRP_PATH, MATERIAL_CONFIG, OUT_ROOT, probe_census_csv,
                             scene_h5, scene_mesh)
from echo_core.scenes import HELD_OUT, SCENE_ORDER, load_scene_locations
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion  # noqa: F401
import habitat_sim
import librosa

M = 40
ANGLES = (0.0, 10.0)
POSITIONS = [("office_1", 33), ("frl_apartment_5", 186)]
OUT = Path(str(MEAS_OUT / "paths_specs.npz"))

chirp, _ = librosa.load(str(CHIRP_PATH), sr=spectrogram.SAMPLE_RATE, mono=True)
MC = str(MATERIAL_CONFIG)


def build(scene):
    class A:
        pass
    a = A()
    a.scene = str(scene_mesh(scene))
    a.sensor_height = SENSOR_HEIGHT
    a.material_config = MC
    a.out_dir = str(OUT_ROOT / "_measurement_scratch")
    a.indirect_ray_count = INDIRECT_RAY_COUNT
    a.thread_count = THREAD_COUNT
    a.gpu_device_id = 0
    return audio.build_simulator(a)


def render_double(sim, pos, ang, mc):
    obs, _, _ = audio.phase3_echolocation(sim, pos, ang, mc)
    return np.array(obs["audio_sensor"])


def render_single(sim, pos, ang, mc):
    """phase3_echolocation BEZ jawnego runSimulation (test_rlr_audio.py:281).

    setAudioSourceTransform ZOSTAJE: Sensor._get_audio_observation() ustawia
    wylacznie transform SLUCHACZA, wiec bez tej linii zrodlo nigdy nie zostaloby
    ustawione i echolokacja by nie zadzialala.
    """
    st = habitat_sim.AgentState()
    st.position = pos
    st.rotation = habitat_sim.utils.common.quat_from_angle_axis(
        np.deg2rad(ang), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)
    a = sim.get_agent(0)._sensors["audio_sensor"]
    if mc is not None:
        a.setAudioMaterialsJSON(mc)
    lp = np.array(a.node.absolute_translation)
    a.setAudioSourceTransform(lp)
    a.setAudioListenerTransform(lp, quaternion.as_float_array(st.rotation))
    return np.array(sim.get_sensor_observations()["audio_sensor"])


store, n_sims = {}, 0
for scene, loc_id in POSITIONS:
    pos = load_scene_locations(scene)[1][loc_id]
    for path_name, fn in (("podwojna", render_double), ("pojedyncza", render_single)):
        sim = build(scene)
        n_sims += 1
        print(f"  [{n_sims}/4 Simulator] {scene}/{loc_id} sciezka {path_name}", flush=True)
        try:
            mc = MC
            for ang in ANGLES:
                specs, times = [], []
                for _ in range(M):
                    t0 = time.perf_counter()
                    raw = fn(sim, pos, ang, mc)
                    times.append(time.perf_counter() - t0)
                    mc = None
                    _e, s = spectrogram.render_spectrogram(np.transpose(raw), chirp)
                    specs.append(s.astype(np.float32))
                key = f"{scene}|{loc_id}|{path_name}|{ang:.0f}"
                store[key] = np.stack(specs)
                store[key + "|times"] = np.array(times)
                print(f"      kat {ang:>4.0f} st.: {M} renderow, mediana "
                      f"{np.median(times)*1000:.1f} ms", flush=True)
        finally:
            sim.close()

np.savez_compressed(OUT, **store)
print(f"\n  Zapisano {len([k for k in store if not k.endswith('|times')])} zestawow "
      f"po {M} spektrogramow -> {OUT} ({OUT.stat().st_size/2**20:.0f} MiB)")
print(f"  Konstrukcji Simulatora w tym procesie: {n_sims} (prog ~30)")
