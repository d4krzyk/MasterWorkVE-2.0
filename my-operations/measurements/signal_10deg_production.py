#!/usr/bin/env python3
"""DOWOD: SIGNAL_10DEG = 0.0644 trzyma sie na geometrii produkcyjnej.

Pytanie: cala charakterystyka szumu byla mierzona na pozycjach `snap_point`, czyli
0.21 m WYZEJ niz idzie produkcja (patrz agent_height_vs_pkl.py i
navmesh_offset_survey.py). Czy stala SIGNAL_10DEG i podloga szumu zmieniaja sie
po poprawce `y`?

Metoda: te same wspolrzedne (x, z), dwie wersje `y` — produkcyjna z graph.pkl
i historyczna z navmesha. Katy 0 i 10 stopni, po 2N renderow, podzial na polowki:

    noise_ab = RMSE(polowka_A, polowka_B)                # sqrt(2)*sigma_N
    sigma_1  = noise_ab / sqrt(2) * sqrt(N)
    sygnal   = sqrt(RMSE(kat0, kat10)^2 - noise_ab^2)    # odszumiony

Wynik (2026-07-28, office_1, 3 pozycje, N=10 na polowke):
    sigma_1 produkcja/charakteryzacja = 0.962x   (produkcja WRECZ CICHSZA)
    sygnal 10 st. na geometrii produkcyjnej: 0.06441-0.06491
    -> miesci sie w udokumentowanym zakresie 0.0639-0.0662, stala pozostaje wazna

ZASTRZEZENIE: 3 pozycje w jednej scenie. To potwierdza zgodnosc rzedu wielkosci,
NIE przemierza stalej od nowa.

Raport: RAPORT_SESJI_2026-07-26_29.md §3.3 | Dokument: GENERATOR_PARAMS.md §5 ogr. 8

Uruchomienie:
    conda activate habitat
    python my-operations/measurements/signal_10deg_production.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion  # noqa: F401
import habitat_sim  # noqa: F401
import librosa

from echo_core import audio, spectrogram
from echo_core.params import (INDIRECT_RAY_COUNT, SENSOR_HEIGHT, SIGNAL_10DEG,
                              TARGET_SNR, THREAD_COUNT)
from echo_core.paths import CHIRP_PATH, MATERIAL_CONFIG, OUT_ROOT, scene_mesh
from echo_core.scenes import load_scene_locations

SCENE = "office_1"
N = 10                      # renderow na polowke
ANGLES = (0.0, 10.0)
LOCS = [5, 10, 11]


class _Args:
    scene = str(scene_mesh(SCENE))
    sensor_height = SENSOR_HEIGHT
    material_config = str(MATERIAL_CONFIG)
    out_dir = str(OUT_ROOT / "_measurement_scratch")
    indirect_ray_count = INDIRECT_RAY_COUNT
    thread_count = THREAD_COUNT
    gpu_device_id = 0


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    _ids, positions = load_scene_locations(SCENE)      # `y` PRODUKCYJNE (graph.pkl)
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=spectrogram.SAMPLE_RATE, mono=True)
    mc = str(MATERIAL_CONFIG)

    sim = audio.build_simulator(_Args)
    rows = []
    try:
        for lid in LOCS:
            pos_prod = positions[lid]
            # wariant historyczny: te same (x, z), ale `y` z powierzchni navmesha
            pos_snap = np.array(sim.pathfinder.snap_point(list(pos_prod)), dtype=np.float32)
            for vname, pos in (("graph.pkl (produkcja)", pos_prod),
                               ("snap_point (charakteryz.)", pos_snap)):
                est = {}
                for ang in ANGLES:
                    specs = []
                    for _ in range(2 * N):
                        obs, _l, _r = audio.phase3_echolocation(
                            sim, pos, ang, mc, run_simulation=False)
                        rir = np.transpose(np.array(obs["audio_sensor"]))
                        _echo, spec = spectrogram.render_spectrogram(rir, chirp)
                        specs.append(spec)
                    est[(ang, "A")] = np.mean(specs[:N], axis=0)
                    est[(ang, "B")] = np.mean(specs[N:], axis=0)

                a0, a1 = ANGLES
                noise_ab = float(np.mean([rmse(est[(a, "A")], est[(a, "B")]) for a in ANGLES]))
                sigma_1 = noise_ab / np.sqrt(2.0) * np.sqrt(N)
                raw = float(np.mean([rmse(est[(a0, h)], est[(a1, h)]) for h in ("A", "B")]))
                signal = float(np.sqrt(max(raw ** 2 - noise_ab ** 2, 0.0)))
                n_req = int(np.ceil((TARGET_SNR * sigma_1 / SIGNAL_10DEG) ** 2))
                rows.append(dict(variant=vname, sigma_1=sigma_1, signal=signal))
                print(f"  lok {lid:<4} {vname:<26} y={pos[1]:+.4f}  sigma_1={sigma_1:.5f}  "
                      f"sygnal_10st={signal:.5f}  N={n_req}")
    finally:
        sim.close()

    print("\n" + "=" * 88)
    for v in ("graph.pkl (produkcja)", "snap_point (charakteryz.)"):
        sub = [r for r in rows if r["variant"] == v]
        s1 = np.array([r["sigma_1"] for r in sub])
        sg = np.array([r["signal"] for r in sub])
        print(f"  {v:<28} sigma_1 sr.={s1.mean():.5f}   sygnal sr.={sg.mean():.5f}")
    g = np.array([r["sigma_1"] for r in rows if "graph" in r["variant"]])
    s = np.array([r["sigma_1"] for r in rows if "snap" in r["variant"]])
    print(f"\n  stosunek sigma_1 produkcja/charakteryzacja: {g.mean() / s.mean():.3f}x")
    print(f"  odniesienie: SIGNAL_10DEG = {SIGNAL_10DEG} (udokumentowany zakres 0.0639-0.0662)")


if __name__ == "__main__":
    main()
