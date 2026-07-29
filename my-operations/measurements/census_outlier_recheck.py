#!/usr/bin/env python3
"""DOWOD: ktore przekroczenia N_MAX z census sa prawdziwe, a ktore to szum sondy.

Pytanie: census sonduje kazda lokalizacje RAZ, estymatorem o SD ~5 %. Siedem
lokalizacji przekroczylo N_MAX=40 — ile z nich to realne gorace miejsca?
Dodatkowo: pierwsza sondowana lokalizacja sceny wypadala systematycznie glosniej
(mediana percentyla 92 %, Wilcoxon p=0.001) — rozgrzewka czy efekt przestrzenny?

Metoda: te same lokalizacje mierzone estymatorem wariancyjnym (M=40, SD ~1 %)
GLEBOKO w stanie ustalonym (po 120 renderach w tej samej instancji). Jesli sigma
spada — rozgrzewka. Jesli zostaje — efekt realny.

Wynik (2026-07-29): 4 z 7 potwierdzone, wszystkie w apartment_0 przy sasiadujacych
loc_id (285, 307, 308, 310) — jedno realne skupisko. Maksimum: sigma_1 = 0.12799
(apartment_0/285) -> N_raw = 49. Trzy odpadly (frl_apartment_2/0: 54 -> 34).
Pierwsze lokalizacje scen: efekt w wiekszosci PRZESTRZENNY (frl_apartment_3/0ma
0.09444 w stanie ustalonym przy medianie sceny 0.04550), rozgrzewka dokladala
kilka procent.

WYMAGA: outputs/probe_census/_all_scenes.csv (z --probe-only)

Raport: RAPORT_SESJI_2026-07-26_29.md §2.5, §3.4 | Dokument: GENERATOR_PARAMS.md §3.2
"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_core import audio, spectrogram
from echo_core.params import (INDIRECT_RAY_COUNT, N_MAX, N_MIN, N_PROBE, SENSOR_HEIGHT,
                              SIGNAL_10DEG, TARGET_SNR, THREAD_COUNT, WARMUP_DISCARD)
from echo_core.paths import CHIRP_PATH, MATERIAL_CONFIG, OUT_ROOT, probe_census_csv, scene_mesh
from echo_core.renderer import Renderer
from echo_core.runtime import setup_logging
from echo_core.scenes import load_scene_locations
from echo_core.store import SPEC_SHAPE
import librosa

M = 40
SETTLE = 100          # renderow na neutralnej pozycji PO rozgrzewce, zeby byc gleboko w stanie ustalonym

# (scena, [lokalizacje do sprawdzenia], lokalizacja neutralna do rozgrzewki/osiadania)
CASES = [
    ("hotel_0", [101, 95, 10], 70),          # 101/95 = outliery, 10 = 1. w scenie
    ("apartment_0", [307, 285, 308, 310, 4], 184),   # 4 outliery + 4 = 1. w scenie
]

chirp, _ = librosa.load(str(CHIRP_PATH), sr=spectrogram.SAMPLE_RATE, mono=True)
MC = str(MATERIAL_CONFIG)
census = {}
import csv as _csv
with open(Path(__file__).resolve().parents[2] / "outputs/probe_census/_all_scenes.csv", newline="") as f:
    for r in _csv.DictReader(f):
        census[(r["scene"], int(r["loc_id"]))] = float(r["sigma_1"])


def sigma_var(specs):
    return float(np.sqrt(np.var(np.stack(specs).astype(np.float64), axis=0, ddof=1).mean()))


def sigma_half8(specs):
    """Dokladnie estymator sondy: 8 renderow, podzial 4+4."""
    a = np.mean(np.stack(specs[:4]), axis=0, dtype=np.float32)
    b = np.mean(np.stack(specs[4:8]), axis=0, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(2.0) * 2.0)


rows, n_sims = [], 0
for scene, targets, neutral in CASES:
    _ids, positions = load_scene_locations(scene)
    r = Renderer(scene, setup_logging(log_path=None))
    n_sims += 1
    print(f"\n[{n_sims}/{len(CASES)}] {scene}: rozgrzewka + {SETTLE} renderow osiadania "
          f"na id={neutral}, potem cele {targets}", flush=True)
    try:
        r.warmup(positions[neutral])
        for _ in range(SETTLE):
            r.render(positions[neutral], 0.0)
        for loc in targets:
            specs = [r.render(positions[loc], 0.0)[0] for _ in range(M)]
            sv, s8 = sigma_var(specs), sigma_half8(specs)
            c = census[(scene, loc)]
            n_var = int(np.ceil((TARGET_SNR * sv / SIGNAL_10DEG) ** 2))
            n_cen = int(np.ceil((TARGET_SNR * c / SIGNAL_10DEG) ** 2))
            rows.append(dict(scene=scene, loc=loc, census=c, var=sv, half8=s8,
                             n_census=n_cen, n_var=n_var))
            print(f"    id={loc:<5} census={c:.5f} (N={n_cen:<3})  "
                  f"teraz: wariancyjny={sv:.5f} (N={n_var:<3})  polowkowy8={s8:.5f}  "
                  f"zmiana {100*(sv/c-1):+.1f} %", flush=True)
    finally:
        r.close()

print(f"\n{'='*104}\n  WERDYKT\n{'='*104}")
print(f"  {'scena':<18}{'lok':>6}{'census':>10}{'ustalony':>11}{'zmiana':>10}"
      f"{'N census':>10}{'N ustal.':>10}{'1. w scenie':>13}")
firsts = {"frl_apartment_2": 0, "frl_apartment_3": 0, "frl_apartment_5": 0,
          "hotel_0": 10, "apartment_0": 4}
for d in rows:
    is_first = "TAK" if firsts.get(d["scene"]) == d["loc"] else ""
    print(f"  {d['scene']:<18}{d['loc']:>6}{d['census']:>10.5f}{d['var']:>11.5f}"
          f"{100*(d['var']/d['census']-1):>9.1f}%{d['n_census']:>10}{d['n_var']:>10}{is_first:>13}")

f_rows = [d for d in rows if firsts.get(d["scene"]) == d["loc"]]
o_rows = [d for d in rows if firsts.get(d["scene"]) != d["loc"]]
fd = np.array([d["var"] / d["census"] - 1 for d in f_rows])
od = np.array([d["var"] / d["census"] - 1 for d in o_rows])
print(f"\n  PIERWSZE lokalizacje scen ({len(fd)}): zmiana mediana {100*np.median(fd):+.1f} %, "
      f"srednia {100*fd.mean():+.1f} %")
print(f"  POZOSTALE ({len(od)}):              zmiana mediana {100*np.median(od):+.1f} %, "
      f"srednia {100*od.mean():+.1f} %")
print(f"\n  Jesli pierwsze spadaja wyraznie, a pozostale nie -> rozgrzewka (H1).")
print(f"  Jesli obie grupy stoja w miejscu -> efekt przestrzenny, outliery prawdziwe (H2).")
print(f"\n  Konstrukcji Simulatora: {n_sims}")
