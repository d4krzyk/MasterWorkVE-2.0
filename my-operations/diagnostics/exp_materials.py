"""Weryfikacja configu materialow akustycznych Repliki."""

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Kolejnosc importow (quaternion przed habitat_sim) wymagana przez lokalny patch
# tego repo - patrz habitat-sim/local_changes.patch / CLAUDE.md.
import quaternion  # noqa: F401
import habitat_sim

from echo_core.audio import build_simulator, phase3_echolocation
from echo_core.spectrogram import (ECHO_MS, ECHO_SAMPLES, EXPECTED_SPEC_SHAPE, SAMPLE_RATE,
                                   STFT_HOP_LENGTH, STFT_N_FFT, STFT_WIN_LENGTH,
                                   render_spectrogram)

from .common import (CHIRP_PATH, MATERIAL_CONFIG_PATH, OUT_DIR, REPLICA_MATERIAL_CONFIG, _get_spec, _rmse, build_sim, load_point_position)


# --- BLOK B.4: weryfikacja configu materialow dla Repliki --------------------
#
# Sam fakt, ze nowy config sie wczytuje i nie loguje ostrzezen, NIE dowodzi, ze
# cokolwiek robi - sciezka materialow moglaby byc po cichu pominieta i wygladaloby
# to tak samo. Dlatego trzy niezalezne sprawdzenia:
#
#  1. POKRYCIE   - ile kategorii lduje na materiale domyslnym przed i po. Liczone
#                  z faktycznych ostrzezen warstwy C++, nie z symulacji reguly.
#  2. KONTROLA POZYTYWNA - czy zamiana configu MIERZALNIE zmienia echo. Jesli
#                  roznica tonie w szumie renderowania, to config nie dziala i
#                  trzeba to zglosic, a nie tlumaczyc szumem.
#  3. KONTROLA NEGATYWNA - config absurdalny (wszystko "Sound Proof", absorpcja
#                  1.0 na kazdej czestotliwosci) musi zmienic echo drastycznie.
#                  To dowodzi, ze przypisanie materialu w ogole dochodzi do
#                  symulatora, a nie tylko ze plik JSON zostal sparsowany.

# Pozycje podane per scena, bo points.txt maja rozna dlugosc (room_0: 136,
# office_0: 65 wierszy) - wspolna trojka (30,50,80) wychodzila poza zakres office_0.
MATVERIFY_POSITIONS = (("room_0", 30), ("room_0", 50), ("room_0", 80),
                       ("office_0", 10), ("office_0", 30), ("office_0", 55))

MATVERIFY_ANGLES = (0.0, 90.0)

MATVERIFY_N = 8  # renderow na polowke; 2*N na (config, pozycja, kat)

MATVERIFY_SCENES = ("room_0", "office_0")

def _capture_native_output(path):
    """Przechwytuje wyjscie warstwy C++ na poziomie deskryptorow.

    habitat-sim i RLRAudioPropagation pisza prosto do fd 1/2, wiec przekierowanie
    sys.stdout w Pythonie ich nie lapie. Bez tego nie da sie POLICZYC ostrzezen
    "Material for category ... was not found" osobno dla kazdego configu.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for f in (fd, saved_out, saved_err):
            os.close(f)

def _make_soundproof_config(dst):
    """Kontrola negatywna: KAZDA kategoria dostaje material o absorpcji 1.0.

    Budowany z naszego configu Repliki, zeby roznil sie od niego wylacznie
    fizyka materialu, a nie zestawem etykiet. Plik tymczasowy - nie commitujemy.
    """
    cfg = json.loads(REPLICA_MATERIAL_CONFIG.read_text())
    all_labels = []
    for m in cfg["materials"]:
        if m["name"] != "Default":
            all_labels.extend(m["labels"])
    for m in cfg["materials"]:
        m["labels"] = all_labels if m["name"] == "Sound Proof" else []
    cfg["materials"] = [m for m in cfg["materials"] if m["name"] in ("Default", "Sound Proof")]
    dst.write_text(json.dumps(cfg, indent=1))
    return dst

def run_materials_verify():
    print("\n=== BLOK B.4: weryfikacja replica_material_config.json ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    tmp_dir = OUT_DIR / "material_verify"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    soundproof = _make_soundproof_config(tmp_dir / "_soundproof_TEMP.json")

    configs = {"mp3d": MATERIAL_CONFIG_PATH, "replica": REPLICA_MATERIAL_CONFIG, "soundproof": soundproof}
    result = {"configs": {k: str(v) for k, v in configs.items()}, "scenes": list(MATVERIFY_SCENES),
              "positions": [{"scene": s, "point_id": p} for s, p in MATVERIFY_POSITIONS],
              "angles": list(MATVERIFY_ANGLES), "n_per_half": MATVERIFY_N}

    per_scene = []
    for scene in MATVERIFY_SCENES:
        print(f"\n--- {scene} ---")
        scene_pids = [p for s, p in MATVERIFY_POSITIONS if s == scene]
        est, warns = {}, {}
        for cname, cpath in configs.items():
            log = tmp_dir / f"native_{scene}_{cname}.log"
            # Renderowanie pod przechwyconymi deskryptorami; zadnych printow w srodku.
            with _capture_native_output(log):
                sim = build_sim(scene_name=scene, material_config=cpath)
                try:
                    for pid in scene_pids:
                        pos = load_point_position(sim, scene, pid)
                        for ang in MATVERIFY_ANGLES:
                            specs = [_get_spec(sim, pos, ang, str(cpath), chirp) for _ in range(2 * MATVERIFY_N)]
                            est[(cname, pid, ang, "A")] = np.mean(specs[:MATVERIFY_N], axis=0)
                            est[(cname, pid, ang, "B")] = np.mean(specs[MATVERIFY_N:], axis=0)
                finally:
                    sim.close()
            text = log.read_text(errors="replace")
            cats = sorted(set(re.findall(r"Material for category '([^']*)' was not found", text)))
            warns[cname] = cats
            print(f"  {cname:<11}: kategorii na materiale domyslnym = {len(cats)}"
                  + (f"  {cats}" if cats else ""))

        keys = [(pid, ang) for pid in scene_pids for ang in MATVERIFY_ANGLES]
        full = {c: {k: (est[(c, k[0], k[1], "A")] + est[(c, k[0], k[1], "B")]) / 2.0 for k in keys} for c in configs}

        # Szum renderowania: rozrzut dwoch niezaleznych polowek TEGO SAMEGO configu.
        # RMSE(A,B) = sqrt(2)*sigma_N, gdzie sigma_N to szum estymaty z N renderow.
        # Estymata "full" ma 2N renderow, wiec jej szum to sigma_N/sqrt(2) = noise/2.
        noise = float(np.mean([_rmse(est[("replica", p, a, "A")], est[("replica", p, a, "B")]) for p, a in keys]))
        sigma_full = noise / 2.0
        # Szum SAMEJ ENERGII (sredniej spektrogramu) miedzy polowkami. Energia jest
        # srednia po ~85 tys. komorek, wiec jej szum jest o rzedy wielkosci mniejszy
        # niz RMSE - to znacznie czulsza statystyka do stwierdzenia "config dziala".
        e_noise = float(np.mean([abs(np.mean(est[("replica", p, a, "A")]) - np.mean(est[("replica", p, a, "B")]))
                                 / np.mean(full["replica"][(p, a)]) for p, a in keys]))

        def _cmp(c1, c2):
            gap = float(np.mean([_rmse(full[c1][k], full[c2][k]) for k in keys]))
            # RMSE(c1,c2)^2 = efekt^2 + 2*sigma_full^2 - odejmujemy szum obu estymat,
            # inaczej porownywalibysmy wielkosc zaszumiona z szumem (blad progu w
            # pierwszej wersji tego testu).
            effect = float(np.sqrt(max(gap ** 2 - 2 * sigma_full ** 2, 0.0)))
            e1 = float(np.mean([np.mean(full[c1][k]) for k in keys]))
            e2 = float(np.mean([np.mean(full[c2][k]) for k in keys]))
            return gap, effect, (e1 - e2) / e2

        pos_gap, pos_effect, pos_energy = _cmp("replica", "mp3d")
        neg_gap, neg_effect, neg_energy = _cmp("soundproof", "replica")
        row = {"scene": scene, "render_noise": noise, "sigma_full": sigma_full,
               "energy_noise_rel": e_noise,
               "default_categories": {c: warns[c] for c in configs},
               "positive_gap": pos_gap, "positive_effect": pos_effect, "positive_energy_rel": pos_energy,
               "positive_effect_vs_noise": pos_effect / sigma_full,
               "positive_energy_vs_noise": abs(pos_energy) / e_noise if e_noise > 0 else float("inf"),
               "negative_gap": neg_gap, "negative_effect": neg_effect, "negative_energy_rel": neg_energy,
               "negative_effect_vs_noise": neg_effect / sigma_full}
        per_scene.append(row)
        print(f"  szum estymaty (2N={2 * MATVERIFY_N} renderow) = {sigma_full:.5f} RMSE, "
              f"szum energii = {e_noise * 100:.3f}%")
        print(f"  KONTROLA POZYTYWNA replica vs mp3d       : roznica surowa {pos_gap:.5f} -> "
              f"EFEKT {pos_effect:.5f} ({pos_effect / sigma_full:.1f}x szum)")
        print(f"      energia {pos_energy * 100:+.2f}% ({abs(pos_energy) / e_noise:.0f}x szum energii)")
        print(f"  KONTROLA NEGATYWNA soundproof vs replica : EFEKT {neg_effect:.5f} "
              f"({neg_effect / sigma_full:.1f}x szum), energia {neg_energy * 100:+.1f}%")

    result["per_scene"] = per_scene
    m_pos = float(np.mean([r["positive_effect_vs_noise"] for r in per_scene]))
    m_pos_e = float(np.mean([r["positive_energy_vs_noise"] for r in per_scene]))
    m_neg = float(np.mean([r["negative_effect_vs_noise"] for r in per_scene]))
    n_mp3d = int(np.mean([len(r["default_categories"]["mp3d"]) for r in per_scene]))
    n_repl = int(np.mean([len(r["default_categories"]["replica"]) for r in per_scene]))
    same_sign = len({np.sign(r["positive_energy_rel"]) for r in per_scene}) == 1
    result["summary"] = {"positive_effect_vs_noise": m_pos, "positive_energy_vs_noise": m_pos_e,
                         "negative_effect_vs_noise": m_neg, "energy_shift_same_sign": bool(same_sign),
                         "default_categories_mp3d": n_mp3d, "default_categories_replica": n_repl}

    print("\n--- WERDYKT BLOK B.4 ---")
    print(f"  kategorii na materiale domyslnym: mp3d={n_mp3d} -> replica={n_repl}")
    print(f"  kontrola pozytywna: efekt {m_pos:.1f}x szum estymaty, energia {m_pos_e:.0f}x szum energii, "
          f"zgodny znak we wszystkich scenach: {same_sign}")
    print(f"  kontrola negatywna: efekt {m_neg:.1f}x szum estymaty")
    # Kryterium glowne to ENERGIA, nie RMSE: energia jest srednia po ~85 tys. komorek
    # spektrogramu, wiec jej szum jest o rzedy wielkosci mniejszy, a systematyczny
    # wzrost energii jest dokladnie tym, czego oczekujemy po utwardzeniu sufitu i
    # podlogi. RMSE sluzy jako miara WIELKOSCI zmiany, nie jako test jej istnienia.
    if m_pos_e < 5.0 or not same_sign:
        verdict = ("BLAD - podmiana configu nie zmienia energii echa w sposob systematyczny. "
                   "Config NIE jest stosowany (albo sciezka materialow jest pominieta); "
                   "nie wolno tego tlumaczyc szumem.")
    elif m_neg < 3.0:
        verdict = ("BLAD - kontrola negatywna (wszystko soundproof) nie zmienia echa drastycznie, "
                   "wiec przypisanie materialu nie dochodzi do symulatora.")
    else:
        verdict = ("OK - config jest stosowany: energia echa przesuwa sie systematycznie i zgodnie "
                   "co do znaku we wszystkich scenach, wielokrotnie ponad wlasny szum, a absurdalny "
                   "config zmienia echo drastycznie. Nowe mapowanie dziala.")
    result["verdict"] = verdict
    print(f"  {verdict}")
    soundproof.unlink(missing_ok=True)  # plik tymczasowy, nie commitujemy
    return result
