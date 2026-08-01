"""Podloga szumu: wysokosc sluchacza, kontrola sygnalu, podloga per scena,
zaleznosc od orientacji i census scen niezmierzonych."""

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
# tego repo - patrz habitat-sim/local_changes.patch.
import quaternion  # noqa: F401
import habitat_sim

from echo_core.audio import build_simulator, phase3_echolocation
from echo_core.spectrogram import (ECHO_MS, ECHO_SAMPLES, EXPECTED_SPEC_SHAPE, SAMPLE_RATE,
                                   STFT_HOP_LENGTH, STFT_N_FFT, STFT_WIN_LENGTH,
                                   render_spectrogram)

from .common import (CHIRP_PATH, MATERIALS_ENABLED, MATERIAL_CONFIG_PATH, PRODUCTION_SENSOR_HEIGHT, REMAINING_ANGLE, REMAINING_FRACTIONS, REMAINING_M, REPLICA_MATERIAL_CONFIG, REPORT_PATH, REPO_ROOT, _get_spec, _material_config_arg, _points_path, _rmse, _sigma_variance, build_sim, load_point_position)


# --- Wysokosc sluchacza: 1.25 m (kamera z pkl) czy 1.5 m (konwencja SoundSpaces)? ---
#
# Kamera odtworzona z scene_observations_128.pkl siedzi na 1.25 m, a AudioSensorSpec
# w build_simulator() na 1.5 m (ta sama wartosc, ktorej uzywa
# sound-spaces/soundspaces/continuous_simulator.py). To 25 cm rozjazdu miedzy
# punktem obserwacji wizualnej i akustycznej - dzis jest to przypadkowy zbieg
# dwoch roznych domyslnych wartosci, a powinno byc decyzja.
#
# Kryterium rozstrzygajace: porownac roznice miedzy wysokosciami z DWOMA skalami
# odniesienia, ktore juz znamy - szumem resztkowym (czego nie da sie odroznic) i
# sygnalem 10 stopni (tym, co caly projekt probuje zmierzyc). Jesli 25 cm zmienia
# echo bardziej niz 10 stopni obrotu, to wybor wysokosci jest konsekwentny i musi
# byc swiadomy; jesli tonie w szumie - jest dowolny.

HEIGHT_SENSOR_OFFSET = 1.5  # offset audio w AudioSensorSpec (build_simulator)

HEIGHT_CANDIDATES = (1.25, 1.5)

HEIGHT_N = 10

HEIGHT_ANGLES = (0.0, 10.0)

HEIGHT_POSITION_IDS = (30, 50, 80)

def run_listener_height():
    print("\n=== Wysokosc sluchacza: 1.25 m (kamera pkl) vs 1.5 m (konwencja SoundSpaces) ===")
    import librosa

    material_config = _material_config_arg()
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    result = {"materials_enabled": MATERIALS_ENABLED, "candidates": list(HEIGHT_CANDIDATES),
              "n_per_estimate": HEIGHT_N, "angles": list(HEIGHT_ANGLES),
              "position_ids": list(HEIGHT_POSITION_IDS)}

    rows = []
    sim = build_sim(scene_name="room_0")
    try:
        for pid in HEIGHT_POSITION_IDS:
            pos = load_point_position(sim, "room_0", pid)
            print(f"\n--- pozycja id={pid} ---")
            est, noise_parts = {}, []
            for h in HEIGHT_CANDIDATES:
                # Sensor audio ma staly offset +1.5 m wzgledem agenta, wiec zamiast
                # rekonstruowac Simulator dla kazdej wysokosci przesuwamy AGENTA -
                # sluchacz laduje wtedy na zadanej wysokosci nad tym samym punktem.
                pos_h = np.array(pos, dtype=np.float64)
                pos_h[1] += h - HEIGHT_SENSOR_OFFSET
                for ang in HEIGHT_ANGLES:
                    specs = [_get_spec(sim, pos_h, ang, material_config, chirp) for _ in range(2 * HEIGHT_N)]
                    est[(h, ang, "A")] = np.mean(specs[:HEIGHT_N], axis=0)
                    est[(h, ang, "B")] = np.mean(specs[HEIGHT_N:], axis=0)
                    noise_parts.append(_rmse(est[(h, ang, "A")], est[(h, ang, "B")]))

            h1, h2 = HEIGHT_CANDIDATES
            a0, a1 = HEIGHT_ANGLES
            noise = float(np.mean(noise_parts))
            signal_10deg = float(np.mean([_rmse(est[(h, a0, "A")], est[(h, a1, "A")]) for h in HEIGHT_CANDIDATES]))
            height_effect = float(np.mean([_rmse(est[(h1, a, "A")], est[(h2, a, "A")]) for a in HEIGHT_ANGLES]))
            row = {"point_id": pid, "noise": noise, "signal_10deg": signal_10deg,
                   "height_effect": height_effect,
                   "height_vs_noise": height_effect / noise if noise > 0 else float("inf"),
                   "height_vs_10deg": height_effect / signal_10deg if signal_10deg > 0 else float("inf")}
            rows.append(row)
            print(f"  szum resztkowy (N={HEIGHT_N})      = {noise:.5f}")
            print(f"  sygnal 10 stopni                  = {signal_10deg:.5f}")
            print(f"  efekt zmiany wysokosci 1.25<->1.5 = {height_effect:.5f}  "
                  f"({row['height_vs_noise']:.1f}x szum, {row['height_vs_10deg']:.2f}x sygnal 10 stopni)")
    finally:
        sim.close()

    result["per_position"] = rows
    m_effect = float(np.mean([r["height_effect"] for r in rows]))
    m_noise = float(np.mean([r["noise"] for r in rows]))
    m_signal = float(np.mean([r["signal_10deg"] for r in rows]))
    result["height_effect_mean"] = m_effect
    result["noise_mean"] = m_noise
    result["signal_10deg_mean"] = m_signal
    result["height_vs_noise"] = m_effect / m_noise
    result["height_vs_10deg"] = m_effect / m_signal

    print("\n--- WERDYKT: wysokosc sluchacza ---")
    print(f"  efekt 25 cm = {m_effect:.5f} | szum = {m_noise:.5f} | sygnal 10 stopni = {m_signal:.5f}")
    print(f"  efekt wysokosci to {m_effect / m_noise:.1f}x szum resztkowy i {m_effect / m_signal:.2f}x sygnal 10 stopni")
    if m_effect < m_noise:
        verdict = ("NIEISTOTNA - roznica miedzy 1.25 a 1.5 m tonie w szumie resztkowym, "
                   "wiec wybor jest dowolny; i tak warto zrownac dla spojnosci opisu")
    elif m_effect < m_signal:
        verdict = ("ISTOTNA, ale mniejsza niz mierzony sygnal - 25 cm zmienia echo zauwazalnie ponad szum, "
                   "choc slabiej niz 10 stopni obrotu; wybor musi byc swiadomy i udokumentowany")
    else:
        verdict = ("KRYTYCZNA - 25 cm zmienia echo BARDZIEJ niz 10 stopni obrotu, czyli bardziej niz efekt, "
                   "ktory caly projekt probuje zmierzyc; nie wolno zostawic tego przypadkowi")
    result["verdict"] = verdict
    # Rekomendacja jest ta sama niezaleznie od skali efektu: agent ucieleśniony
    # widzi i slyszy z jednego punktu, a jedyna wartosc twardo narzucona z
    # zewnatrz to 1.25 m (wymog odtworzenia pkl) - wiec to audio powinno sie
    # dopasowac do kamery, nie odwrotnie.
    result["recommendation"] = 1.25
    print(f"  {verdict}")
    print("  Rekomendacja: audio na 1.25 m, zrownane z kamera. 1.25 m jest twardo narzucone przez")
    print("  odtworzenie pkl, a 1.5 m to tylko konwencja SoundSpaces - to audio ma sie dopasowac.")
    return result

# --- BLOK C: przemiar sygnalu i szumu na FINALNYCH materialach ---------------
#
# Cala dotychczasowa charakteryzacja (sygnal 10 stopni = 0.064, podloga szumu
# 0.03-0.16, SNR ~3 przy N=10) byla mierzona na configu mp3d, w ktorym sufit mial
# absorpcje 0.60 przy 500 Hz, a podloga 0.65 przy 4 kHz. Nowy config Repliki
# zmienia oba na twarde (0.05 i 0.07), czyli WYDLUZA pogłos. Dluzszy ogon to
# wiecej odbic posrednich w oknie 60 ms, a odbicia posrednie sa jedynym zrodlem
# szumu Monte Carlo - wiec i sygnal, i szum moga sie przesunac, a od ich stosunku
# zalezy dobor N dla produkcji. Bez tego przemiaru przenosilibysmy do generatora
# liczbe N wyznaczona dla innej akustyki.
#
# Mierzymy oba configi w jednym przebiegu, zeby porownanie bylo bezposrednie.

SNR_POSITIONS = (("room_0", 30), ("room_0", 50), ("room_0", 80), ("office_0", 30))

SNR_ANGLES = (0.0, 10.0)

SNR_N = 10          # renderow na polowke (2*N na kat) - jak w E2/E3

SNR_TARGET = 3.0    # docelowy stosunek sygnal/szum estymaty

def run_signal_noise_recheck():
    print("\n=== BLOK C: sygnal 10 stopni i szum na finalnym configu materialow ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    configs = {"mp3d": MATERIAL_CONFIG_PATH, "replica": REPLICA_MATERIAL_CONFIG}
    result = {"configs": {k: str(v) for k, v in configs.items()}, "n_per_half": SNR_N,
              "angles": list(SNR_ANGLES), "target_snr": SNR_TARGET,
              "positions": [{"scene": s, "point_id": p} for s, p in SNR_POSITIONS],
              "historical_signal_10deg": 0.064}

    scenes = sorted({s for s, _ in SNR_POSITIONS})
    rows = []
    for cname, cpath in configs.items():
        for scene in scenes:
            pids = [p for s, p in SNR_POSITIONS if s == scene]
            sim = build_sim(scene_name=scene, material_config=cpath)
            try:
                for pid in pids:
                    pos = load_point_position(sim, scene, pid)
                    est = {}
                    for ang in SNR_ANGLES:
                        specs = [_get_spec(sim, pos, ang, str(cpath), chirp) for _ in range(2 * SNR_N)]
                        est[(ang, "A")] = np.mean(specs[:SNR_N], axis=0)
                        est[(ang, "B")] = np.mean(specs[SNR_N:], axis=0)
                    a0, a1 = SNR_ANGLES
                    # RMSE(A,B) = sqrt(2)*sigma_N, gdzie sigma_N to szum estymaty z N renderow
                    noise_ab = float(np.mean([_rmse(est[(a, "A")], est[(a, "B")]) for a in SNR_ANGLES]))
                    sigma_n = noise_ab / np.sqrt(2.0)
                    sigma_1 = sigma_n * np.sqrt(SNR_N)          # szum POJEDYNCZEGO renderu
                    raw = float(np.mean([_rmse(est[(a0, h)], est[(a1, h)]) for h in ("A", "B")]))
                    # raw^2 = sygnal^2 + 2*sigma_N^2  ->  odszumiony sygnal
                    signal = float(np.sqrt(max(raw ** 2 - noise_ab ** 2, 0.0)))
                    snr = signal / sigma_n if sigma_n > 0 else float("inf")
                    n_needed = int(np.ceil((SNR_TARGET * sigma_1 / signal) ** 2)) if signal > 0 else -1
                    energy = float(np.mean([np.mean(est[(a, "A")]) for a in SNR_ANGLES]))
                    rows.append({"config": cname, "scene": scene, "point_id": pid,
                                 "noise_halfsplit": noise_ab, "sigma_single_render": sigma_1,
                                 "raw_10deg": raw, "signal_10deg": signal, "snr_at_N": snr,
                                 "n_for_target_snr": n_needed, "energy": energy})
                    print(f"  [{cname:<7}] {scene:<9} id={pid:<3} szum(N={SNR_N})={noise_ab:.5f} "
                          f"szum 1 renderu={sigma_1:.5f} | sygnal 10 st.={signal:.5f} "
                          f"| SNR={snr:.2f} | N dla SNR {SNR_TARGET:.0f} = {n_needed}")
            finally:
                sim.close()

    result["per_position"] = rows

    def _agg(cname, key):
        return float(np.mean([r[key] for r in rows if r["config"] == cname]))

    summary = {}
    for cname in configs:
        summary[cname] = {k: _agg(cname, k) for k in
                          ("noise_halfsplit", "sigma_single_render", "signal_10deg", "snr_at_N", "energy")}
        summary[cname]["n_for_target_snr"] = int(max(r["n_for_target_snr"] for r in rows if r["config"] == cname))
    result["summary"] = summary

    s_old, s_new = summary["mp3d"], summary["replica"]
    print("\n--- WERDYKT BLOK C ---")
    print(f"{'':<22}{'mp3d':>12}{'replica':>12}{'zmiana':>12}")
    for k, label in (("signal_10deg", "sygnal 10 st."), ("sigma_single_render", "szum 1 renderu"),
                     ("noise_halfsplit", f"szum estymaty N={SNR_N}"), ("snr_at_N", f"SNR przy N={SNR_N}"),
                     ("energy", "energia")):
        chg = (s_new[k] - s_old[k]) / s_old[k] * 100 if s_old[k] else float("nan")
        print(f"{label:<22}{s_old[k]:>12.5f}{s_new[k]:>12.5f}{chg:>11.1f}%")
    print(f"{'N dla SNR ' + str(int(SNR_TARGET)):<22}{s_old['n_for_target_snr']:>12d}{s_new['n_for_target_snr']:>12d}")

    sig_shift = abs(s_new["signal_10deg"] - s_old["signal_10deg"]) / s_old["signal_10deg"]
    if sig_shift < 0.10 and s_new["n_for_target_snr"] <= s_old["n_for_target_snr"]:
        verdict = ("BEZ ZMIAN - nowe materialy nie przesuwaja istotnie ani sygnalu 10 stopni, ani "
                   "podlogi szumu; dotychczasowy dobor N pozostaje wazny.")
    else:
        verdict = (f"PRZESUNIETE - sygnal 10 stopni zmienil sie o {sig_shift * 100:.0f}%, a wymagane N "
                   f"dla SNR {SNR_TARGET:.0f} wynosi teraz {s_new['n_for_target_snr']} (bylo "
                   f"{s_old['n_for_target_snr']}). Liczby szumu z wczesniejszej charakteryzacji nalezy "
                   "cytowac jako dotyczace configu mp3d, a do generatora wziac wartosci z tego wpisu.")
    result["verdict"] = verdict
    print(f"\n  {verdict}")
    print("  UWAGA: pomiar przy indirectRayCount=500. Produkcja ma isc na 5000-10000, gdzie szum")
    print("  Monte Carlo jest nizszy - wyliczone N jest wiec gornym ograniczeniem, nie wartoscia docelowa.")
    return result

# --- BLOK 3: podloga szumu na kolejnych scenach ------------------------------
#
# N dobieramy pod NAJGORSZA scene, nie pod srednia, a scharakteryzowane mamy 4 z 18
# (room_0, apartment_1, office_0, frl_apartment_0). office_0 juz wymagal N=9 zamiast 7.
#
# Priorytet: najpierw sceny HELD-OUT (apartment_2, frl_apartment_5, office_4), bo to
# z nich pochodza finalne liczby pracy, wiec ich charakterystyka szumu wazy wiecej niz
# treningowych. office_4 dodatkowo ma najgorsze pokrycie materialowe (9.4% powierzchni
# to class_id=-1, ktore zawsze dostaje material domyslny - patrz REPLICA_MATERIALS.md).
# hotel_0 jako czwarta, bo to kategoria sceny nieobecna w dotychczasowej czworce.

NOISEFLOOR_SCENES = ("apartment_2", "frl_apartment_5", "office_4", "hotel_0")

NOISEFLOOR_ANGLES = (0.0, 10.0)

NOISEFLOOR_N = 10

NOISEFLOOR_TARGET_SNR = 3.5

# Ulamki dlugosci points.txt zamiast stalych id - sceny maja rozna liczbe punktow
# (65-258), a chodzi o dwie pozycje ODLEGLE od siebie, nie o konkretne indeksy.
NOISEFLOOR_FRACTIONS = (0.20, 0.75)

def run_noise_floor_scenes():
    print("\n=== BLOK 3: podloga szumu na scenach spoza dotychczasowej czworki ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    rays = int(os.environ.get("VE_RAYS", "500"))
    threads = int(os.environ.get("VE_THREADS", "1"))
    result = {"scenes": list(NOISEFLOOR_SCENES), "angles": list(NOISEFLOOR_ANGLES),
              "n_per_half": NOISEFLOOR_N, "target_snr": NOISEFLOOR_TARGET_SNR,
              "rays": rays, "threads": threads, "sensor_height": PRODUCTION_SENSOR_HEIGHT,
              "material_config": mc, "position_fractions": list(NOISEFLOOR_FRACTIONS)}

    rows = []
    for scene in NOISEFLOOR_SCENES:
        n_points = sum(1 for _ in _points_path(scene).open())
        pids = [int(f * n_points) for f in NOISEFLOOR_FRACTIONS]
        print(f"\n--- {scene} ({n_points} lokalizacji, pozycje {pids}) ---")
        sim = build_sim(scene_name=scene, indirect_ray_count=rays, thread_count=threads,
                        material_config=REPLICA_MATERIAL_CONFIG,
                        sensor_height=PRODUCTION_SENSOR_HEIGHT)
        try:
            for pid in pids:
                pos = load_point_position(sim, scene, pid)
                est = {}
                for ang in NOISEFLOOR_ANGLES:
                    specs = [_get_spec(sim, pos, ang, mc, chirp) for _ in range(2 * NOISEFLOOR_N)]
                    est[(ang, "A")] = np.mean(specs[:NOISEFLOOR_N], axis=0)
                    est[(ang, "B")] = np.mean(specs[NOISEFLOOR_N:], axis=0)
                a0, a1 = NOISEFLOOR_ANGLES
                # RMSE(A,B) = sqrt(2)*sigma_N; sigma_1 = sigma_N*sqrt(N)
                noise_ab = float(np.mean([_rmse(est[(a, "A")], est[(a, "B")]) for a in NOISEFLOOR_ANGLES]))
                sigma_n = noise_ab / np.sqrt(2.0)
                sigma_1 = sigma_n * np.sqrt(NOISEFLOOR_N)
                raw = float(np.mean([_rmse(est[(a0, h)], est[(a1, h)]) for h in ("A", "B")]))
                signal = float(np.sqrt(max(raw ** 2 - noise_ab ** 2, 0.0)))
                snr = signal / sigma_n if sigma_n > 0 else float("inf")
                n_needed = int(np.ceil((NOISEFLOOR_TARGET_SNR * sigma_1 / signal) ** 2)) if signal > 0 else -1
                rows.append({"scene": scene, "point_id": pid, "n_points": n_points,
                             "noise_halfsplit": noise_ab, "sigma_single_render": sigma_1,
                             "raw_10deg": raw, "signal_10deg": signal, "snr_at_N": snr,
                             "n_for_target_snr": n_needed,
                             "energy": float(np.mean(est[(a0, "A")]))})
                print(f"  id={pid:<4} szum(N={NOISEFLOOR_N})={noise_ab:.5f} szum 1 renderu={sigma_1:.5f} "
                      f"| sygnal 10 st.={signal:.5f} | SNR={snr:.2f} "
                      f"| N dla SNR {NOISEFLOOR_TARGET_SNR} = {n_needed}")
        finally:
            sim.close()

    result["per_position"] = rows

    # --- WERDYKT: pokazujemy ROZKLAD, nie samo ekstremum ----------------------
    # Poprzednim razem automat wypisal "PRZESUNIETE" zdominowany przez jeden punkt,
    # podczas gdy srednie stały w miejscu - stad tu jawnie mediana obok maksimum.
    sig = np.array([r["signal_10deg"] for r in rows])
    s1 = np.array([r["sigma_single_render"] for r in rows])
    nn = np.array([r["n_for_target_snr"] for r in rows])
    per_scene = {}
    for scene in NOISEFLOOR_SCENES:
        sub = [r for r in rows if r["scene"] == scene]
        per_scene[scene] = {"signal_mean": float(np.mean([r["signal_10deg"] for r in sub])),
                            "sigma_single_mean": float(np.mean([r["sigma_single_render"] for r in sub])),
                            "snr_mean": float(np.mean([r["snr_at_N"] for r in sub])),
                            "n_required": int(max(r["n_for_target_snr"] for r in sub))}
    result["per_scene"] = per_scene
    result["summary"] = {"signal_median": float(np.median(sig)), "signal_min": float(sig.min()),
                         "signal_max": float(sig.max()),
                         "sigma_single_median": float(np.median(s1)), "sigma_single_max": float(s1.max()),
                         "n_median": int(np.median(nn)), "n_max": int(nn.max()),
                         "reference_signal": 0.0648, "reference_sigma_single": 0.05393}

    print("\n--- WERDYKT BLOK 3 ---")
    print(f"{'scena':<18}{'sygnal 10 st.':>15}{'szum 1 rend.':>14}{'SNR N=10':>10}{'N dla 3.5':>11}")
    for scene, v in per_scene.items():
        print(f"{scene:<18}{v['signal_mean']:>15.5f}{v['sigma_single_mean']:>14.5f}"
              f"{v['snr_mean']:>10.2f}{v['n_required']:>11d}")
    print(f"\n  sygnal 10 st.: mediana {np.median(sig):.5f}, zakres {sig.min():.5f}-{sig.max():.5f} "
          f"(odniesienie 0.0648)")
    print(f"  szum 1 renderu: mediana {np.median(s1):.5f}, max {s1.max():.5f} "
          f"(odniesienie 0.05393, znany zakres 0.03-0.16)")
    print(f"  N dla SNR {NOISEFLOOR_TARGET_SNR}: mediana {int(np.median(nn))}, MAKSIMUM {int(nn.max())}")
    in_range = bool(s1.max() <= 0.16 and sig.min() >= 0.04)
    result["within_known_range"] = in_range
    return result

# --- Czy podloga szumu zalezy od ORIENTACJI, czy tylko od POZYCJI? -----------
#
# Blok 3 pokazal, ze sygnal 10 stopni jest praktycznie stala (0.0639-0.0662 na 8
# pozycjach w 4 scenach), a caly rozrzut siedzi w szumie (0.031-0.080, czyli 2.5x).
# To uzasadnia adaptacyjne N: renderuj az szum spadnie ponizej sygnal/SNR_celu.
#
# Pytanie, ktore decyduje o KSZTALCIE tej reguly: czy szum jest wlasnoscia POZYCJI,
# czy tez zmienia sie takze miedzy orientacjami w tym samym punkcie?
#  - jesli jest wlasnoscia pozycji: N wyznaczamy RAZ na lokalizacje (1740 decyzji)
#    i stosujemy do wszystkich 36 orientacji. Decyzja podejmowana na jednej probce
#    i przenoszona na 36 praktycznie usuwa problem "optional stopping" (obciazenia
#    od zatrzymywania sie wtedy, gdy oszacowanie szumu akurat wypadnie nisko).
#  - jesli zmienia sie z orientacja: regula musi dzialac per probka (62 640 decyzji),
#    a wtedy trzeba jawnego marginesu na to obciazenie.
#
# Mierzymy na dwoch skrajnosciach z Bloku 3: najgorszej zmierzonej pozycji
# (hotel_0 id=76, wymagalo N=18) i najlepszej (frl_apartment_5 id=186, N=3).

ORIENT_POSITIONS = (("hotel_0", 76), ("frl_apartment_5", 186))

ORIENT_ANGLES = tuple(float(a) for a in range(0, 360, 10))

ORIENT_N = 10               # renderow na polowke

ORIENT_SIGNAL = 0.0644      # mediana sygnalu 10 st. z Bloku 3 (zakres 0.0639-0.0662)

ORIENT_TARGET_SNR = 3.5

def run_noise_floor_orientation():
    print("\n=== Czy podloga szumu zalezy od orientacji, czy tylko od pozycji? ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    result = {"positions": [{"scene": s, "point_id": p} for s, p in ORIENT_POSITIONS],
              "angles": len(ORIENT_ANGLES), "n_per_half": ORIENT_N,
              "signal_assumed": ORIENT_SIGNAL, "target_snr": ORIENT_TARGET_SNR,
              "sensor_height": PRODUCTION_SENSOR_HEIGHT, "rays": 500, "threads": 1}

    rows = []
    for scene, pid in ORIENT_POSITIONS:
        sim = build_sim(scene_name=scene, indirect_ray_count=500, thread_count=1,
                        material_config=REPLICA_MATERIAL_CONFIG,
                        sensor_height=PRODUCTION_SENSOR_HEIGHT)
        try:
            pos = load_point_position(sim, scene, pid)
            per_angle = []
            for ang in ORIENT_ANGLES:
                specs = [_get_spec(sim, pos, ang, mc, chirp) for _ in range(2 * ORIENT_N)]
                a = np.mean(specs[:ORIENT_N], axis=0)
                b = np.mean(specs[ORIENT_N:], axis=0)
                noise_ab = _rmse(a, b)                       # sqrt(2)*sigma_N
                sigma_1 = noise_ab / np.sqrt(2.0) * np.sqrt(ORIENT_N)
                n_req = int(np.ceil((ORIENT_TARGET_SNR * sigma_1 / ORIENT_SIGNAL) ** 2))
                per_angle.append({"angle": ang, "noise_halfsplit": noise_ab,
                                  "sigma_single_render": sigma_1, "n_required": n_req})
        finally:
            sim.close()

        s1 = np.array([p["sigma_single_render"] for p in per_angle])
        nr = np.array([p["n_required"] for p in per_angle])
        row = {"scene": scene, "point_id": pid, "per_angle": per_angle,
               "sigma_median": float(np.median(s1)), "sigma_min": float(s1.min()),
               "sigma_max": float(s1.max()), "sigma_cv": float(s1.std() / s1.mean()),
               "n_median": int(np.median(nr)), "n_min": int(nr.min()), "n_max": int(nr.max())}
        rows.append(row)
        print(f"\n--- {scene} id={pid} ---")
        print(f"  szum 1 renderu po 36 orientacjach: mediana {np.median(s1):.5f}, "
              f"zakres {s1.min():.5f}-{s1.max():.5f} (rozrzut wzgledny {s1.std() / s1.mean() * 100:.1f}%)")
        print(f"  wymagane N po 36 orientacjach: mediana {int(np.median(nr))}, "
              f"zakres {int(nr.min())}-{int(nr.max())}")

    result["per_position"] = rows

    # Porownanie dwoch zrodel zmiennosci: WEWNATRZ pozycji (po orientacjach) vs
    # MIEDZY pozycjami. Jesli to pierwsze jest duzo mniejsze, szum jest wlasnoscia
    # pozycji i N mozna wyznaczac raz na lokalizacje.
    within = float(np.mean([r["sigma_cv"] for r in rows]))
    med = [r["sigma_median"] for r in rows]
    between = float(abs(med[0] - med[1]) / np.mean(med))
    result["cv_within_position"] = within
    result["relative_gap_between_positions"] = between

    print("\n--- WERDYKT ---")
    print(f"  rozrzut szumu WEWNATRZ pozycji (po 36 orientacjach): {within * 100:.1f}%")
    print(f"  roznica szumu MIEDZY dwiema pozycjami:               {between * 100:.1f}%")
    if within < 0.25 * between:
        verdict = ("SZUM JEST WLASNOSCIA POZYCJI - rozrzut po orientacjach jest o rzad wielkosci mniejszy "
                   "niz roznica miedzy pozycjami. N mozna wyznaczyc RAZ na lokalizacje (1740 decyzji "
                   "zamiast 62 640) i zastosowac do wszystkich 36 orientacji.")
    else:
        verdict = ("SZUM ZALEZY TAKZE OD ORIENTACJI - rozrzut wewnatrz pozycji jest porownywalny z roznica "
                   "miedzy pozycjami, wiec regula stopu musi dzialac per probka, z jawnym marginesem na "
                   "obciazenie od optional stopping.")
    result["verdict"] = verdict
    print(f"  {verdict}")
    return result

def run_noise_floor_remaining():
    print("\n=== Podloga szumu w scenach bez pomiaru (rozstrzygniecie o N_MAX) ===")
    import librosa
    sys.path.insert(0, str(REPO_ROOT / "my-operations"))
    import generate_echo_dataset as G

    # Ktore sceny nie maja jeszcze zadnego sigma_1 — czytane z RAPORTU, nie zgadywane.
    report = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            report = json.load(f)
    measured = set()
    for blk in report.values():
        if not isinstance(blk, dict):
            continue
        for sub in ("per_position", "rows"):
            for r in (blk.get(sub) or []):
                if isinstance(r, dict) and "scene" in r and (
                        r.get("sigma_single_render") or r.get("sigma_median")):
                    measured.add(r["scene"])
    if G.scene_h5("office_1").exists():
        measured.add("office_1")          # pelna scena zmierzona w HDF5
    todo = [s for s in G.SCENE_ORDER if s not in measured]

    print(f"  sceny z pomiarem ({len(measured)}): {', '.join(sorted(measured))}")
    print(f"  DO ZMIERZENIA ({len(todo)}): {', '.join(todo)}")

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    result = {"scenes": todo, "angle": REMAINING_ANGLE, "m_per_position": REMAINING_M,
              "fractions": list(REMAINING_FRACTIONS), "rays": 500, "threads": 1,
              "sensor_height": PRODUCTION_SENSOR_HEIGHT,
              "warmup_discard": G.WARMUP_DISCARD, "audio_sims_per_render": 1,
              "estimator": "variance (sigma^2 = mean over cells of Var across renders)",
              "y_source": "graph.pkl (produkcyjna), NIE snap_point",
              "signal_10deg": G.SIGNAL_10DEG, "target_snr": G.TARGET_SNR,
              "n_max": G.N_MAX, "material_config": mc}

    rows, n_sims = [], 0
    for scene in todo:
        loc_ids, positions = G.load_scene_locations(scene)
        pids = [loc_ids[int(f * len(loc_ids))] for f in REMAINING_FRACTIONS]
        sim = build_sim(scene_name=scene, indirect_ray_count=500, thread_count=1,
                        material_config=REPLICA_MATERIAL_CONFIG,
                        sensor_height=PRODUCTION_SENSOR_HEIGHT)
        n_sims += 1
        print(f"\n  [{n_sims}/{len(todo)}] {scene} ({len(loc_ids)} lokalizacji, "
              f"pozycje {pids})", flush=True)
        try:
            # Rozgrzewka: pierwsze ~10 renderow instancji ma szum wyzszy o 10-20 %
            # (GENERATOR_PARAMS.md §2). Bez tego pomiar pierwszej pozycji bylby zawyzony.
            for _ in range(G.WARMUP_DISCARD):
                phase3_echolocation(sim, positions[pids[0]], REMAINING_ANGLE, mc,
                                        run_simulation=False)
            first = True
            for pid in pids:
                specs = []
                for _ in range(REMAINING_M):
                    obs, _lp, _rot = phase3_echolocation(
                        sim, positions[pid], REMAINING_ANGLE, mc if first else None,
                        run_simulation=False)
                    first = False
                    _echo, spec = render_spectrogram(
                        np.transpose(np.array(obs["audio_sensor"])), chirp)
                    specs.append(spec)
                sigma_1 = _sigma_variance(specs)
                n_raw = int(np.ceil((G.TARGET_SNR * sigma_1 / G.SIGNAL_10DEG) ** 2))
                rows.append({"scene": scene, "point_id": int(pid),
                             "n_locations": len(loc_ids),
                             "sigma_single_render": sigma_1, "n_raw": n_raw,
                             "energy": float(np.mean(specs[0]))})
                print(f"    id={pid:<4} sigma_1={sigma_1:.5f}  N_raw={n_raw:<3}"
                      f"{'  <-- POWYZEJ N_MAX' if n_raw > G.N_MAX else ''}")
        finally:
            sim.close()

    result["per_position"] = rows
    result["n_simulator_constructions"] = n_sims

    per_scene = {}
    for scene in todo:
        sub = [r["sigma_single_render"] for r in rows if r["scene"] == scene]
        nr = [r["n_raw"] for r in rows if r["scene"] == scene]
        per_scene[scene] = {"sigma_median": float(np.median(sub)),
                            "sigma_min": float(min(sub)), "sigma_max": float(max(sub)),
                            "n_raw_median": float(np.median(nr)), "n_raw_max": int(max(nr))}
    result["per_scene"] = per_scene

    s1 = np.array([r["sigma_single_render"] for r in rows])
    nr = np.array([r["n_raw"] for r in rows])
    sigma_at_nmax = float(np.sqrt(G.N_MAX) * G.SIGNAL_10DEG / G.TARGET_SNR)
    result["summary"] = {
        "n_positions": len(rows),
        "sigma_median": float(np.median(s1)), "sigma_max": float(s1.max()),
        "sigma_min": float(s1.min()),
        "n_raw_max": int(nr.max()),
        "sigma_threshold_at_n_max": sigma_at_nmax,
        "positions_above_n_max": int((nr > G.N_MAX).sum()),
        "positions_above_24": int((nr > 24).sum()),
    }

    print(f"\n  --- WERDYKT ---")
    print(f"  prog sigma_1 przy N_MAX={G.N_MAX}: {sigma_at_nmax:.5f}")
    print(f"  zmierzone pozycje: {len(rows)}")
    print(f"  sigma_1: mediana {np.median(s1):.5f}, zakres {s1.min():.5f}-{s1.max():.5f}")
    print(f"  N_raw:   mediana {int(np.median(nr))}, MAKSIMUM {int(nr.max())}")
    print(f"  pozycji z N_raw > {G.N_MAX}: {int((nr > G.N_MAX).sum())} / {len(rows)}")
    print(f"  pozycji z N_raw > 24:       {int((nr > 24).sum())} / {len(rows)}")
    result["verdict_n_max_sufficient"] = bool((nr > G.N_MAX).sum() == 0)
    return result
