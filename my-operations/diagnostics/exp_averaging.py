"""E3 (domena usredniania) i E4 (dlugosc odpowiedzi impulsowej)."""

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

from .common import (CHIRP_PATH, E3_ANGLES, E3_M, E3_N, E3_POSITION_IDS, MATERIALS_ENABLED, _estimator, _material_config_arg, _rmse, build_sim, load_point_position, render_raw)


# --- E4: skad bierze sie zmienna dlugosc IR ----------------------------------
#
# Motywacja: w raportach z poprzednich sesji IR ma ROZNE dlugosci miedzy
# renderami ([2, 60134] i [2, 65617] w sekcji e1, [1, 71327] w tools/rlr_minimal_example.py).
# To nie jest ciekawostka - decyduje o tym, jak w ogole MOZNA usredniac. Wariant
# usredniania w dziedzinie czasu (E3, estymator "time") wymaga zsumowania
# surowych RIR-ow, a tych o roznej dlugosci nie da sie dodac bez jawnego kroku
# wyrownania. E3 zalezy wiec od E4 i dlatego E4 idzie pierwszy.

E4_POSITION_IDS = (10, 30, 50, 80, 110)  # rozne punkty room_0, ten sam kat

E4_ANGLE = 0.0

# Ile razy powtorzyc TEN SAM render, by odsiac stochastyke od pozycji. Pierwotnie
# 5 - za malo: przy 5 powtorzeniach dlugosc wychodzila stala i E4 bledenie orzekl
# "dlugosc zalezy wylacznie od pozycji", co obalilo dopiero E3 (20 renderow tej
# samej pozy dalo trzy rozne dlugosci). Wahania sa rzadkie i zalezne od punktu,
# wiec potrzeba wiekszej proby i wiecej niz jednego punktu.
E4_REPEATS_SAME_POS = 20

E4_REPEAT_POSITION_IDS = (10, 30, 50)

E4_SCENES = ("office_0", "room_0", "apartment_1")  # male -> duze, do testu hipotezy "wiekszy pokoj = dluzszy IR"

def _acoustics_config_report():
    """Wypisuje wszystkie pola acousticsConfig i wyroznia te, ktore MOGLYBY
    sterowac dlugoscia/progiem IR. Tylko raportujemy - nic nie zmieniamy."""
    spec = habitat_sim.AudioSensorSpec()
    cfg = spec.acousticsConfig
    interesting_kw = ("time", "duration", "length", "threshold", "depth", "ray", "order", "decay", "ir")
    fields = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        try:
            value = getattr(cfg, name)
        except Exception as exc:  # pragma: no cover - defensywnie, gdyby ktores pole rzucalo
            value = f"<blad odczytu: {exc!r}>"
        if callable(value):
            continue
        fields[name] = value if isinstance(value, (int, float, bool, str)) else repr(value)
    hits = sorted(n for n in fields if any(k in n.lower() for k in interesting_kw))
    print("Pola acousticsConfig mogace dotyczyc dlugosci/progu IR:")
    for n in hits:
        print(f"    {n} = {fields[n]}")
    print("Pozostale pola acousticsConfig:")
    for n in sorted(set(fields) - set(hits)):
        print(f"    {n} = {fields[n]}")
    return {"all_fields": fields, "length_related_candidates": hits}

def run_e4_ir_length():
    print("\n=== E4: skad bierze sie zmienna dlugosc IR ===")
    result = {"materials_enabled": MATERIALS_ENABLED, "echo_window_samples": ECHO_SAMPLES}
    material_config = _material_config_arg()

    # --- (3) najpierw inspekcja configu: tania, nie wymaga renderowania ---
    print("\n--- (3) parametry acousticsConfig ---")
    result["acoustics_config"] = _acoustics_config_report()

    sim = build_sim(scene_name="room_0")
    try:
        # --- (1) rozne pozycje, ten sam kat ---
        print(f"\n--- (1) {len(E4_POSITION_IDS)} roznych pozycji (room_0), kat={E4_ANGLE} ---")
        by_position = []
        for pid in E4_POSITION_IDS:
            pos = load_point_position(sim, "room_0", pid)
            raw = render_raw(sim, pos, E4_ANGLE, material_config)
            n = int(raw.shape[1])
            by_position.append({"point_id": pid, "position": [float(v) for v in pos], "ir_samples": n,
                                "ir_ms": n / SAMPLE_RATE * 1000.0})
            print(f"  id={pid:4d} pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}): {n:7d} probek "
                  f"= {n / SAMPLE_RATE * 1000.0:8.1f} ms")
        lens_pos = [d["ir_samples"] for d in by_position]
        result["by_position"] = by_position
        print(f"  rozrzut miedzy pozycjami: min={min(lens_pos)} max={max(lens_pos)} "
              f"rozpietosc={max(lens_pos) - min(lens_pos)} probek")

        # --- (2) ta sama pozycja i kat, N powtorzen, na kilku punktach ---
        print(f"\n--- (2) TA SAMA poza, {E4_REPEATS_SAME_POS} renderow po kolei, na {len(E4_REPEAT_POSITION_IDS)} punktach ---")
        lens_rep = []
        same_pos_detail = []
        varies_within = False
        for pid in E4_REPEAT_POSITION_IDS:
            pos = load_point_position(sim, "room_0", pid)
            lens = [int(render_raw(sim, pos, E4_ANGLE, material_config).shape[1])
                    for _ in range(E4_REPEATS_SAME_POS)]
            lens_rep.extend(lens)
            uniq = sorted(set(lens))
            varies_within = varies_within or len(uniq) > 1
            same_pos_detail.append({"point_id": pid, "lengths": lens, "unique": uniq})
            print(f"  id={pid:4d}: {len(uniq)} unikalnych dlugosci na {E4_REPEATS_SAME_POS} renderow -> {uniq}")
        result["same_position_lengths"] = same_pos_detail
        result["length_varies_within_same_position"] = bool(varies_within)
        print(f"  czy dlugosc waha sie przy USTALONEJ pozie: {'TAK' if varies_within else 'NIE'}")
    finally:
        sim.close()

    # --- (1b) czy dlugosc rosnie z rozmiarem sceny ---
    print(f"\n--- (1b) dlugosc IR a rozmiar sceny ---")
    by_scene = []
    for scene in E4_SCENES:
        s = build_sim(scene_name=scene)
        try:
            bmin, bmax = s.pathfinder.get_bounds()
            extent = [float(bmax[i] - bmin[i]) for i in range(3)]
            volume = float(np.prod(extent))
            pos = load_point_position(s, scene, 10)
            n = int(render_raw(s, pos, E4_ANGLE, material_config).shape[1])
            by_scene.append({"scene": scene, "bbox_extent": extent, "bbox_volume": volume,
                             "ir_samples": n, "ir_ms": n / SAMPLE_RATE * 1000.0})
            print(f"  {scene:18s}: bbox={extent[0]:5.1f}x{extent[1]:5.1f}x{extent[2]:5.1f} m "
                  f"(V={volume:7.1f} m3) -> IR {n:7d} probek = {n / SAMPLE_RATE * 1000.0:8.1f} ms")
        finally:
            s.close()
    result["by_scene"] = by_scene

    # --- werdykt ---
    all_lengths = lens_pos + lens_rep + [d["ir_samples"] for d in by_scene]
    min_len = int(min(all_lengths))
    longer_than_window = bool(min_len > ECHO_SAMPLES)
    result["min_ir_samples_observed"] = min_len
    result["echo_window_always_covered"] = longer_than_window

    varies_across_pos = len(set(lens_pos)) > 1
    if varies_within and varies_across_pos:
        cause = ("oba: dlugosc zalezy i od pozycji (geometria/pogloc - wieksza scena daje dluzszy IR), "
                 "i od stochastyki renderu (ta sama poza daje rozne dlugosci). Silnik urywa IR, gdy energia "
                 "spadnie ponizej progu, a stochastyczny ogon Monte Carlo osiaga ten prog w nieco innym "
                 "momencie przy kazdym przebiegu")
    elif varies_within:
        cause = "wylacznie stochastyka renderu - ten sam punkt daje rozne dlugosci"
    elif varies_across_pos:
        cause = ("wylacznie pozycja/geometria - przy ustalonym punkcie dlugosc jest powtarzalna, "
                 "wiec silnik konczy IR deterministycznie dla danej pozy")
    else:
        cause = "dlugosc stala we wszystkich testowanych warunkach"
    result["length_cause"] = cause

    print("\n--- WERDYKT E4 ---")
    print(f"  Zrodlo zmiennej dlugosci: {cause}")
    print(f"  Najkrotszy zaobserwowany IR: {min_len} probek "
          f"({min_len / SAMPLE_RATE * 1000.0:.1f} ms) vs okno echa {ECHO_SAMPLES} probek (60 ms)")
    if longer_than_window:
        print(f"  => Kazdy IR jest DLUZSZY niz okno 60 ms, wiec po przycieciu do okna zmienna dlugosc "
              f"nie ma znaczenia dla spektrogramu (margines {min_len - ECHO_SAMPLES} probek).")
    else:
        print(f"  => UWAGA: co najmniej jeden IR jest KROTSZY niz okno 60 ms - przyciecie dopelnia zerami, "
              f"co realnie zmienia dane. Trzeba to uwzglednic.")
    return result

def run_e3_averaging_domain():
    print("\n=== E3: domena usredniania ===")
    import librosa

    material_config = _material_config_arg()
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    estimators = ("mag", "en", "time_pad", "time_trunc")
    result = {"materials_enabled": MATERIALS_ENABLED, "n_per_estimate": E3_N, "m_per_angle": E3_M,
              "angles": list(E3_ANGLES), "position_ids": list(E3_POSITION_IDS), "estimators": list(estimators)}

    per_position = []
    sim = build_sim(scene_name="room_0")
    try:
        for pid in E3_POSITION_IDS:
            pos = load_point_position(sim, "room_0", pid)
            print(f"\n--- pozycja id={pid} ---")

            # M renderow na kazdy kat; polowki A (0..N-1) i B (N..2N-1) sa
            # rozlaczne, wiec daja dwie NIEZALEZNE estymaty tego samego kata.
            data = {}
            for ang in E3_ANGLES:
                rirs, specs = [], []
                for _ in range(E3_M):
                    rir = np.transpose(render_raw(sim, pos, ang, material_config))
                    _echo, spec = render_spectrogram(rir, chirp)
                    rirs.append(rir)
                    specs.append(spec)
                data[ang] = {"rirs": rirs, "specs": specs}
                lens = sorted({r.shape[0] for r in rirs})
                print(f"  kat {ang:5.1f}: {E3_M} renderow, dlugosci IR: {lens if len(lens) <= 4 else f'{len(lens)} roznych, {min(lens)}..{max(lens)}'}")

            row = {"point_id": pid, "position": [float(v) for v in pos], "estimators": {}}
            for kind in estimators:
                est = {}
                for ang in E3_ANGLES:
                    d = data[ang]
                    est[("A", ang)] = _estimator(kind, d["specs"][:E3_N], d["rirs"][:E3_N], chirp)
                    est[("B", ang)] = _estimator(kind, d["specs"][E3_N:], d["rirs"][E3_N:], chirp)
                a0, a1 = E3_ANGLES
                # szum: dwie niezalezne estymaty TEGO SAMEGO kata
                noise = float(np.mean([_rmse(est[("A", a0)], est[("B", a0)]),
                                       _rmse(est[("A", a1)], est[("B", a1)])]))
                # sygnal: estymaty SASIEDNICH katow, liczone w obrebie tej samej polowki
                signal = float(np.mean([_rmse(est[("A", a0)], est[("A", a1)]),
                                        _rmse(est[("B", a0)], est[("B", a1)])]))
                snr = signal / noise if noise > 0 else float("inf")
                energy = float(np.mean(est[("A", a0)]))
                row["estimators"][kind] = {"signal": signal, "noise": noise, "snr": snr, "mean_energy": energy}
                print(f"    {kind:11s}: sygnal={signal:.5f}  szum={noise:.5f}  SNR={snr:6.2f}  "
                      f"srednia energia spektrogramu={energy:.5f}")

            # --- diagnostyka energii dla "time": czy usrednianie w czasie wygasza pogloc? ---
            # Odbicia posrednie maja losowe fazy miedzy przebiegami, wiec sumowanie
            # w dziedzinie czasu powinno je czesciowo wygaszac (~1/sqrt(N) dla czesci
            # niekoherentnej), podczas gdy sciezka bezposrednia jest deterministyczna
            # i przetrwa. Jesli to widac, "time" zanizza energie pogloru fizycznie -
            # czyli zniekształca dane, nawet gdyby SNR wyszedl dobrze.
            d0 = data[E3_ANGLES[0]]
            decay = []
            for n in (1, 2, 5, 10, 20):
                e_time = float(np.mean(_estimator("time_trunc", d0["specs"][:n], d0["rirs"][:n], chirp)))
                e_mag = float(np.mean(_estimator("mag", d0["specs"][:n], d0["rirs"][:n], chirp)))
                decay.append({"n": n, "time_energy": e_time, "mag_energy": e_mag})
            e1_time = decay[0]["time_energy"]
            e1_mag = decay[0]["mag_energy"]
            print("    energia spektrogramu w funkcji N (znormalizowana do N=1):")
            for d in decay:
                print(f"      N={d['n']:2d}: time={d['time_energy'] / e1_time:6.3f}  mag={d['mag_energy'] / e1_mag:6.3f}  "
                      f"(1/sqrt(N)={1.0 / np.sqrt(d['n']):6.3f})")
            row["energy_vs_n"] = decay
            per_position.append(row)
    finally:
        sim.close()

    result["per_position"] = per_position

    # --- agregacja po pozycjach ---
    print("\n--- PODSUMOWANIE E3 (srednia po pozycjach) ---")
    agg = {}
    for kind in estimators:
        snrs = [p["estimators"][kind]["snr"] for p in per_position]
        agg[kind] = {
            "snr_mean": float(np.mean(snrs)), "snr_min": float(np.min(snrs)), "snr_max": float(np.max(snrs)),
            "signal_mean": float(np.mean([p["estimators"][kind]["signal"] for p in per_position])),
            "noise_mean": float(np.mean([p["estimators"][kind]["noise"] for p in per_position])),
        }
        print(f"  {kind:11s}: SNR sr={agg[kind]['snr_mean']:6.2f} (zakres {agg[kind]['snr_min']:.2f}-{agg[kind]['snr_max']:.2f})  "
              f"sygnal={agg[kind]['signal_mean']:.5f}  szum={agg[kind]['noise_mean']:.5f}")
    result["aggregate"] = agg

    # czy wybor wyrownania (pad vs trunc) ma znaczenie dla "time"?
    pad_vs_trunc = abs(agg["time_pad"]["snr_mean"] - agg["time_trunc"]["snr_mean"])
    result["time_alignment_matters"] = bool(pad_vs_trunc > 0.01 * max(agg["time_pad"]["snr_mean"], 1e-9))
    print(f"\n  Wyrownanie RIR dla 'time': |SNR_pad - SNR_trunc| = {pad_vs_trunc:.6f} "
          f"-> {'ma znaczenie' if result['time_alignment_matters'] else 'bez znaczenia'}")

    # czy "time" wygasza energie pogloru?
    ratios = [p["energy_vs_n"][-1]["time_energy"] / p["energy_vs_n"][0]["time_energy"] for p in per_position]
    mag_ratios = [p["energy_vs_n"][-1]["mag_energy"] / p["energy_vs_n"][0]["mag_energy"] for p in per_position]
    result["time_energy_ratio_n20_vs_n1"] = float(np.mean(ratios))
    result["mag_energy_ratio_n20_vs_n1"] = float(np.mean(mag_ratios))
    time_distorts = bool(np.mean(ratios) < 0.9)
    result["time_distorts_energy"] = time_distorts
    print(f"  Energia przy N=20 wzgledem N=1: time={np.mean(ratios):.3f}, mag={np.mean(mag_ratios):.3f} "
          f"(1/sqrt(20)={1 / np.sqrt(20):.3f})")
    print(f"  -> 'time' {'ZANIZA' if time_distorts else 'nie zaniza'} energie wraz z N "
          f"{'(zniekształca pogloc fizycznie)' if time_distorts else ''}")

    # --- werdykt: SNR + odpornosc na zmienna dlugosc IR + wiernosc fizyczna ---
    spectral = {k: agg[k]["snr_mean"] for k in ("mag", "en")}
    best_spectral = max(spectral, key=spectral.get)
    time_snr = max(agg["time_pad"]["snr_mean"], agg["time_trunc"]["snr_mean"])
    # Uwaga: kwestia wyrownania RIR okazala sie NIE byc argumentem przeciw "time" -
    # zmierzylismy, ze pad i trunc daja identyczny wynik, bo po przycieciu echa do
    # 60 ms licza sie tylko pierwsze ECHO_SAMPLES probek, a kazdy IR jest od tego
    # wielokrotnie dluzszy (E4). Nie powolujemy sie wiec na ten argument, mimo ze
    # z gory wydawal sie naturalny.
    recommended = best_spectral
    if time_distorts and time_snr < spectral[best_spectral]:
        reason = (f"'time' przegrywa na obu kryteriach naraz: ma nizszy SNR ({time_snr:.2f} vs "
                  f"{spectral[best_spectral]:.2f}) i zaniza energie pogloru (przy N=20 zostaje "
                  f"{np.mean(ratios):.3f} energii z N=1, podczas gdy 'mag' trzyma "
                  f"{np.mean(mag_ratios):.3f}), czyli zniekształca dane fizycznie. Rekomendacja: '{best_spectral}'.")
    elif time_distorts:
        reason = (f"'time' ma wprawdzie wyzszy SNR ({time_snr:.2f} vs {spectral[best_spectral]:.2f}), ale zaniza "
                  f"energie pogloru (przy N=20 zostaje {np.mean(ratios):.3f} energii z N=1) - zniekształca dane "
                  f"fizycznie niezaleznie od SNR. Rekomendacja: '{best_spectral}'.")
    else:
        reason = (f"'{best_spectral}' ma najwyzszy SNR ({spectral[best_spectral]:.2f}), dziala na spektrogramach o "
                  f"stalym ksztalcie i nie zniekształca energii.")
    result["recommended_estimator"] = recommended
    result["recommendation_reason"] = reason
    print(f"\n--- WERDYKT E3 ---\n  Rekomendowany estymator: {recommended}\n  {reason}")
    return result
