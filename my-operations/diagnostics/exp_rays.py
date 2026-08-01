"""E2: liczba promieni, bias katowy i budzet watkow."""

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

from .common import (CHIRP_PATH, E2BO_ANGLES, E2BO_N, E2BO_POSITIONS, E2BO_RAYS, MATERIALS_ENABLED, OUT_DIR, PRODUCTION_SENSOR_HEIGHT, REPLICA_MATERIAL_CONFIG, _energy, _get_spec, _material_config_arg, _rms, _rmse, build_sim, load_point_position, render_raw)


# --- E2: wiecej promieni czy wiecej renderow? --------------------------------
#
# Oba sposoby redukuja ten sam szum Monte Carlo, ale kosztuja inaczej: koszt
# renderu rosnie PODLINIOWO z liczba promieni (zmierzone: 10x promieni = 5.8x
# czasu), bo kazdy render ma staly narzut. Przy rownym budzecie promieni jeden
# render 5000-promieniowy jest wiec tanszy niz 10 renderow po 500. Pytanie, czy
# jest tak samo dobry - a to NIE jest oczywiste, bo to sa rozne estymatory:
#   - wiecej promieni redukuje wariancje WEWNATRZ symulacji, zanim policzymy |STFT|
#   - usrednianie N magnitud usrednia PO wzieciu modulu, a E[|X|] > |E[X]|, wiec
#     ten estymator ma dodatnie obciazenie, ktore NIE znika ze wzrostem N
# Dlatego porownujemy przy STALYM budzecie promieni (N * rays = const) i patrzymy
# nie tylko na SNR, ale i na sredni poziom energii - systematyczny spadek energii
# wraz z liczba promieni bylby wlasnie tym obciazeniem magnitudy.

E2_RAY_BUDGET = 5000  # laczny budzet promieni na jedna estymate (N * rays)

E2_CONFIGS = ((10, 500), (5, 1000), (2, 2500), (1, 5000))  # (N renderow, promieni na render)

E2_ANGLES = (0.0, 10.0)

E2_POSITION_IDS = (30, 50, 80)

def run_e2_rays_vs_renders():
    print("\n=== E2: wiecej promieni vs wiecej renderow (staly budzet promieni) ===")
    import librosa

    material_config = _material_config_arg()
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    result = {"materials_enabled": MATERIALS_ENABLED, "ray_budget": E2_RAY_BUDGET,
              "configs": [list(c) for c in E2_CONFIGS], "angles": list(E2_ANGLES),
              "position_ids": list(E2_POSITION_IDS), "thread_count": 1}

    per_config = []
    for n_renders, ray_count in E2_CONFIGS:
        assert n_renders * ray_count == E2_RAY_BUDGET, "konfiguracje musza miec rowny budzet promieni"
        label = f"N={n_renders} x {ray_count} promieni"
        print(f"\n--- {label} ---")
        # threadCount zostawiamy na 1 (domyslne) - cala reszta charakteryzacji
        # jest na jednym watku, wiec porownanie zostaje na jednej osi.
        sim = build_sim(scene_name="room_0", indirect_ray_count=ray_count)
        try:
            rows, render_times = [], []
            for pid in E2_POSITION_IDS:
                pos = load_point_position(sim, "room_0", pid)
                est = {}
                for ang in E2_ANGLES:
                    # 2*N renderow -> dwie ROZLACZNE, niezalezne estymaty tego kata
                    specs = []
                    for _ in range(2 * n_renders):
                        t0 = time.perf_counter()
                        raw = render_raw(sim, pos, ang, material_config)
                        render_times.append(time.perf_counter() - t0)
                        _echo, spec = render_spectrogram(np.transpose(raw), chirp)
                        specs.append(spec)
                    est[("A", ang)] = np.mean(specs[:n_renders], axis=0)
                    est[("B", ang)] = np.mean(specs[n_renders:], axis=0)
                a0, a1 = E2_ANGLES
                noise = float(np.mean([_rmse(est[("A", a0)], est[("B", a0)]),
                                       _rmse(est[("A", a1)], est[("B", a1)])]))
                signal = float(np.mean([_rmse(est[("A", a0)], est[("A", a1)]),
                                        _rmse(est[("B", a0)], est[("B", a1)])]))
                energy = float(np.mean(est[("A", a0)]))
                rows.append({"point_id": pid, "signal": signal, "noise": noise,
                             "snr": signal / noise if noise > 0 else float("inf"), "mean_energy": energy})
                print(f"  id={pid:3d}: sygnal={signal:.5f}  szum={noise:.5f}  "
                      f"SNR={rows[-1]['snr']:5.2f}  energia={energy:.5f}")
        finally:
            sim.close()

        t_render = float(np.mean(render_times))
        t_estimate = t_render * n_renders  # koszt JEDNEJ estymaty (N renderow)
        agg = {"n_renders": n_renders, "ray_count": ray_count, "label": label,
               "snr_mean": float(np.mean([r["snr"] for r in rows])),
               "snr_min": float(np.min([r["snr"] for r in rows])),
               "snr_max": float(np.max([r["snr"] for r in rows])),
               "signal_mean": float(np.mean([r["signal"] for r in rows])),
               "noise_mean": float(np.mean([r["noise"] for r in rows])),
               "energy_mean": float(np.mean([r["mean_energy"] for r in rows])),
               "seconds_per_render": t_render, "seconds_per_estimate": t_estimate,
               "per_position": rows}
        per_config.append(agg)
        print(f"  => SNR sr={agg['snr_mean']:5.2f}  energia sr={agg['energy_mean']:.5f}  "
              f"{t_render:.4f} s/render  {t_estimate:.3f} s/estymate")

    result["per_config"] = per_config

    print("\n--- PODSUMOWANIE E2 (staly budzet promieni = %d) ---" % E2_RAY_BUDGET)
    print(f"  {'konfiguracja':22s} {'SNR':>6s} {'szum':>9s} {'energia':>9s} {'s/estymate':>11s} {'SNR/s':>8s}")
    for c in per_config:
        print(f"  {c['label']:22s} {c['snr_mean']:6.2f} {c['noise_mean']:9.5f} {c['energy_mean']:9.5f} "
              f"{c['seconds_per_estimate']:11.3f} {c['snr_mean'] / c['seconds_per_estimate']:8.3f}")

    best_snr = max(per_config, key=lambda c: c["snr_mean"])
    best_eff = max(per_config, key=lambda c: c["snr_mean"] / c["seconds_per_estimate"])
    cheapest = min(per_config, key=lambda c: c["seconds_per_estimate"])
    baseline = per_config[0]  # N=10 x 500 - konfiguracja, na ktorej stoi E3

    # Czy energia systematycznie zalezy od liczby promieni? (obciazenie magnitudy)
    energies = [c["energy_mean"] for c in per_config]
    energy_drift = (energies[0] - energies[-1]) / energies[0]
    result["energy_drift_lowray_to_highray"] = float(energy_drift)

    result["best_snr"] = best_snr["label"]
    result["best_snr_per_second"] = best_eff["label"]
    speedup = baseline["seconds_per_estimate"] / best_eff["seconds_per_estimate"]
    result["speedup_vs_baseline"] = float(speedup)
    print(f"\n  Najwyzszy SNR:            {best_snr['label']} (SNR {best_snr['snr_mean']:.2f})")
    print(f"  Najlepszy SNR na sekunde: {best_eff['label']} "
          f"({best_eff['snr_mean'] / best_eff['seconds_per_estimate']:.3f} SNR/s, "
          f"{speedup:.2f}x taniej niz N=10x500 przy SNR {best_eff['snr_mean']:.2f})")
    print(f"  Najtansza estymata:       {cheapest['label']} ({cheapest['seconds_per_estimate']:.3f} s)")
    print(f"  Dryf energii (N=10x500 -> N=1x5000): {energy_drift * 100:+.2f}% "
          f"{'- magnitude averaging zawyza energie przy malej liczbie promieni' if energy_drift > 0.02 else '- brak istotnego obciazenia'}")
    return result

# --- E2b: czy 500 promieni OBCIAZA wynik (a nie tylko zaszumia)? -------------
#
# E2 pokazal, ze przy stalym budzecie promieni lepiej usredniac wiele renderow
# po 500 promieni niz robic jeden po 5000 - ale to byl pomiar WARIANCJI. Zupelnie
# osobne pytanie to OBCIAZENIE: czy estymata z 500 promieni zbiega do tego samego,
# co estymata z duzo wieksza liczba promieni, czy do czegos systematycznie innego.
# Ma to znaczenie, bo:
#  - habitat-sim/docs/AUDIO.md podaje indirectRayCount=5000 jako wartosc domyslna
#    i opisuje ja jako "the main parameter for controlling quality", a my uzywamy 500;
#  - paper SoundSpaces 2.0 (NeurIPS 2022, Tab. 2) raportuje 9.5% wzglednego bledu
#    RT60 dla trybu "high-speed" (mniej promieni) wzgledem "high-quality" - czyli
#    zmniejszenie liczby promieni realnie przesuwa wynik, nie tylko go zaszumia.
# Referencja: N=10 renderow po 5000 promieni (10x budzet produkcyjny). Zeby odroznic
# obciazenie od szumu samej referencji, liczymy tez jej wlasny rozrzut z dwoch
# rozlacznych polowek.

E2B_REFERENCE_RAYS = 5000

E2B_PRODUCTION_RAYS = 500

E2B_N = 10

E2B_POSITION_IDS = (30, 50, 80)

E2B_ANGLE = 0.0

def run_e2_ray_bias():
    print("\n=== E2b: czy 500 promieni obciaza estymate (vs 5000)? ===")
    import librosa

    material_config = _material_config_arg()
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    result = {"materials_enabled": MATERIALS_ENABLED, "reference_rays": E2B_REFERENCE_RAYS,
              "production_rays": E2B_PRODUCTION_RAYS, "n_per_estimate": E2B_N,
              "position_ids": list(E2B_POSITION_IDS), "angle": E2B_ANGLE}

    # Zbieramy po 2*N renderow dla obu ustawien, w dwoch osobnych sesjach
    # (rayCount jest wlasciwoscia specu, wiec wymaga osobnego Simulatora).
    estimates = {}
    for rays in (E2B_PRODUCTION_RAYS, E2B_REFERENCE_RAYS):
        sim = build_sim(scene_name="room_0", indirect_ray_count=rays)
        try:
            for pid in E2B_POSITION_IDS:
                pos = load_point_position(sim, "room_0", pid)
                specs = [_get_spec(sim, pos, E2B_ANGLE, material_config, chirp) for _ in range(2 * E2B_N)]
                estimates[(rays, pid, "A")] = np.mean(specs[:E2B_N], axis=0)
                estimates[(rays, pid, "B")] = np.mean(specs[E2B_N:], axis=0)
            print(f"  zebrano {2 * E2B_N} renderow x {len(E2B_POSITION_IDS)} pozycji przy {rays} promieniach")
        finally:
            sim.close()

    rows = []
    for pid in E2B_POSITION_IDS:
        ref_a, ref_b = estimates[(E2B_REFERENCE_RAYS, pid, "A")], estimates[(E2B_REFERENCE_RAYS, pid, "B")]
        prod_a, prod_b = estimates[(E2B_PRODUCTION_RAYS, pid, "A")], estimates[(E2B_PRODUCTION_RAYS, pid, "B")]
        ref_noise = _rmse(ref_a, ref_b)      # wlasny rozrzut referencji
        prod_noise = _rmse(prod_a, prod_b)   # wlasny rozrzut produkcji
        # roznica produkcja-referencja, usredniona po polowkach, zeby nie zalezala
        # od tego, ktora polowke wybierzemy
        gap = float(np.mean([_rmse(prod_a, ref_a), _rmse(prod_a, ref_b),
                             _rmse(prod_b, ref_a), _rmse(prod_b, ref_b)]))
        # Obciazenie = ta czesc roznicy, ktorej NIE tlumaczy szum obu estymat.
        # RMSE(prod, ref)^2 = bias^2 + sigma_prod^2 + sigma_ref^2, a RMSE(A,B)^2 = 2*sigma^2.
        explained = (prod_noise ** 2 + ref_noise ** 2) / 2.0
        bias = float(np.sqrt(max(gap ** 2 - explained, 0.0)))
        e_prod = float(np.mean(prod_a)); e_ref = float(np.mean(ref_a))
        rows.append({"point_id": pid, "ref_noise": ref_noise, "prod_noise": prod_noise,
                     "gap": gap, "bias": bias, "energy_prod": e_prod, "energy_ref": e_ref,
                     "energy_rel_diff": (e_prod - e_ref) / e_ref})
        print(f"  id={pid:3d}: szum ref={ref_noise:.5f} szum prod={prod_noise:.5f} "
              f"| roznica={gap:.5f} -> OBCIAZENIE={bias:.5f} | energia prod/ref={e_prod / e_ref:.4f}")

    result["per_position"] = rows
    m_bias = float(np.mean([r["bias"] for r in rows]))
    m_refnoise = float(np.mean([r["ref_noise"] for r in rows]))
    m_energy = float(np.mean([r["energy_rel_diff"] for r in rows]))
    result["bias_mean"] = m_bias
    result["reference_noise_mean"] = m_refnoise
    result["energy_rel_diff_mean"] = m_energy

    # Skala odniesienia: prawdziwy sygnal 10 stopni ~0.064 (E2, po odszumieniu).
    signal_10deg = 0.064
    result["bias_vs_10deg_signal"] = m_bias / signal_10deg
    print(f"\n--- WERDYKT E2b ---")
    print(f"  Obciazenie 500 vs 5000 promieni: {m_bias:.5f}")
    print(f"  Dla skali: sygnal 10 stopni (odszumiony) ~ {signal_10deg:.3f}, "
          f"wlasny szum referencji przy N=10 = {m_refnoise:.5f}")
    print(f"  Obciazenie to {m_bias / signal_10deg * 100:.1f}% sygnalu 10 stopni; "
          f"roznica energii {m_energy * 100:+.2f}%")
    if m_bias < 0.1 * signal_10deg:
        verdict = ("POMIJALNE - 500 promieni daje praktycznie te sama estymate co 5000, "
                   "wiec produkcyjna konfiguracja nie jest systematycznie przesunieta")
    elif m_bias < 0.3 * signal_10deg:
        verdict = ("MALE, ale niezerowe - 500 promieni lekko przesuwa estymate wzgledem 5000; "
                   "do zaraportowania w pracy jako ograniczenie, nie do zignorowania")
    else:
        verdict = ("ISTOTNE - 500 promieni daje systematycznie inna estymate niz 5000, "
                   "porownywalnie z mierzonym sygnalem; nalezy podniesc liczbe promieni")
    result["verdict"] = verdict
    print(f"  {verdict}")
    return result

def _circular_delta(values, angles, step_deg=10.0):
    """Pary sasiednich orientacji (z zawinieciem 360->0) jako indeksy."""
    n = len(angles)
    return [(i, (i + 1) % n) for i in range(n)]

def _dominant_harmonic(series):
    """Najsilniejsza harmoniczna przebiegu R(theta) po pelnym obrocie.

    Losowy rozrzut rozklada energie rowno po harmonicznych; systematyczny wzorzec
    (np. okresowosc co 90 stopni od czterech scian) skupia ja w jednym prazku.
    Zwraca (indeks harmonicznej, amplituda, okres w stopniach); indeks 4 = okres 90.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    amps = np.abs(np.fft.rfft(x)) / len(x)
    k = int(np.argmax(amps[1:]) + 1)  # pomijamy skladowa stala
    return k, float(amps[k]), 360.0 / k

def run_e2_bias_orientation():
    print("\n=== BLOK A: czy obciazenie od liczby promieni zalezy od orientacji? ===")
    import librosa

    material_config = _material_config_arg()
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    half = E2BO_N // 2
    signal_10deg = 0.064  # prawdziwy (odszumiony) sygnal 10 stopni z E2 - skala odniesienia

    result = {"materials_enabled": MATERIALS_ENABLED, "rays": list(E2BO_RAYS), "n_per_angle": E2BO_N,
              "half_size": half, "n_angles": len(E2BO_ANGLES), "angle_step_deg": 10.0,
              "positions": [{"scene": s, "point_id": p} for s, p in E2BO_POSITIONS],
              "thread_count": 1, "reference_signal_10deg": signal_10deg,
              "material_config": "mp3d_material_config.json (jak w calej dotychczasowej charakteryzacji)"}

    per_position = []
    for scene, pid in E2BO_POSITIONS:
        print(f"\n--- {scene} id={pid} ---")
        # halves[rays] = {"A": [spec per kat], "B": [...]}; trzymamy tylko polowki,
        # pelna estymata N=8 to ich srednia - oszczedza polowe pamieci.
        halves = {}
        for rays in E2BO_RAYS:
            sim = build_sim(scene_name=scene, indirect_ray_count=rays, thread_count=1)
            try:
                pos = load_point_position(sim, scene, pid)
                a_list, b_list = [], []
                t0 = time.perf_counter()
                for ang in E2BO_ANGLES:
                    specs = [_get_spec(sim, pos, ang, material_config, chirp) for _ in range(E2BO_N)]
                    a_list.append(np.mean(specs[:half], axis=0).astype(np.float32))
                    b_list.append(np.mean(specs[half:], axis=0).astype(np.float32))
                dt = time.perf_counter() - t0
                halves[rays] = {"A": a_list, "B": b_list}
                print(f"  {rays:5d} promieni: {len(E2BO_ANGLES)} katow x {E2BO_N} renderow w {dt:.1f} s "
                      f"({dt / (len(E2BO_ANGLES) * E2BO_N):.3f} s/render)")
            finally:
                sim.close()

        full = {r: [(a + b) / 2.0 for a, b in zip(halves[r]["A"], halves[r]["B"])] for r in E2BO_RAYS}
        r_lo, r_hi = E2BO_RAYS  # 500 (produkcja) i 5000 (referencja)

        # --- METRYKA 1: skalarny wskaznik obciazenia per orientacja ---------------
        # R(theta) = energia_500 / energia_5000. Wartosc <1 odpowiada zmierzonemu
        # w E2b spadkowi energii o 2.2%; pytanie brzmi, czy ten spadek jest taki sam
        # dla kazdego kata.
        r_theta = [_energy(full[r_lo][i]) / _energy(full[r_hi][i]) for i in range(len(E2BO_ANGLES))]
        # Kontrola: to samo, ale miedzy dwiema NIEZALEZNYMI polowkami przy TYM SAMYM
        # rayCount - czyli czysty szum, zero wplywu liczby promieni.
        r_ctrl = {r: [_energy(halves[r]["A"][i]) / _energy(halves[r]["B"][i])
                      for i in range(len(E2BO_ANGLES))] for r in E2BO_RAYS}

        sd_r = float(np.std(r_theta))
        sd_ctrl = {r: float(np.std(v)) for r, v in r_ctrl.items()}
        # Skalowanie kontroli do tej samej liczby renderow. Var(log E_{r,N}) = v_r/N.
        #   glowna metryka: Var = (v_lo + v_hi)/N
        #   kontrola przy r: Var = 2*v_r/(N/2) = 4*v_r/N  =>  v_r/N = Var_ctrl,r / 4
        # stad Var_null_glownej = (Var_ctrl,lo + Var_ctrl,hi)/4, czyli sd/2.
        sd_null = 0.5 * float(np.sqrt(sd_ctrl[r_lo] ** 2 + sd_ctrl[r_hi] ** 2))
        k_h, amp_h, period_h = _dominant_harmonic(r_theta)
        k_hc, amp_hc, _ = _dominant_harmonic(r_ctrl[r_hi])

        # --- METRYKA 2: zalezna od kata czesc obciazenia W JEDNOSTKACH RMSE -------
        # Jesli obciazenie bylo tylko jednorodnym wzmocnieniem g, to
        # spec_500(theta) = g * spec_5000(theta) dla kazdego theta i roznice miedzy
        # orientacjami skaluja sie tak samo - nieszkodliwe. Szkodliwa jest dopiero
        # ZMIENNOSC g po katach. Jej wklad do RMSE to |g(theta)-g_sr| * RMS(spec).
        g = np.sqrt(np.asarray(r_theta))
        g_ctrl = {r: np.sqrt(np.asarray(v)) for r, v in r_ctrl.items()}
        sd_g = float(np.std(g))
        sd_g_null = 0.5 * float(np.sqrt(np.std(g_ctrl[r_lo]) ** 2 + np.std(g_ctrl[r_hi]) ** 2))
        sd_g_true = float(np.sqrt(max(sd_g ** 2 - sd_g_null ** 2, 0.0)))  # odjecie szumu w kwadraturze
        mean_rms = float(np.mean([_rms(s) for s in full[r_hi]]))
        gain_bias_rmse = sd_g_true * mean_rms

        # --- METRYKA 3 (ROZSTRZYGAJACA): czy liczba promieni zmienia ZMIERZONA -----
        # ROZNICE MIEDZY SASIEDNIMI ORIENTACJAMI? To jest dokladnie ta wielkosc,
        # na ktorej stoi cala praca. Czula tez na ksztalt, nie tylko na energie.
        pairs = _circular_delta(None, E2BO_ANGLES)
        d_lo = np.array([_rmse(full[r_lo][i], full[r_lo][j]) for i, j in pairs])
        d_hi = np.array([_rmse(full[r_hi][i], full[r_hi][j]) for i, j in pairs])
        delta = d_lo - d_hi
        # Kontrola: ta sama roznica, ale miedzy dwiema polowkami przy tym samym
        # rayCount. Obie polowki sa jednakowo zaszumione, wiec systematyczne
        # zawyzenie RMSE przez szum sie skraca i zostaje sam rozrzut losowy.
        delta_ctrl = {}
        for r in E2BO_RAYS:
            da = np.array([_rmse(halves[r]["A"][i], halves[r]["A"][j]) for i, j in pairs])
            db = np.array([_rmse(halves[r]["B"][i], halves[r]["B"][j]) for i, j in pairs])
            delta_ctrl[r] = da - db
        mean_abs_delta = float(np.mean(np.abs(delta)))
        mean_abs_delta_ctrl = float(np.mean([np.mean(np.abs(v)) for v in delta_ctrl.values()]))

        row = {
            "scene": scene, "point_id": pid,
            "R_theta": [float(v) for v in r_theta],
            "R_theta_mean": float(np.mean(r_theta)), "R_theta_std": sd_r,
            "R_control_std_per_rays": sd_ctrl, "R_control_std_scaled": sd_null,
            "R_spread_vs_control": sd_r / sd_null if sd_null > 0 else float("inf"),
            "dominant_harmonic_k": k_h, "dominant_harmonic_period_deg": period_h,
            "dominant_harmonic_amp": amp_h, "control_dominant_harmonic_amp": amp_hc,
            "gain_sd": sd_g, "gain_sd_null": sd_g_null, "gain_sd_noise_corrected": sd_g_true,
            "spectrogram_rms": mean_rms, "angle_dependent_bias_rmse": gain_bias_rmse,
            "delta10_mean_500": float(np.mean(d_lo)), "delta10_mean_5000": float(np.mean(d_hi)),
            "delta10_shift_mean_abs": mean_abs_delta,
            "delta10_shift_control_mean_abs": mean_abs_delta_ctrl,
            "delta10_shift_vs_signal": mean_abs_delta / signal_10deg,
        }
        per_position.append(row)

        print(f"  M1 R(theta): srednia={np.mean(r_theta):.4f} (E2b przewidywal ~0.978), "
              f"rozrzut po katach={sd_r:.5f}")
        print(f"     kontrola (polowki, ten sam rayCount): surowy={sd_ctrl[r_lo]:.5f}/{sd_ctrl[r_hi]:.5f}, "
              f"przeskalowany do N={E2BO_N}: {sd_null:.5f}  -> stosunek {sd_r / sd_null:.2f}x")
        print(f"     najsilniejsza harmoniczna: k={k_h} (okres {period_h:.0f} st.), amp={amp_h:.5f} "
              f"| kontrola amp={amp_hc:.5f}")
        print(f"  M2 zalezna od kata czesc obciazenia = {gain_bias_rmse:.5f} RMSE "
              f"({gain_bias_rmse / signal_10deg * 100:.1f}% sygnalu 10 st.)")
        print(f"  M3 zmierzona roznica 10 st.: przy 500 = {np.mean(d_lo):.5f}, przy 5000 = {np.mean(d_hi):.5f}")
        print(f"     |przesuniecie| = {mean_abs_delta:.5f} ({mean_abs_delta / signal_10deg * 100:.1f}% sygnalu) "
              f"| kontrola = {mean_abs_delta_ctrl:.5f}")

    result["per_position"] = per_position
    _plot_bias_orientation(per_position, result)

    # --- WERDYKT ---------------------------------------------------------------
    # Decyduje METRYKA 3 (przesuniecie zmierzonej roznicy 10 stopni), bo to ona
    # bezposrednio odpowiada na pytanie "czy liczba promieni zmienia to, co
    # mierzymy". M1/M2 sluza jako spojne potwierdzenie i jako wykryty/niewykryty
    # wzorzec strukturalny.
    m3 = float(np.mean([r["delta10_shift_mean_abs"] for r in per_position]))
    m3_ctrl = float(np.mean([r["delta10_shift_control_mean_abs"] for r in per_position]))
    m2 = float(np.mean([r["angle_dependent_bias_rmse"] for r in per_position]))
    spread_ratio = float(np.mean([r["R_spread_vs_control"] for r in per_position]))
    result["summary"] = {"delta10_shift_mean_abs": m3, "delta10_shift_control_mean_abs": m3_ctrl,
                         "angle_dependent_bias_rmse": m2, "R_spread_vs_control": spread_ratio}

    print("\n--- WERDYKT BLOK A ---")
    print(f"  Rozrzut R(theta) / rozrzut kontrolny        = {spread_ratio:.2f}x")
    print(f"  Zalezna od kata czesc obciazenia (M2)       = {m2:.5f} RMSE")
    print(f"  Przesuniecie zmierzonej roznicy 10 st. (M3) = {m3:.5f} RMSE (kontrola {m3_ctrl:.5f})")
    print(f"  Sygnal 10 stopni (odszumiony, E2)           = {signal_10deg:.3f} RMSE")
    print(f"  -> M3 to {m3 / signal_10deg * 100:.1f}% sygnalu, M2 to {m2 / signal_10deg * 100:.1f}% sygnalu")

    # Progi: 10% sygnalu = "o rzad wielkosci mniej" (kryterium z opisu zadania),
    # 50% = juz porownywalne z mierzonym efektem. Dodatkowo zadamy, zeby efekt
    # przewyzszal wlasna kontrole - inaczej mierzymy szum, a nie orientacje.
    worst = max(m2, m3)
    above_control = m3 > m3_ctrl
    if worst < 0.1 * signal_10deg and not above_control:
        verdict = ("NIEISTOTNY - obciazenie od liczby promieni jest praktycznie jednorodne po "
                   "orientacjach: jego czesc zalezna od kata jest o rzad wielkosci mniejsza niz "
                   "sygnal 10 stopni i nie przekracza wlasnej kontroli szumowej. Ablacja 36 vs 4 "
                   "jest czysta - obciazenie stoi po obu jej stronach jednakowo.")
    elif worst < 0.1 * signal_10deg:
        verdict = ("NIEISTOTNY - czesc zalezna od kata jest o rzad wielkosci mniejsza niz sygnal "
                   "10 stopni, choc nieznacznie przekracza kontrole szumowa; wielkosc efektu jest "
                   "za mala, zeby zaburzyc ablacje.")
    elif worst >= 0.5 * signal_10deg:
        verdict = ("ISTOTNY - obciazenie od liczby promieni zalezy od orientacji w skali "
                   "porownywalnej z mierzonym sygnalem 10 stopni. To powazny konfundent: silnik "
                   "roznicuje orientacje sam z siebie. Wymaga decyzji (wyzszy rayCount w produkcji, "
                   "korekta, albo jawne ograniczenie w pracy).")
    else:
        verdict = ("NIEJEDNOZNACZNY - czesc zalezna od kata jest wieksza niz 10% sygnalu, ale "
                   "mniejsza niz polowa; przy N=%d nie da sie rozstrzygnac, czy to realny efekt "
                   "orientacji czy resztkowy szum. Potrzebne wieksze N (albo wiecej pozycji)." % E2BO_N)
    result["verdict"] = verdict
    print(f"\n  {verdict}")
    return result

def _plot_bias_orientation(per_position, result):
    """Wykres R(theta): liniowy + polarny. Ksztalt jest wazniejszy niz sam rozrzut -
    gladki trend albo okresowosc co 90 stopni imituje sygnal geometryczny, a losowy
    rozrzut nie."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = np.array(E2BO_ANGLES)
    n = len(per_position)
    fig = plt.figure(figsize=(13, 4.6 * n))
    for i, row in enumerate(per_position):
        r = np.array(row["R_theta"])
        ax = fig.add_subplot(n, 2, 2 * i + 1)
        ax.plot(angles, r, "o-", lw=1.4, ms=4, label="R(theta) = E(500)/E(5000)")
        ax.axhline(row["R_theta_mean"], color="k", ls="--", lw=1, label=f"srednia {row['R_theta_mean']:.4f}")
        band = row["R_control_std_scaled"]
        ax.fill_between(angles, row["R_theta_mean"] - band, row["R_theta_mean"] + band,
                        color="gray", alpha=0.25, label=f"+/-1 sd kontroli ({band:.4f})")
        ax.set_xlabel("orientacja [stopnie]"); ax.set_ylabel("R(theta)")
        ax.set_title(f"{row['scene']} id={row['point_id']}  |  rozrzut/kontrola = {row['R_spread_vs_control']:.2f}x")
        ax.set_xticks(range(0, 360, 45)); ax.grid(alpha=0.3); ax.legend(fontsize=7)

        axp = fig.add_subplot(n, 2, 2 * i + 2, projection="polar")
        th = np.deg2rad(np.append(angles, 360.0))
        rr = np.append(r, r[0])
        axp.plot(th, rr, "o-", lw=1.4, ms=3)
        axp.plot(th, np.full_like(th, row["R_theta_mean"]), "k--", lw=1)
        axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
        axp.set_title(f"{row['scene']}: harmoniczna k={row['dominant_harmonic_k']} "
                      f"(okres {row['dominant_harmonic_period_deg']:.0f} st.)", fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "e2_bias_orientation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    result["plot_path"] = str(path)
    print(f"\n  Wykres zapisany: {path}")

THREADEST_RAYS = 500

THREADEST_THREADS = (1, 8)

# M=20 dalo werdykt NIEJEDNOZNACZNY (roznica energii 1.6x wlasny szum, przy tym
# PRZECIWNYCH znakow w dwoch scenach). Eskalacja do 60: szum roznicy energii spada
# o sqrt(3), a koszt to 60 x 0.28 s = 17 s na konfiguracje - pomijalny.
THREADEST_M = 60            # renderow na (pozycja, threadCount); dzielone 30+30

THREADEST_ANGLE = 0.0

THREADEST_POSITIONS = (("room_0", 50), ("office_0", 30))

# Osobna sonda czasowa dla 5000 promieni - potrzebna do tabeli decyzyjnej, ale
# nie do samego testu estymatora (tam wystarcza 500).
THREADEST_TIMING_RAYS = (500, 5000)

THREADEST_TIMING_REPEATS = 12

def _median_render_time(sim, pos, material_config, chirp, repeats):
    _get_spec(sim, pos, 0.0, material_config, chirp)  # rozgrzewka, nie liczona
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        _get_spec(sim, pos, float((i * 37) % 360), material_config, chirp)
        times.append(time.perf_counter() - t0)
    # mediana, nie srednia - pojedyncze zacieciae systemu nie ma zaburzac wyniku
    return float(np.median(times))

def run_e2_thread_estimator():
    print("\n=== BLOK 1: czy threadCount zmienia estymator, czy tylko tempo? ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    half = THREADEST_M // 2
    result = {"rays": THREADEST_RAYS, "threads": list(THREADEST_THREADS), "m_per_config": THREADEST_M,
              "angle": THREADEST_ANGLE, "sensor_height": PRODUCTION_SENSOR_HEIGHT,
              "material_config": mc,
              "positions": [{"scene": s, "point_id": p} for s, p in THREADEST_POSITIONS]}

    rows = []
    for scene, pid in THREADEST_POSITIONS:
        print(f"\n--- {scene} id={pid} ---")
        est, tmed, ir_len = {}, {}, {}
        for th in THREADEST_THREADS:
            sim = build_sim(scene_name=scene, indirect_ray_count=THREADEST_RAYS, thread_count=th,
                            material_config=REPLICA_MATERIAL_CONFIG,
                            sensor_height=PRODUCTION_SENSOR_HEIGHT)
            try:
                pos = load_point_position(sim, scene, pid)
                # Dlugosc IR jako niezalezna sonda "czy liczone jest to samo" -
                # silnik urywa IR przy progu energii, wiec inna liczba probek
                # oznaczalaby inna symulacje, a nie inne tempo.
                ir_len[th] = int(np.asarray(render_raw(sim, pos, THREADEST_ANGLE, mc)).shape[-1])
                specs, times = [], []
                for _ in range(THREADEST_M):
                    t0 = time.perf_counter()
                    specs.append(_get_spec(sim, pos, THREADEST_ANGLE, mc, chirp))
                    times.append(time.perf_counter() - t0)
                est[(th, "A")] = np.mean(specs[:half], axis=0)
                est[(th, "B")] = np.mean(specs[half:], axis=0)
                est[(th, "full")] = np.mean(specs, axis=0)
                tmed[th] = float(np.median(times))
            finally:
                sim.close()

        t1, t8 = THREADEST_THREADS
        # RMSE(A,B) = sqrt(2)*sigma_N (N = half). Estymata "full" ma 2N renderow,
        # wiec jej szum to sigma_N/sqrt(2), czyli RMSE(A,B)/2.
        noise = {th: _rmse(est[(th, "A")], est[(th, "B")]) for th in THREADEST_THREADS}
        sigma_full = {th: noise[th] / 2.0 for th in THREADEST_THREADS}
        energy = {th: float(np.mean(est[(th, "full")])) for th in THREADEST_THREADS}
        e_noise = {th: abs(float(np.mean(est[(th, "A")])) - float(np.mean(est[(th, "B")]))) / energy[th]
                   for th in THREADEST_THREADS}
        gap = _rmse(est[(t1, "full")], est[(t8, "full")])
        effect = float(np.sqrt(max(gap ** 2 - sigma_full[t1] ** 2 - sigma_full[t8] ** 2, 0.0)))
        e_diff = (energy[t8] - energy[t1]) / energy[t1]
        # Szum roznicy energii miedzy dwoma estymatami "full": kazda ma szum
        # e_noise/2 (polowki maja 2x mniej renderow), wiec lacznie w kwadraturze.
        e_diff_noise = float(np.sqrt((e_noise[t1] / 2.0) ** 2 + (e_noise[t8] / 2.0) ** 2))

        row = {"scene": scene, "point_id": pid,
               "noise_halfsplit": noise, "sigma_full": sigma_full,
               "energy": energy, "energy_noise_rel": e_noise,
               "ir_samples": ir_len, "median_s_per_render": tmed,
               "gap_rmse": gap, "effect_rmse": effect,
               "effect_vs_noise": effect / max(sigma_full.values()),
               "energy_diff_rel": e_diff, "energy_diff_noise_rel": e_diff_noise,
               "energy_diff_vs_noise": abs(e_diff) / e_diff_noise if e_diff_noise > 0 else float("inf"),
               "noise_ratio_8_vs_1": noise[t8] / noise[t1],
               "speedup_8_vs_1": tmed[t1] / tmed[t8]}
        rows.append(row)
        print(f"  czas/render (mediana): 1 watek {tmed[t1]:.4f} s, 8 watkow {tmed[t8]:.4f} s "
              f"-> przyspieszenie {tmed[t1] / tmed[t8]:.2f}x")
        print(f"  dlugosc IR: 1 watek {ir_len[t1]} probek, 8 watkow {ir_len[t8]} probek")
        print(f"  szum wlasny (N={half}): 1 watek {noise[t1]:.5f}, 8 watkow {noise[t8]:.5f} "
              f"-> stosunek {noise[t8] / noise[t1]:.3f}x")
        print(f"  RMSE(1 watek, 8 watkow) surowe = {gap:.5f} -> EFEKT po dekompozycji = {effect:.5f} "
              f"({effect / max(sigma_full.values()):.1f}x szum estymaty)")
        print(f"  ENERGIA: 1 watek {energy[t1]:.5f}, 8 watkow {energy[t8]:.5f} -> roznica "
              f"{e_diff * 100:+.2f}% (szum roznicy {e_diff_noise * 100:.2f}%, "
              f"czyli {abs(e_diff) / e_diff_noise:.1f}x)")

    result["per_position"] = rows

    # --- sonda czasowa dla tabeli decyzyjnej ---------------------------------
    print("\n--- sonda czasowa (room_0 id=50, mediana z "
          f"{THREADEST_TIMING_REPEATS} renderow) ---")
    timing = {}
    for rays in THREADEST_TIMING_RAYS:
        for th in THREADEST_THREADS:
            if rays == THREADEST_RAYS:
                timing[f"{rays}/{th}"] = float(np.mean([r["median_s_per_render"][th] for r in rows]))
                continue
            sim = build_sim(scene_name="room_0", indirect_ray_count=rays, thread_count=th,
                            material_config=REPLICA_MATERIAL_CONFIG,
                            sensor_height=PRODUCTION_SENSOR_HEIGHT)
            try:
                pos = load_point_position(sim, "room_0", 50)
                timing[f"{rays}/{th}"] = _median_render_time(sim, pos, mc, chirp, THREADEST_TIMING_REPEATS)
            finally:
                sim.close()
    for k, v in timing.items():
        print(f"  {k:>10} promieni/watkow: {v:.4f} s/render")
    result["timing_s_per_render"] = timing

    # --- WERDYKT --------------------------------------------------------------
    m_e_ratio = float(np.mean([r["energy_diff_vs_noise"] for r in rows]))
    m_e_diff = float(np.mean([r["energy_diff_rel"] for r in rows]))
    m_effect = float(np.mean([r["effect_vs_noise"] for r in rows]))
    same_sign = len({np.sign(r["energy_diff_rel"]) for r in rows}) == 1
    m_noise_ratio = float(np.mean([r["noise_ratio_8_vs_1"] for r in rows]))
    m_speedup = float(np.mean([r["speedup_8_vs_1"] for r in rows]))
    result["summary"] = {"energy_diff_rel_mean": m_e_diff, "energy_diff_vs_noise": m_e_ratio,
                         "effect_vs_noise": m_effect, "energy_shift_same_sign": bool(same_sign),
                         "noise_ratio_8_vs_1": m_noise_ratio, "speedup_8_vs_1": m_speedup}

    print("\n--- WERDYKT BLOK 1 ---")
    print(f"  roznica energii 8 vs 1 watek: {m_e_diff * 100:+.2f}%, czyli {m_e_ratio:.1f}x wlasny szum "
          f"(zgodny znak w obu pozycjach: {same_sign})")
    print(f"  efekt w RMSE po dekompozycji: {m_effect:.1f}x szum estymaty")
    print(f"  stosunek szumu 8/1 watek: {m_noise_ratio:.3f}x | przyspieszenie: {m_speedup:.2f}x")
    if m_e_ratio >= 3.0 and same_sign:
        verdict = ("WATKI ZMIENIAJA ESTYMATOR - energia echa przesuwa sie systematycznie i zgodnie co do "
                   "znaku miedzy 1 a 8 watkami, wielokrotnie ponad wlasny szum obu estymat. To nie jest ta "
                   "sama estymata policzona szybciej. Watki WYKLUCZONE z produkcji; generator idzie na "
                   "threadCount=1.")
    elif m_e_ratio < 1.5 and m_effect < 1.5:
        verdict = ("WATKI TO TYLKO PREDKOSC - roznica energii miesci sie w szumie obu estymat, a efekt w "
                   "RMSE po dekompozycji nie przekracza szumu. Watki sa bezpieczne pod wzgledem wartosci "
                   "oczekiwanej, ale kosztuja odtwarzalnosc bit-exact i podnosza szum pojedynczego renderu.")
    else:
        verdict = (f"NIEJEDNOZNACZNY - roznica energii to {m_e_ratio:.1f}x szum (prog rozstrzygajacy: <1.5 "
                   f"lub >=3.0), a efekt RMSE {m_effect:.1f}x. Przy M={THREADEST_M} nie da sie rozdzielic "
                   "realnego przesuniecia od resztkowego szumu - potrzebne wieksze M albo wiecej pozycji.")
    result["verdict"] = verdict
    print(f"\n  {verdict}")
    return result

# --- BLOK 1b: ILE PROMIENI naprawde liczy 8 watkow? --------------------------
#
# Test rozstrzygajacy hipoteze "threadCount dzieli budzet promieni miedzy watki".
# Zamiast pytac "czy 8 watkow rozni sie od 1 watku" (roznica tonela w szumie przy
# M=20), pytamy WPROST: ktorej liczbie promieni na JEDNYM watku odpowiada wynik
# z 8 watkow? Budujemy drabine 1-watkowa 62/125/250/500 promieni (500/8 = 62.5,
# wiec gdyby budzet byl dzielony, 8 watkow wypadloby przy ~62) i szukamy szczebla,
# ktory najlepiej pasuje energia i szumem do 8 watkow przy 500 promieniach.
#
# To jest test o duzej mocy, bo energia i szum zmieniaja sie z liczba promieni
# MONOTONICZNIE i o wiele bardziej niz roznica 1 vs 8 watkow - E2b zmierzyl -2.2%
# energii miedzy 500 a 5000, wiec skala jest znana.

THREADRAYS_SCENE = "room_0"

THREADRAYS_POINT = 50

THREADRAYS_LADDER = (62, 125, 250, 500)   # 1 watek; 62 ~ 500/8

THREADRAYS_REFERENCE = (500, 8)           # (promienie, watki) - punkt odniesienia

THREADRAYS_M = 60

THREADRAYS_ANGLE = 0.0

def run_e2_thread_effective_rays():
    print("\n=== BLOK 1b: ilu promieniom na 1 watku odpowiada 8 watkow? ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    half = THREADRAYS_M // 2
    configs = [(r, 1) for r in THREADRAYS_LADDER] + [THREADRAYS_REFERENCE]
    result = {"scene": THREADRAYS_SCENE, "point_id": THREADRAYS_POINT, "m_per_config": THREADRAYS_M,
              "ladder": list(THREADRAYS_LADDER), "reference": list(THREADRAYS_REFERENCE),
              "sensor_height": PRODUCTION_SENSOR_HEIGHT, "material_config": mc}

    rows = []
    for rays, th in configs:
        sim = build_sim(scene_name=THREADRAYS_SCENE, indirect_ray_count=rays, thread_count=th,
                        material_config=REPLICA_MATERIAL_CONFIG,
                        sensor_height=PRODUCTION_SENSOR_HEIGHT)
        try:
            pos = load_point_position(sim, THREADRAYS_SCENE, THREADRAYS_POINT)
            specs, times = [], []
            for _ in range(THREADRAYS_M):
                t0 = time.perf_counter()
                specs.append(_get_spec(sim, pos, THREADRAYS_ANGLE, mc, chirp))
                times.append(time.perf_counter() - t0)
            a, b = np.mean(specs[:half], axis=0), np.mean(specs[half:], axis=0)
            full = np.mean(specs, axis=0)
            noise = _rmse(a, b)                       # sqrt(2)*sigma_half
            sigma_1 = noise / np.sqrt(2.0) * np.sqrt(half)   # szum POJEDYNCZEGO renderu
            e = float(np.mean(full))
            e_noise = abs(float(np.mean(a)) - float(np.mean(b))) / e
            rows.append({"rays": rays, "threads": th, "energy": e, "energy_noise_rel": e_noise,
                         "noise_halfsplit": noise, "sigma_single_render": sigma_1,
                         "sigma_full": noise / 2.0, "full": full,
                         "median_s_per_render": float(np.median(times))})
            print(f"  {rays:>4} promieni / {th} watek: energia {e:.5f} (+/-{e_noise * 100:.2f}%), "
                  f"szum 1 renderu {sigma_1:.5f}, {np.median(times):.4f} s/render")
        finally:
            sim.close()

    ref = rows[-1]
    ladder = rows[:-1]
    # Dopasowanie: ktory szczebel jest najblizszy energia i ktory szumem.
    for r in ladder:
        r["energy_gap_rel"] = (ref["energy"] - r["energy"]) / r["energy"]
        gap = _rmse(ref["full"], r["full"])
        r["gap_rmse"] = gap
        r["effect_rmse"] = float(np.sqrt(max(gap ** 2 - ref["sigma_full"] ** 2 - r["sigma_full"] ** 2, 0.0)))
        r["sigma_ratio"] = ref["sigma_single_render"] / r["sigma_single_render"]
    best_energy = min(ladder, key=lambda r: abs(r["energy_gap_rel"]))
    best_noise = min(ladder, key=lambda r: abs(np.log(r["sigma_ratio"])))
    best_spec = min(ladder, key=lambda r: r["effect_rmse"])

    for r in ladder + [ref]:
        r.pop("full", None)
    result["per_config"] = rows
    result["best_match_energy"] = best_energy["rays"]
    result["best_match_noise"] = best_noise["rays"]
    result["best_match_spectrogram"] = best_spec["rays"]

    print(f"\n  {'szczebel':>9}{'roznica energii':>18}{'szum ref/szczebel':>20}{'efekt RMSE':>13}")
    for r in ladder:
        print(f"  {r['rays']:>9}{r['energy_gap_rel'] * 100:>17.2f}%{r['sigma_ratio']:>20.3f}"
              f"{r['effect_rmse']:>13.5f}")
    print(f"\n  8 watkow @500 pasuje najlepiej do 1 watku @ {best_energy['rays']} promieni (energia), "
          f"{best_noise['rays']} (szum), {best_spec['rays']} (spektrogram)")

    # Werdykt: jesli 8 watkow odpowiada ~500 promieniom -> watki nie dziela budzetu.
    # Jesli ~62 -> dziela go dokladnie. Cokolwiek pomiedzy = czesciowa utrata.
    votes = [best_energy["rays"], best_noise["rays"], best_spec["rays"]]
    if all(v == 500 for v in votes):
        verdict = ("WATKI NIE DZIELA BUDZETU - 8 watkow przy 500 promieniach odpowiada 1 watkowi przy "
                   "500 promieniach we wszystkich trzech kryteriach (energia, szum, spektrogram). "
                   "Watki licza te sama prace, tylko szybciej.")
    elif all(v <= 125 for v in votes):
        verdict = ("WATKI DZIELA BUDZET - 8 watkow przy 500 promieniach odpowiada 1 watkowi przy "
                   f"~{max(votes)} promieniach, czyli okolo 500/8. To potwierdza, ze threadCount dzieli "
                   "budzet promieni: dostajesz gorsza jakosc szybciej, nie te sama jakosc szybciej.")
    else:
        verdict = (f"CZESCIOWA UTRATA - kryteria wskazuja na {votes} promieni (energia/szum/spektrogram), "
                   "czyli 8 watkow nie odpowiada ani pelnym 500, ani 500/8. Watki zmieniaja jakosc, ale "
                   "nie przez proste dzielenie budzetu.")
    result["verdict"] = verdict
    print(f"\n  {verdict}")
    return result

# --- BLOK 1c: test symetryczny hipotezy "watki dziela budzet promieni" -------
#
# Drabina z Bloku 1b wskazala, ze 8 watkow @500 lezy spektralnie NAJBLIZEJ 1 watku
# @62 promieni (62 = 500/8), ale dwa z trzech kryteriow nie mialy mocy rozdzielczej:
#  - energia zmienia sie o 0.67% w calym zakresie 62-500 promieni, przy szumie
#    +/-0.1-0.3% - bo energia jest zdominowana przez sciezke bezposrednia i wczesne
#    odbicia, ktore sa DETERMINISTYCZNE i niezalezne od liczby promieni. (Energia
#    byla czula na MATERIALY, bo te zmieniaja absorpcje globalnie - to inny wplyw.)
#  - szum pojedynczego renderu spada z liczba promieni bardzo wolno (E2: 10x promieni
#    = 1.23x mniej szumu) i wyszedl niemonotonicznie.
# Zostalo jedno kryterium, wiec trzeba je sprawdzic testem o przeciwnym kierunku.
#
# Hipoteza H1 "threadCount dzieli budzet miedzy watki" przewiduje DWIE rownosci:
#     500/1 watek  ==  4000/8 watkow      (bo 4000/8 = 500 na watek)
#      62/1 watek  ==   500/8 watkow      (bo  500/8 = 62.5 na watek)
# Hipoteza H0 "watki to tylko predkosc" przewiduje:
#     500/1 watek  ==   500/8 watkow
# Te przewidywania sie wykluczaja, wiec macierz odleglosci miedzy czterema
# konfiguracjami rozstrzyga sprawe niezaleznie od tego, ktora metryke uznamy za
# czula. Kazda odleglosc liczona po dekompozycji: efekt = sqrt(RMSE^2 - s1^2 - s2^2).

THREADBUDGET_CONFIGS = ((500, 1), (4000, 8), (500, 8), (62, 1))

THREADBUDGET_M = 60

THREADBUDGET_SCENE = "room_0"

THREADBUDGET_POINT = 50

THREADBUDGET_ANGLE = 0.0

def run_e2_thread_budget_confirm():
    print("\n=== BLOK 1c: test symetryczny - czy watki dziela budzet promieni? ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    half = THREADBUDGET_M // 2
    result = {"configs": [list(c) for c in THREADBUDGET_CONFIGS], "m_per_config": THREADBUDGET_M,
              "scene": THREADBUDGET_SCENE, "point_id": THREADBUDGET_POINT,
              "sensor_height": PRODUCTION_SENSOR_HEIGHT, "material_config": mc}

    data = {}
    for rays, th in THREADBUDGET_CONFIGS:
        sim = build_sim(scene_name=THREADBUDGET_SCENE, indirect_ray_count=rays, thread_count=th,
                        material_config=REPLICA_MATERIAL_CONFIG,
                        sensor_height=PRODUCTION_SENSOR_HEIGHT)
        try:
            pos = load_point_position(sim, THREADBUDGET_SCENE, THREADBUDGET_POINT)
            specs, times = [], []
            for _ in range(THREADBUDGET_M):
                t0 = time.perf_counter()
                specs.append(_get_spec(sim, pos, THREADBUDGET_ANGLE, mc, chirp))
                times.append(time.perf_counter() - t0)
            a, b = np.mean(specs[:half], axis=0), np.mean(specs[half:], axis=0)
            data[(rays, th)] = {"full": np.mean(specs, axis=0), "sigma_full": _rmse(a, b) / 2.0,
                                "sigma_single": _rmse(a, b) / np.sqrt(2.0) * np.sqrt(half),
                                "energy": float(np.mean(np.mean(specs, axis=0))),
                                "median_s": float(np.median(times))}
            d = data[(rays, th)]
            print(f"  {rays:>4} promieni / {th} watek: energia {d['energy']:.5f}, "
                  f"szum 1 renderu {d['sigma_single']:.5f}, {d['median_s']:.4f} s/render")
        finally:
            sim.close()

    labels = [f"{r}/{t}" for r, t in THREADBUDGET_CONFIGS]
    matrix = {}
    print(f"\n  macierz odleglosci (efekt RMSE po odjeciu szumu obu estymat):")
    print("  " + " " * 10 + "".join(f"{l:>10}" for l in labels))
    for i, ci in enumerate(THREADBUDGET_CONFIGS):
        row = []
        for j, cj in enumerate(THREADBUDGET_CONFIGS):
            if i == j:
                row.append(0.0)
                continue
            gap = _rmse(data[ci]["full"], data[cj]["full"])
            eff = float(np.sqrt(max(gap ** 2 - data[ci]["sigma_full"] ** 2 - data[cj]["sigma_full"] ** 2, 0.0)))
            row.append(eff)
            matrix[f"{labels[i]} vs {labels[j]}"] = eff
        print(f"  {labels[i]:>10}" + "".join(f"{v:>10.5f}" for v in row))

    for c in data:
        data[c].pop("full")
    result["per_config"] = {f"{r}/{t}": data[(r, t)] for r, t in THREADBUDGET_CONFIGS}
    result["distance_matrix"] = matrix

    d_h1_a = matrix["500/1 vs 4000/8"]   # H1 przewiduje ~0
    d_h1_b = matrix["500/8 vs 62/1"]     # H1 przewiduje ~0
    d_h0 = matrix["500/1 vs 500/8"]      # H0 przewiduje ~0
    d_far = matrix["500/1 vs 62/1"]      # skala odniesienia: realna roznica 8x promieni
    result["h1_pred_500_1_vs_4000_8"] = d_h1_a
    result["h1_pred_500_8_vs_62_1"] = d_h1_b
    result["h0_pred_500_1_vs_500_8"] = d_h0
    result["reference_500_1_vs_62_1"] = d_far

    print(f"\n  H1 (watki dziela budzet) przewiduje ~0 dla:")
    print(f"      500/1 vs 4000/8 = {d_h1_a:.5f}")
    print(f"      500/8 vs   62/1 = {d_h1_b:.5f}")
    print(f"  H0 (watki to tylko predkosc) przewiduje ~0 dla:")
    print(f"      500/1 vs  500/8 = {d_h0:.5f}")
    print(f"  skala odniesienia (realna roznica 8x promieni na 1 watku):")
    print(f"      500/1 vs   62/1 = {d_far:.5f}")

    h1_score = max(d_h1_a, d_h1_b)
    if h1_score < 0.5 * d_h0 and d_h0 > 0.3 * d_far:
        verdict = ("WATKI DZIELA BUDZET PROMIENI - obie rownosci przewidziane przez H1 sa spelnione, a "
                   "rownosc przewidziana przez H0 nie. threadCount=8 przy R promieniach daje ten sam "
                   "wynik co threadCount=1 przy R/8 promieniach: dostajesz gorsza jakosc szybciej, nie "
                   "te sama jakosc szybciej. Watki WYKLUCZONE z produkcji.")
    elif d_h0 < 0.5 * h1_score:
        verdict = ("WATKI TO TYLKO PREDKOSC - 500/1 i 500/8 sa nieodroznialne, a konfiguracje o rozniacej "
                   "sie liczbie promieni na watek juz nie. Watki sa bezpieczne pod wzgledem jakosci "
                   "(kosztuja tylko odtwarzalnosc bit-exact).")
    else:
        verdict = (f"NIEJEDNOZNACZNY - zadna z hipotez nie jest wyraznie lepsza: H1 daje {h1_score:.5f}, "
                   f"H0 daje {d_h0:.5f}, przy skali odniesienia {d_far:.5f}. Potrzebne wieksze M albo "
                   "druga pozycja.")
    result["verdict"] = verdict
    print(f"\n  {verdict}")
    return result
