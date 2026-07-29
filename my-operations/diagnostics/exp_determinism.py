"""E1 + checkpoint-boundary: determinizm sekwencji RNG i bezpieczenstwo
wznawiania na granicy probki."""

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

from .common import (CHECKPOINT_CORR_THRESHOLD, CHECKPOINT_N, CHECKPOINT_R, CHECKPOINT_REPEAT_CONFIGS, CHIRP_PATH, MATERIALS_ENABLED, REPORT_PATH, _material_config_arg, _render_n_specs, _residuals, build_sim, load_listener_position, load_point_position, render_raw)


def run_p0():
    print("\n=== P0: bramka wstepna - binauralnosc ===")
    result = {"materials_enabled": MATERIALS_ENABLED}

    sim = build_sim()
    try:
        position = load_listener_position(sim)
        raw = render_raw(sim, position, 0.0, _material_config_arg())
        print("Ksztalt surowy obs['audio_sensor']:", raw.shape, raw.dtype)
        rir = np.transpose(raw)
        print("Ksztalt po transpozycji:", rir.shape)

        result["raw_shape"] = list(raw.shape)
        result["transposed_shape"] = list(rir.shape)

        if rir.shape[1] != 2:
            result["status"] = "FAIL"
            result["reason"] = f"Oczekiwano 2 kanalow (binaural), dostano {rir.shape[1]}."
            print("[FAIL]", result["reason"])
        else:
            left, right = rir[:, 0], rir[:, 1]
            channels_identical = bool(np.array_equal(left, right))
            rmse_l_r = float(np.sqrt(np.mean((left - right) ** 2)))
            result["channels_identical"] = channels_identical
            result["rmse_l_r"] = rmse_l_r

            if channels_identical:
                result["status"] = "FAIL"
                result["reason"] = (
                    "Kanaly L i R sa identyczne - layout binauralny nie zadzialal. "
                    "Reszta metody (ITD/ILD) jest bez sensu bez realnej roznicy L/R."
                )
                print("[FAIL]", result["reason"])
                audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
                print("dir(audio_sensor):", [a for a in dir(audio_sensor) if not a.startswith("_")])
                spec = habitat_sim.AudioSensorSpec()
                print("dir(AudioSensorSpec().channelLayout):", [a for a in dir(spec.channelLayout) if not a.startswith("_")])
                print("dir(AudioSensorSpec().acousticsConfig):", [a for a in dir(spec.acousticsConfig) if not a.startswith("_")])
            else:
                result["status"] = "PASS"
                print(f"Kanaly L/R roznia sie - RMSE(L,R) = {rmse_l_r:.6g}. PASS.")
    finally:
        sim.close()

    return result

def run_e1():
    print("\n=== E1: determinizm ziarna losowego ===")
    result = {"materials_enabled": MATERIALS_ENABLED}
    material_config = _material_config_arg()

    # 1. Przeszukaj acousticsConfig pod katem seed/random/rng/sample.
    probe_spec = habitat_sim.AudioSensorSpec()
    seed_related = [
        a
        for a in dir(probe_spec.acousticsConfig)
        if not a.startswith("_") and any(k in a.lower() for k in ("seed", "random", "rng", "sample"))
    ]
    print("dir(acousticsConfig) - atrybuty zwiazane z seed/random/rng/sample:", seed_related)
    print("Pelny dir(acousticsConfig):", [a for a in dir(probe_spec.acousticsConfig) if not a.startswith("_")])
    result["seed_related_attrs"] = seed_related

    # 2. Test wewnatrz sesji: 0 -> 90 -> 0 (90 w srodku, zeby wykluczyc cache).
    sim = build_sim()
    try:
        position = load_listener_position(sim)
        raw_0_a = render_raw(sim, position, 0.0, material_config)
        raw_90 = render_raw(sim, position, 90.0, material_config)  # noqa: F841 - w srodku sekwencji celowo
        raw_0_b = render_raw(sim, position, 0.0, material_config)

        within_identical = bool(np.array_equal(raw_0_a, raw_0_b))
        result["within_session_identical"] = within_identical
        result["within_session_shapes"] = [list(raw_0_a.shape), list(raw_0_b.shape)]
        if not within_identical:
            if raw_0_a.shape == raw_0_b.shape:
                result["within_session_max_abs_diff"] = float(np.abs(raw_0_a - raw_0_b).max())
            else:
                # Same angle/pozycja, ale rozna liczba probek w IR - sam ten fakt
                # juz dowodzi niedeterminizmu (dlugosc IR zalezy od tego, kiedy
                # ray tracing Monte Carlo uzna odbicia za wygaszone), niezaleznie
                # od tego czy wspolna czesc probek tez sie rozni.
                print(
                    "  UWAGA: ksztalty sie roznia "
                    f"({raw_0_a.shape} vs {raw_0_b.shape}) - IR ma rozna dlugosc miedzy renderami "
                    "tego samego kata/pozycji, nie tylko rozne wartosci probek."
                )
        print("Test wewnatrz sesji (0 -> 90 -> 0), pierwszy == trzeci:", within_identical)
    finally:
        sim.close()

    # 3. Test miedzy sesjami: zamknij, otworz nowy symulator, zrenderuj 0 ponownie.
    t0 = time.perf_counter()
    sim2 = build_sim()
    restart_cost_s = time.perf_counter() - t0
    try:
        position2 = load_listener_position(sim2)
        raw_0_c = render_raw(sim2, position2, 0.0, material_config)

        between_identical = bool(np.array_equal(raw_0_a, raw_0_c))
        result["between_session_identical"] = between_identical
        result["between_session_shapes"] = [list(raw_0_a.shape), list(raw_0_c.shape)]
        result["restart_cost_seconds"] = restart_cost_s
        if not between_identical:
            if raw_0_a.shape == raw_0_c.shape:
                result["between_session_max_abs_diff"] = float(np.abs(raw_0_a - raw_0_c).max())
            else:
                print(
                    "  UWAGA: ksztalty sie roznia "
                    f"({raw_0_a.shape} vs {raw_0_c.shape}) - IR ma rozna dlugosc miedzy sesjami."
                )
        print(
            f"Test miedzy sesjami (nowy Simulator), rownie == pierwszemu: {between_identical} "
            f"(koszt restartu symulatora: {restart_cost_s:.3f} s)"
        )
    finally:
        sim2.close()

    within = result["within_session_identical"]
    between = result.get("between_session_identical")
    if not within and not between:
        interpretation = (
            "rendery ROZNE w obu testach -> usrednianie ma sens; generacja datasetu NIE jest "
            "bit-reproducible i trzeba to opisac w pracy."
        )
    elif within and not between:
        interpretation = (
            "identyczne wewnatrz sesji, rozne miedzy sesjami -> usrednianie mozliwe, ale kazdy "
            f"render wymaga restartu symulatora (koszt restartu: {restart_cost_s:.3f} s/restart) - "
            "to moze zdominowac budzet czasu dla pelnego datasetu."
        )
    elif within and between:
        interpretation = (
            "rendery IDENTYCZNE w obu testach -> usrednianie renderow bezuzyteczne, jedyna droga "
            "to podbicie indirectRayCount. Zmienia to plan kolejnej sesji."
        )
    else:
        interpretation = (
            "przypadek nieoczekiwany: rozne wewnatrz sesji, identyczne miedzy sesjami - wymaga "
            "recznej analizy, nie pasuje do zadnego z trzech przewidzianych przypadkow."
        )
    result["interpretation"] = interpretation
    print("INTERPRETACJA:", interpretation)

    return result

# --- E1 rozszerzone: wiecej katow i wiecej restartow, do umocnienia tezy z E1 -

# Docelowa gestosc katowa metody (36 orientacji co 10 stopni) - pelny lap
# odzwierciedla realny wzorzec wywolan generatora, nie tylko probke 3 katow.
FULL_ANGLE_SWEEP = tuple(float(a) for a in range(0, 360, 10))  # 36 katow

# Krotka, stala sekwencja katow powtarzana identycznie w kazdej swiezej sesji -
# celowo NIE identyczna z FULL_ANGLE_SWEEP, zeby test byl niezalezny od niej.
FIXED_CALL_SEQUENCE = (0.0, 90.0, 180.0, 270.0)

N_RESTARTS = 5

def _renders_all_equal(arrays):
    """True tylko jesli WSZYSTKIE tablice w liscie sa parami bit-identyczne."""
    first = arrays[0]
    return all(np.array_equal(first, a) for a in arrays[1:])

def run_e1_extended():
    print("\n=== E1 rozszerzone: wiecej katow i restartow (umocnienie tezy z E1) ===")
    result = {"materials_enabled": MATERIALS_ENABLED}
    material_config = _material_config_arg()

    # --- Scenariusz A: pelny dwu-lapowy sweep 36 katow w JEDNEJ sesji -------
    # Cel: sprawdzic teze "kolejne wywolania w tej samej instancji roznia sie"
    # na realistycznym wzorcu wywolan (36 katow), nie tylko na probce 3 katow
    # jak w podstawowym E1. Oczekiwanie zgodne z hipoteza z E1: KAZDA para
    # (lap1[kat], lap2[kat]) powinna sie roznic, bo RNG advance'uje sekwencyjnie
    # z kazdym runSimulation() w tej samej instancji, niezaleznie od kata.
    print(f"\n--- Scenariusz A: 2x sweep po {len(FULL_ANGLE_SWEEP)} katow w jednej sesji ---")
    sim = build_sim()
    try:
        position = load_listener_position(sim)
        lap1 = [render_raw(sim, position, angle, material_config) for angle in FULL_ANGLE_SWEEP]
        lap2 = [render_raw(sim, position, angle, material_config) for angle in FULL_ANGLE_SWEEP]
    finally:
        sim.close()

    per_angle_identical = [bool(np.array_equal(a, b)) for a, b in zip(lap1, lap2)]
    n_identical = sum(per_angle_identical)
    result["scenario_a_n_angles"] = len(FULL_ANGLE_SWEEP)
    result["scenario_a_n_identical_pairs"] = n_identical
    result["scenario_a_per_angle_identical"] = dict(zip((str(a) for a in FULL_ANGLE_SWEEP), per_angle_identical))
    print(f"Identyczne pary (lap1==lap2) na {len(FULL_ANGLE_SWEEP)} katow: {n_identical}")
    if n_identical:
        identical_angles = [a for a, same in zip(FULL_ANGLE_SWEEP, per_angle_identical) if same]
        print("  UWAGA - katy z identyczna para (nieoczekiwane wg hipotezy z E1):", identical_angles)

    # Sanity-check: czy jakikolwiek render z lap2 przypadkiem powiela render z
    # lap1 pod INNYM katem (wykluczenie krotkiego, malego cyklu RNG zamiast
    # prawdziwego advance'u).
    cross_matches = []
    for i, b in enumerate(lap2):
        for j, a in enumerate(lap1):
            if i != j and np.array_equal(a, b):
                cross_matches.append((FULL_ANGLE_SWEEP[j], FULL_ANGLE_SWEEP[i]))
    result["scenario_a_cross_angle_matches"] = cross_matches
    print("Przypadkowe dopasowania lap1[inny kat]==lap2[ten kat]:", cross_matches if cross_matches else "brak")

    # --- Scenariusz B: N niezaleznych restartow, ta sama stala sekwencja katow -
    # Cel: umocnic "identyczne miedzy sesjami" z n=2 (podstawowe E1) do n=5, ORAZ
    # sprawdzic, czy determinizm dotyczy TYLKO pierwszego wywolania po starcie,
    # czy calej sekwencji wywolan (czyli czy "szum" to w rzeczywistosci STALA,
    # deterministyczna sciezka wywolan, a nie prawdziwa losowosc per-wywolanie).
    print(f"\n--- Scenariusz B: {N_RESTARTS} restartow, stala sekwencja katow {FIXED_CALL_SEQUENCE} ---")
    per_restart_calls = []
    restart_costs = []
    for restart_idx in range(N_RESTARTS):
        t0 = time.perf_counter()
        sim_r = build_sim()
        restart_costs.append(time.perf_counter() - t0)
        try:
            pos_r = load_listener_position(sim_r)
            calls = [render_raw(sim_r, pos_r, angle, material_config) for angle in FIXED_CALL_SEQUENCE]
            per_restart_calls.append(calls)
        finally:
            sim_r.close()

    # Dla kazdego indeksu wywolania (0..len(FIXED_CALL_SEQUENCE)-1) sprawdz, czy
    # wynik jest identyczny we WSZYSTKICH N_RESTARTS restartach.
    per_call_index_all_equal = []
    for call_idx in range(len(FIXED_CALL_SEQUENCE)):
        arrays_at_this_call_idx = [per_restart_calls[r][call_idx] for r in range(N_RESTARTS)]
        per_call_index_all_equal.append(_renders_all_equal(arrays_at_this_call_idx))

    result["scenario_b_n_restarts"] = N_RESTARTS
    result["scenario_b_fixed_angle_sequence"] = list(FIXED_CALL_SEQUENCE)
    result["scenario_b_per_call_index_identical_across_restarts"] = per_call_index_all_equal
    result["scenario_b_restart_costs_seconds"] = restart_costs
    for call_idx, identical in enumerate(per_call_index_all_equal):
        print(
            f"  wywolanie #{call_idx} (kat {FIXED_CALL_SEQUENCE[call_idx]}), "
            f"identyczne we wszystkich {N_RESTARTS} restartach: {identical}"
        )

    # --- Interpretacja rozszerzona -----------------------------------------
    all_calls_deterministic = all(per_call_index_all_equal)
    only_first_deterministic = per_call_index_all_equal[0] and not all(per_call_index_all_equal[1:])
    none_deterministic = not any(per_call_index_all_equal)

    if all_calls_deterministic and n_identical == 0:
        interpretation = (
            "POTWIERDZONE na szerszej probce: caly ciag wywolan po swiezym starcie jest "
            "STALY i deterministyczny (kazdy z 4 indeksow wywolania identyczny w 5/5 "
            "restartach), a mimo to kolejne wywolania W TEJ SAMEJ sesji sie roznia (0/36 "
            "identycznych par w dwulapowym sweepie). Wniosek: RLRAudioPropagation NIE ma "
            "prawdziwej losowosci per-proces - ma DETERMINISTYCZNY, z gory ustalony ciag "
            "stanow RNG konsumowany kolejnymi wywolaniami runSimulation(), zresetowany do "
            "tego samego punktu startowego przy kazdej nowej instancji. Usrednianie N "
            "sekwencyjnych renderow w JEDNEJ sesji daje wiec niezalezne probki szumu, ALE "
            "usrednianie przez wielokrotne URUCHOMIENIE calego pipeline'u od zera (restart "
            "symulatora) da IDENTYCZNY wynik za kazdym razem, jesli kolejnosc wywolan jest "
            "taka sama - restart NIE jest zrodlem niezaleznosci prob, tylko powtorzeniem."
        )
    elif only_first_deterministic:
        interpretation = (
            "Tylko PIERWSZE wywolanie po starcie jest deterministyczne miedzy restartami; "
            "kolejne (#1, #2, #3) juz nie - restart resetuje RNG do stalego stanu "
            "poczatkowego, ale dalsza ewolucja zalezy od czegos poza samym indeksem "
            "wywolania (np. rzeczywisty czas, stan watkow ray tracera). Trzeba to zbadac "
            "dalej w kolejnej sesji, zanim zalozy sie ktorykolwiek z prostszych modeli."
        )
    elif none_deterministic:
        interpretation = (
            "Zaden indeks wywolania nie jest deterministyczny miedzy restartami (nawet "
            "pierwszy) - to PRZECZY podstawowemu wynikowi E1 (gdzie n=2 restarty dawaly "
            "identyczny pierwszy render). Mozliwe zrodla: zaleznosc od czasu rzeczywistego, "
            "wielowatkowosc ray tracera (threadCount), albo stan systemowy (np. obciazenie "
            "GPU) - wymaga dalszego zbadania, wynik z n=2 w podstawowym E1 mogl byc "
            "przypadkiem."
        )
    else:
        interpretation = (
            f"Wzorzec czesciowy: {sum(per_call_index_all_equal)}/{len(per_call_index_all_equal)} "
            "indeksow wywolania deterministycznych miedzy restartami, bez prostego wzorca "
            "'tylko pierwsze' - wymaga recznej analizy per-indeks (patrz "
            "scenario_b_per_call_index_identical_across_restarts w raporcie)."
        )

    result["interpretation_extended"] = interpretation
    print("\nINTERPRETACJA ROZSZERZONA:", interpretation)

    return result

def _run_checkpoint_repeats(repeat_indices):
    """Rdzen eksperymentu: renderuje i zwraca SUROWE dane (bez agregacji) dla
    podanych indeksow do CHECKPOINT_REPEAT_CONFIGS.

    Wydzielone z run_e1_checkpoint_boundary(), zeby dalo sie uruchomic
    podzbior powtorzen w OSOBNYM procesie (patrz run_e1_checkpoint_boundary_batch_a/b)
    - kazda konstrukcja Simulatora otwiera nowy kontekst GL/EGL, a po ~30
    konstrukcjach w JEDNYM procesie ten build napotyka realny wyciek zasobow
    GPU (framebuffer completeness assertion w RenderTarget.cpp, zaobserwowane
    empirycznie 2026-07-24 przy probie R=16 w jednym procesie) - podzial na
    osobne procesy resetuje ten stan, bo kazdy proces dostaje swiezy kontekst.
    """
    from scipy.stats import pearsonr
    import librosa

    material_config = _material_config_arg()
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=SAMPLE_RATE, mono=True)

    repeats_out = []

    for repeat_idx in repeat_indices:
        cfg = CHECKPOINT_REPEAT_CONFIGS[repeat_idx]
        scene = cfg["scene"]
        print(
            f"\n--- Powtorzenie {repeat_idx + 1}/{len(CHECKPOINT_REPEAT_CONFIGS)}: scena={scene}, "
            f"S1=(id={cfg['loc1_id']}, kat={cfg['angle1']}), S2=(id={cfg['loc2_id']}, kat={cfg['angle2']}) ---"
        )

        # CONTINUED: S1 i S2 w JEDNEJ ciaglej sesji, S2 zaraz po S1.
        sim = build_sim(scene_name=scene)
        try:
            pos1 = load_point_position(sim, scene, cfg["loc1_id"])
            pos2 = load_point_position(sim, scene, cfg["loc2_id"])
            specs_s1 = _render_n_specs(sim, pos1, cfg["angle1"], material_config, chirp, CHECKPOINT_N)
            specs_s2 = _render_n_specs(sim, pos2, cfg["angle2"], material_config, chirp, CHECKPOINT_N)
        finally:
            sim.close()

        # RESTARTED: S1 w procesie A, zamkniecie, S2 w NOWEJ instancji B.
        sim_a = build_sim(scene_name=scene)
        try:
            pos1r = load_point_position(sim_a, scene, cfg["loc1_id"])
            specs_s1r = _render_n_specs(sim_a, pos1r, cfg["angle1"], material_config, chirp, CHECKPOINT_N)
        finally:
            sim_a.close()

        sim_b = build_sim(scene_name=scene)
        try:
            pos2r = load_point_position(sim_b, scene, cfg["loc2_id"])
            specs_s2r = _render_n_specs(sim_b, pos2r, cfg["angle2"], material_config, chirp, CHECKPOINT_N)
        finally:
            sim_b.close()

        residuals_s1 = _residuals(specs_s1)
        residuals_s2 = _residuals(specs_s2)
        residuals_s1r = _residuals(specs_s1r)
        residuals_s2r = _residuals(specs_s2r)

        r_continued = []
        r_restarted = []
        for i in range(CHECKPOINT_N):
            rc, _p = pearsonr(residuals_s1[i].flatten(), residuals_s2[i].flatten())
            rr, _p = pearsonr(residuals_s1r[i].flatten(), residuals_s2r[i].flatten())
            r_continued.append(float(rc))
            r_restarted.append(float(rr))
            print(f"  i={i}: r_continued={rc:+.4f}  r_restarted={rr:+.4f}")

        repeats_out.append(
            {
                "repeat_idx": repeat_idx,
                "scene": scene,
                "loc1_id": cfg["loc1_id"],
                "angle1": cfg["angle1"],
                "loc2_id": cfg["loc2_id"],
                "angle2": cfg["angle2"],
                "r_continued": r_continued,
                "r_restarted": r_restarted,
            }
        )

    return repeats_out

def _aggregate_checkpoint_results(repeats_out):
    """Agregacja: mediany, Wilcoxon, werdykt - z listy powtorzen (kazde z
    r_continued/r_restarted juz policzonymi). Wspoldzielona miedzy
    pojedynczym-procesowym uruchomieniem a merge'em wynikow z osobnych batchy.
    """
    from scipy.stats import wilcoxon

    r_continued_all = []
    r_restarted_all = []
    for rep in repeats_out:
        r_continued_all.extend(rep["r_continued"])
        r_restarted_all.extend(rep["r_restarted"])

    result = {
        "materials_enabled": MATERIALS_ENABLED,
        "n_per_sample": CHECKPOINT_N,
        "r_repeats": len(repeats_out),
        "corr_threshold": CHECKPOINT_CORR_THRESHOLD,
        "repeats": repeats_out,
    }

    r_continued_arr = np.array(r_continued_all)
    r_restarted_arr = np.array(r_restarted_all)
    abs_continued = np.abs(r_continued_arr)
    abs_restarted = np.abs(r_restarted_arr)

    median_abs_continued = float(np.median(abs_continued))
    median_abs_restarted = float(np.median(abs_restarted))
    n_exceed_continued = int(np.sum(abs_continued > CHECKPOINT_CORR_THRESHOLD))
    n_exceed_restarted = int(np.sum(abs_restarted > CHECKPOINT_CORR_THRESHOLD))

    result["r_continued_all"] = r_continued_all
    result["r_restarted_all"] = r_restarted_all
    result["median_abs_r_continued"] = median_abs_continued
    result["median_abs_r_restarted"] = median_abs_restarted
    result["range_r_continued"] = [float(r_continued_arr.min()), float(r_continued_arr.max())]
    result["range_r_restarted"] = [float(r_restarted_arr.min()), float(r_restarted_arr.max())]
    result["n_exceeding_threshold_continued"] = n_exceed_continued
    result["n_exceeding_threshold_restarted"] = n_exceed_restarted
    result["n_total_pairs"] = len(r_continued_all)

    diffs = abs_restarted - abs_continued
    if np.all(diffs == 0):
        result["wilcoxon_statistic"] = None
        result["wilcoxon_pvalue"] = None
        result["wilcoxon_note"] = "wszystkie roznice |r_restarted|-|r_continued| sa dokladnie zero - test niewykonalny (brak wariancji)."
        print("\nWilcoxon: niewykonalny -", result["wilcoxon_note"])
    else:
        stat, pvalue = wilcoxon(diffs)
        result["wilcoxon_statistic"] = float(stat)
        result["wilcoxon_pvalue"] = float(pvalue)
        print(f"\nWilcoxon signed-rank (|r_restarted| - |r_continued|): statystyka={stat:.4f}, p={pvalue:.4g}")

    print(f"Mediana |r_continued| = {median_abs_continued:.4f} (zakres {result['range_r_continued']})")
    print(f"Mediana |r_restarted| = {median_abs_restarted:.4f} (zakres {result['range_r_restarted']})")
    print(
        f"Przekroczenia progu {CHECKPOINT_CORR_THRESHOLD}: continued={n_exceed_continued}/{len(r_continued_all)}, "
        f"restarted={n_exceed_restarted}/{len(r_restarted_all)}"
    )

    # Werdykt: mediana jest glowna miara (odporna na pojedyncze wartosci
    # odstajace przy tak malej probce R*N=32), prog roznicy median dobrany tak,
    # zeby "systematyczne przesuniecie" bylo widoczne golym okiem w liczbach,
    # nie tylko w p-value przy duzym N-per-residuum.
    median_shift = median_abs_restarted - median_abs_continued
    exceed_shift = n_exceed_restarted - n_exceed_continued

    if median_shift > 0.02 or exceed_shift >= 3:
        verdict = "NIEBEZPIECZNY"
        verdict_reason = (
            f"Mediana |r_restarted| ({median_abs_restarted:.4f}) systematycznie wyzsza niz "
            f"|r_continued| ({median_abs_continued:.4f}), przesuniecie={median_shift:+.4f}; "
            f"przekroczenia progu: {n_exceed_restarted} (restarted) vs {n_exceed_continued} (continued). "
            "Restart wprowadza wykrywalna korelacje miedzy roznymi probkami - generator MUSI byc "
            "jednym nieprzerywalnym procesem."
        )
    elif abs(median_shift) <= 0.01 and abs(exceed_shift) <= 1:
        verdict = "BEZPIECZNY"
        verdict_reason = (
            f"Rozklady |r_continued| (mediana {median_abs_continued:.4f}) i |r_restarted| "
            f"(mediana {median_abs_restarted:.4f}) praktycznie nierozroznialne, przesuniecie="
            f"{median_shift:+.4f}, brak systematycznej roznicy w przekroczeniach progu. "
            "Checkpointing symulatora na granicy probki jest bezpieczny."
        )
    else:
        verdict = "NIEJEDNOZNACZNY"
        verdict_reason = (
            f"Przesuniecie median ({median_shift:+.4f}) i przekroczen progu ({exceed_shift:+d}) sa w strefie "
            f"posredniej - ani wyraznie zaniedbywalne, ani wyraznie systematyczne przy R={len(repeats_out)} "
            "powtorzeniach. Nie da sie jednoznacznie rozstrzygnac bez wiekszej probki (wiekszego R) - "
            "decyzja co dalej nalezy do uzytkownika, nie zostala podjeta automatycznie."
        )

    result["verdict"] = verdict
    result["verdict_reason"] = verdict_reason
    print(f"\nWERDYKT: {verdict}")
    print(verdict_reason)

    return result

def run_e1_checkpoint_boundary():
    """Pelny przebieg w JEDNYM procesie - dziala tylko do ok. R=8-9 (patrz
    komentarz w _run_checkpoint_repeats o wycieku zasobow GL po ~30
    konstrukcjach Simulatora). Dla R=16 uzyj batch_a/batch_b + merge."""
    print("\n=== E1 checkpoint boundary: restart miedzy ROZNYMI probkami ===")
    repeats_out = _run_checkpoint_repeats(range(CHECKPOINT_R))
    return _aggregate_checkpoint_results(repeats_out)

def run_e1_checkpoint_boundary_batch_a():
    """Powtorzenia 0..7 (pierwsza polowa CHECKPOINT_REPEAT_CONFIGS przy R=16),
    do uruchomienia jako OSOBNY proces - patrz komentarz w
    _run_checkpoint_repeats. Wynik surowy (bez agregacji), scalany pozniej
    przez run_e1_checkpoint_boundary_merge()."""
    print("\n=== E1 checkpoint boundary - batch A (powtorzenia 1-8) ===")
    repeats_out = _run_checkpoint_repeats(range(0, 8))
    return {"repeats": repeats_out}

def run_e1_checkpoint_boundary_batch_b():
    """Powtorzenia 8..15 (druga polowa), w OSOBNYM procesie od batch_a."""
    print("\n=== E1 checkpoint boundary - batch B (powtorzenia 9-16) ===")
    repeats_out = _run_checkpoint_repeats(range(8, 16))
    return {"repeats": repeats_out}

def run_e1_checkpoint_boundary_merge():
    """Laczy surowe wyniki batch_a + batch_b (juz zapisane w
    diagnostics_report.json przez wczesniejsze, osobne procesy) w jeden
    zagregowany wynik pod kluczem e1_checkpoint_boundary. Nie wymaga
    Simulatora/GPU - tylko odczyt raportu z dysku."""
    if not REPORT_PATH.exists():
        raise RuntimeError(f"{REPORT_PATH} nie istnieje - uruchom najpierw batch_a i batch_b.")
    with open(REPORT_PATH) as f:
        existing = json.load(f)

    missing = [k for k in ("e1_checkpoint_boundary_batch_a", "e1_checkpoint_boundary_batch_b") if k not in existing]
    if missing:
        raise RuntimeError(f"Brakuje w raporcie: {missing} - uruchom te batche przed merge.")

    repeats_out = existing["e1_checkpoint_boundary_batch_a"]["repeats"] + existing["e1_checkpoint_boundary_batch_b"]["repeats"]
    repeats_out.sort(key=lambda r: r["repeat_idx"])
    print(f"\n=== E1 checkpoint boundary - MERGE batch_a + batch_b ({len(repeats_out)} powtorzen) ===")
    return _aggregate_checkpoint_results(repeats_out)
