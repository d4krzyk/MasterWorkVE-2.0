#!/usr/bin/env python3
"""Diagnostyka szumu i wlasciwosci RLRAudioPropagation (Visual Echoes 2.0).

Uruchomienie (env `habitat` aktywne, PYTHONPATH wskazuje na habitat-sim/src_python
- patrz CLAUDE.md):

    conda activate habitat
    export PYTHONPATH=<repo>/habitat-sim/src_python:$PYTHONPATH
    python my-operations/diagnose_rlr_noise.py --exp p0
    python my-operations/diagnose_rlr_noise.py --exp e1
    python my-operations/diagnose_rlr_noise.py --exp p0 e1

Struktura: kazdy eksperyment to jedna funkcja rejestrowana w EXPERIMENTS,
dopisywana niezaleznie od pozostalych - dodanie e2/e3/e4/e5 w kolejnej sesji to
tylko nowa funkcja + wpis w rejestrze, bez zmian gdzie indziej. Wspolny raport
JSON (diagnostics_report.json) jest scalany po kazdym uruchomieniu, wiec
kolejne sesje moga dopisywac wyniki bez nadpisywania wczesniejszych.
"""

import argparse
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Kolejnosc importow (quaternion przed habitat_sim) wymagana przez lokalny
# patch tego repo - patrz habitat-sim/local_changes.patch / CLAUDE.md.
import quaternion  # noqa: F401
import habitat_sim

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rlr_audio as tra  # noqa: E402 - import z istniejacego smoke testu, nie duplikujemy logiki

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = REPO_ROOT / "sound-spaces/data/scene_datasets/replica/room_0/habitat/mesh_semantic.ply"
POINTS_PATH = REPO_ROOT / "my-operations/metadata/replica/room_0/points.txt"
# Brak dedykowanego pliku materialow akustycznych dla Replica w tym repo (tylko
# mp3d) - patrz CLAUDE.md. Uzywamy go mimo to: kategorie Replica bez dopasowania
# dostaja material domyslny (log "Using default material instead."), co NIE
# powoduje crasha (to wlasnie zweryfikowalismy w Zadaniu 0) - tylko oznacza,
# ze dokladnosc przypisania materialu jest tu nieistotna dla celu tego skryptu
# (determinizm/binauralnosc szumu), a nie sprawdzana.
MATERIAL_CONFIG_PATH = REPO_ROOT / "sound-spaces/data/mp3d_material_config.json"
OUT_DIR = REPO_ROOT / "outputs/diagnose_rlr_noise_out"  # gitignored, patrz .gitignore
REPORT_PATH = OUT_DIR / "diagnostics_report.json"

# enableMaterials=True jest bezpieczne na tym buildzie - zweryfikowane w Zadaniu 0
# (2026-07-23): fix w drzewie roboczym, binarka nowsza niz zrodlo, runtime bez
# SIGSEGV na room_0. Gdyby Zadanie 0 kiedys nie przeszlo, ta stala powinna
# wrocic na False z jawna adnotacja w raporcie (patrz sekcja audio w CLAUDE.md).
MATERIALS_ENABLED = True

# Dowolny punkt ze srodka points.txt, z dala od scian zewnetrznych room_0
# (id=0..4 sa blisko krawedzi siatki wg wizualnej inspekcji points.txt).
LISTENER_POINT_ID = 50


class _Args:
    """Namespace kompatybilny z test_rlr_audio.build_simulator()."""

    def __init__(self, materials_enabled, scene_path=SCENE_PATH, indirect_ray_count=None, thread_count=None,
                 material_config=None, sensor_height=None):
        self.scene = str(scene_path)
        # None => build_simulator uzyje 1.5 m (wysokosc calej wczesniejszej
        # charakteryzacji). Eksperymenty produkcyjne podaja 1.25 m - patrz
        # PRODUCTION_SENSOR_HEIGHT.
        self.sensor_height = sensor_height
        # material_config=None => domyslny mp3d (konfiguracja calej dotychczasowej
        # charakteryzacji). Nadpisujemy tylko tam, gdzie porownujemy same configi
        # materialow miedzy soba (Blok B).
        self.material_config = str(material_config or MATERIAL_CONFIG_PATH) if materials_enabled else None
        self.out_dir = str(OUT_DIR)
        # None => build_simulator uzyje swoich domyslnych (500 promieni, 1 watek),
        # czyli konfiguracji, na ktorej oparta jest cala reszta charakteryzacji.
        # Nadpisujemy je tylko w E2, ktore z definicji przemiata liczbe promieni.
        self.indirect_ray_count = indirect_ray_count
        self.thread_count = thread_count


def _scene_path(scene_name):
    return REPO_ROOT / f"sound-spaces/data/scene_datasets/replica/{scene_name}/habitat/mesh_semantic.ply"


def _points_path(scene_name):
    return REPO_ROOT / f"my-operations/metadata/replica/{scene_name}/points.txt"


def build_sim(materials_enabled=MATERIALS_ENABLED, scene_name="room_0",
              indirect_ray_count=None, thread_count=None, material_config=None, sensor_height=None):
    return tra.build_simulator(
        _Args(materials_enabled, _scene_path(scene_name), indirect_ray_count, thread_count,
              material_config, sensor_height)
    )


def load_point_position(sim, scene_name, point_id):
    """points.txt (danej sceny) -> pozycja habitat, zweryfikowana na navmeshu.

    Konwersja (a, b) -> (x=a, z=-b) i wysokosc z pathfindera (points.txt nie
    zapisuje y) zgodnie ze sprawdzonym empirycznie wzorem z
    my-operations/rlraudiotest.py.
    """
    points = pd.read_csv(_points_path(scene_name), sep="\t", header=None, names=["id", "a", "b", "c"])
    row = points.iloc[point_id]
    x, z = float(row["a"]), -float(row["b"])

    bounds_min, _bounds_max = sim.pathfinder.get_bounds()
    y_guess = float(bounds_min[1]) + 0.1

    pos = sim.pathfinder.snap_point([x, y_guess, z])
    if not sim.pathfinder.is_navigable(pos):
        raise RuntimeError(
            f"Punkt id={point_id} sceny {scene_name} nie lezy na navmeshu po snap_point: {pos}"
        )
    return np.array(pos)


def load_listener_position(sim):
    """Kompatybilnosc wsteczna dla P0/E1/E1-extended: zawsze room_0, LISTENER_POINT_ID."""
    return load_point_position(sim, "room_0", LISTENER_POINT_ID)


def render_raw(sim, position, angle_deg, material_config):
    """Echolokacja (zrodlo=odbiornik) -> surowy obs['audio_sensor'] (kanaly, probki)."""
    obs, _listener_pos, _rot = tra.phase3_echolocation(sim, position, angle_deg, material_config)
    return np.array(obs["audio_sensor"])


def _material_config_arg():
    return str(MATERIAL_CONFIG_PATH) if MATERIALS_ENABLED else None


# --- P0: bramka wstepna - binauralnosc ---------------------------------------


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


# --- E1: determinizm ziarna losowego (BLOKUJACY) -----------------------------


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


# --- E1 checkpoint boundary: czy restart miedzy DWIEMA ROZNYMI probkami -------
# koreluje ich szum? (wezsze pytanie niz podstawowe E1 - tam sprawdzalismy tylko
# powtorzenie IDENTYCZNEJ sekwencji wywolan po restarcie; tu sprawdzamy, czy
# restart miedzy roznymi probkami (inna lokalizacja/kat) nadal daje nieskorelowany
# szum, czy "talia kart" jest zablokowana do indeksu wywolania niezaleznie od tego,
# co faktycznie jest renderowane.)

CHIRP_PATH = REPO_ROOT / "my-operations/sweep_audio/3ms_sweep.wav"

CHECKPOINT_N = 8  # renderow na probke
CHECKPOINT_R = 16  # powtorzen (rozne pary lokalizacja/kat) - eskalowane 4->8->16 po niejednoznacznych/granicznych wynikach (2026-07-24)
CHECKPOINT_CORR_THRESHOLD = 0.05  # umowny prog "zauwazalnej korelacji" (patrz uzasadnienie w opisie zadania)

# Rozne pary (lokalizacja, kat) per powtorzenie, cztery na kazda ze scen z
# charakteryzacji szumu z 07-20 (room_0, apartment_1, office_0, frl_apartment_0),
# dla odpornosci wniosku na konkretna geometrie sceny. Id punktow dobrane w
# granicach dlugosci points.txt kazdej sceny (room_0: 136, apartment_1: 257,
# office_0: 65, frl_apartment_0: 237 wierszy), bez szczegolnej selekcji poza
# "rozne od siebie" - dokladna geometria nie ma znaczenia dla testu korelacji.
CHECKPOINT_REPEAT_CONFIGS = (
    {"scene": "room_0", "loc1_id": 30, "angle1": 0.0, "loc2_id": 100, "angle2": 180.0},
    {"scene": "apartment_1", "loc1_id": 20, "angle1": 45.0, "loc2_id": 200, "angle2": 270.0},
    {"scene": "office_0", "loc1_id": 5, "angle1": 90.0, "loc2_id": 55, "angle2": 10.0},
    {"scene": "frl_apartment_0", "loc1_id": 15, "angle1": 200.0, "loc2_id": 180, "angle2": 60.0},
    {"scene": "room_0", "loc1_id": 60, "angle1": 270.0, "loc2_id": 5, "angle2": 90.0},
    {"scene": "apartment_1", "loc1_id": 100, "angle1": 10.0, "loc2_id": 150, "angle2": 190.0},
    {"scene": "office_0", "loc1_id": 20, "angle1": 150.0, "loc2_id": 45, "angle2": 330.0},
    {"scene": "frl_apartment_0", "loc1_id": 60, "angle1": 20.0, "loc2_id": 120, "angle2": 280.0},
    {"scene": "room_0", "loc1_id": 45, "angle1": 120.0, "loc2_id": 90, "angle2": 300.0},
    {"scene": "apartment_1", "loc1_id": 50, "angle1": 160.0, "loc2_id": 230, "angle2": 320.0},
    {"scene": "office_0", "loc1_id": 10, "angle1": 200.0, "loc2_id": 60, "angle2": 50.0},
    {"scene": "frl_apartment_0", "loc1_id": 90, "angle1": 140.0, "loc2_id": 200, "angle2": 350.0},
    {"scene": "room_0", "loc1_id": 10, "angle1": 200.0, "loc2_id": 125, "angle2": 40.0},
    {"scene": "apartment_1", "loc1_id": 5, "angle1": 80.0, "loc2_id": 180, "angle2": 250.0},
    {"scene": "office_0", "loc1_id": 30, "angle1": 280.0, "loc2_id": 1, "angle2": 110.0},
    {"scene": "frl_apartment_0", "loc1_id": 40, "angle1": 310.0, "loc2_id": 150, "angle2": 100.0},
)


def _get_spec(sim, position, angle_deg, material_config, chirp):
    raw = render_raw(sim, position, angle_deg, material_config)
    rir = np.transpose(raw)
    _echo, spec = tra.render_spectrogram(rir, chirp)
    return spec


def _render_n_specs(sim, position, angle_deg, material_config, chirp, n):
    return [_get_spec(sim, position, angle_deg, material_config, chirp) for _ in range(n)]


def _residuals(specs):
    mean_spec = np.mean(specs, axis=0)
    return [s - mean_spec for s in specs]


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
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)

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


# --- E4: skad bierze sie zmienna dlugosc IR ----------------------------------
#
# Motywacja: w raportach z poprzednich sesji IR ma ROZNE dlugosci miedzy
# renderami ([2, 60134] i [2, 65617] w sekcji e1, [1, 71327] w rlraudiotest.py).
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
    result = {"materials_enabled": MATERIALS_ENABLED, "echo_window_samples": tra.ECHO_SAMPLES}
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
                                "ir_ms": n / tra.SAMPLE_RATE * 1000.0})
            print(f"  id={pid:4d} pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}): {n:7d} probek "
                  f"= {n / tra.SAMPLE_RATE * 1000.0:8.1f} ms")
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
                             "ir_samples": n, "ir_ms": n / tra.SAMPLE_RATE * 1000.0})
            print(f"  {scene:18s}: bbox={extent[0]:5.1f}x{extent[1]:5.1f}x{extent[2]:5.1f} m "
                  f"(V={volume:7.1f} m3) -> IR {n:7d} probek = {n / tra.SAMPLE_RATE * 1000.0:8.1f} ms")
        finally:
            s.close()
    result["by_scene"] = by_scene

    # --- werdykt ---
    all_lengths = lens_pos + lens_rep + [d["ir_samples"] for d in by_scene]
    min_len = int(min(all_lengths))
    longer_than_window = bool(min_len > tra.ECHO_SAMPLES)
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
          f"({min_len / tra.SAMPLE_RATE * 1000.0:.1f} ms) vs okno echa {tra.ECHO_SAMPLES} probek (60 ms)")
    if longer_than_window:
        print(f"  => Kazdy IR jest DLUZSZY niz okno 60 ms, wiec po przycieciu do okna zmienna dlugosc "
              f"nie ma znaczenia dla spektrogramu (margines {min_len - tra.ECHO_SAMPLES} probek).")
    else:
        print(f"  => UWAGA: co najmniej jeden IR jest KROTSZY niz okno 60 ms - przyciecie dopelnia zerami, "
              f"co realnie zmienia dane. Trzeba to uwzglednic.")
    return result


# --- E3: domena usredniania --------------------------------------------------
#
# Trzy kandydujace estymatory na spektrogram z N renderow:
#   mag  = (1/N) * suma |STFT(echo_i)|            - usrednianie magnitud
#   en   = sqrt( (1/N) * suma |STFT(echo_i)|^2 )  - usrednianie energii
#   time = |STFT( (1/N) * suma rir_i )|           - usrednianie surowych RIR
#
# Kryterium jest operacyjne, bez sztucznego "ground truth": nie porownujemy do
# usrednienia M=50 jednym ze wzorow, bo taka referencja z definicji faworyzuje
# ten wzor. Zamiast tego mierzymy osobno SYGNAL (roznica miedzy sasiednimi
# katami - to chcemy zachowac) i SZUM (roznica dwoch niezaleznych estymat tego
# samego kata - to chcemy zredukowac), i porownujemy ich iloraz.

E3_N = 10               # renderow na jedna estymate
E3_M = 2 * E3_N         # renderow na kat - dwie ROZLACZNE polowki, zeby zmierzyc szum
E3_ANGLES = (0.0, 10.0)  # sasiednie katy z docelowej siatki 36 orientacji
E3_POSITION_IDS = (30, 50, 80)


def _align_rirs(rirs, mode):
    """Wyrownanie RIR-ow o roznej dlugosci do wspolnego ksztaltu.

    Potrzebne WYLACZNIE dla estymatora "time" - usrednianie w dziedzinie czasu
    wymaga dodania surowych przebiegow, a te maja rozna dlugosc (patrz E4).
    Estymatory "mag"/"en" dzialaja na spektrogramach juz przycietych do 60 ms,
    wiec sa od tego problemu wolne - i to jest argument przeciw "time"
    niezalezny od jego jakosci statystycznej.
    """
    lengths = [r.shape[0] for r in rirs]
    if mode == "trunc":
        n = min(lengths)
        return np.mean([r[:n] for r in rirs], axis=0)
    if mode == "pad":
        n = max(lengths)
        padded = [np.pad(r, ((0, n - r.shape[0]), (0, 0))) for r in rirs]
        return np.mean(padded, axis=0)
    raise ValueError(f"nieznany tryb wyrownania: {mode}")


def _estimator(kind, specs, rirs, chirp):
    """Jedna estymata spektrogramu z zestawu N renderow."""
    if kind == "mag":
        return np.mean(specs, axis=0)
    if kind == "en":
        return np.sqrt(np.mean(np.square(specs), axis=0))
    if kind.startswith("time_"):
        mean_rir = _align_rirs(rirs, kind.split("_", 1)[1])
        _echo, spec = tra.render_spectrogram(mean_rir, chirp)
        return spec
    raise ValueError(f"nieznany estymator: {kind}")


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_e3_averaging_domain():
    print("\n=== E3: domena usredniania ===")
    import librosa

    material_config = _material_config_arg()
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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
                    _echo, spec = tra.render_spectrogram(rir, chirp)
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
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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
                        _echo, spec = tra.render_spectrogram(np.transpose(raw), chirp)
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
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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


# --- BLOK A: czy obciazenie od liczby promieni ZALEZY OD ORIENTACJI? ---------
#
# E2b wykazal, ze 500 promieni daje estymate systematycznie przesunieta wzgledem
# 5000 (obciazenie 0.0220 RMSE = 34% sygnalu 10 stopni, energia -2.2%) i ze
# usrednianie tego nie usuwa. Zostalo jednak pytanie, ktore jako jedyne moze
# UNIEWAZNIC glowna teze pracy:
#
#   czy to obciazenie jest JEDNORODNE po orientacjach, czy zalezy od kata?
#
# - jednorodne  -> ten sam renderer stoi po obu stronach ablacji (36 vs 4 katy),
#                  obciazenie w duzej mierze sie skraca, porownanie zostaje wazne,
#                  a rzecz jest tylko zastrzezeniem o realizmie bezwzglednym;
# - zalezne od kata -> silnik wstrzykuje sztuczny sygnal ROZNICUJACY ORIENTACJE,
#                  ktory wyglada jak geometria pokoju, a jest artefaktem samplingu
#                  Monte Carlo. To zatruwa dokladnie ten efekt, ktory mierzymy.
#
# Konfiguracja celowo taka sama jak w calej dotychczasowej charakteryzacji
# (mp3d_material_config.json, threadCount=1) - inaczej wynik nie bylby porownywalny
# z E2/E2b. threadCount=1 dodatkowo dlatego, ze watki lamia determinizm RNG i
# podnosza szum o 22-35%, a tutaj mierzymy male roznice na tle szumu.
#
# Metoda: dla kazdej pozycji i kazdej z 36 orientacji renderujemy N razy przy 500
# i N razy przy 5000 promieniach. Kazdy zestaw N dzielimy na dwie rozlaczne polowki
# (A/B) - polowki sluza za KONTROLE: pokazuja, o ile dana metryka skacze z samego
# szumu, przy TYM SAMYM rayCount. Bez tej kontroli rozrzut po katach jest
# nieinterpretowalny, bo nie wiadomo, czy to orientacja czy Monte Carlo.

E2BO_RAYS = (500, 5000)
E2BO_N = 8                     # renderow na (pozycja, kat, rayCount); dzielone 4+4 na kontrole
E2BO_ANGLES = tuple(float(a) for a in range(0, 360, 10))  # docelowa siatka 36 katow
# Dwie sceny o roznej akustyce: room_0 (baza calej charakteryzacji) i office_0
# (najglosniejsza w pomiarach z 07-20 - inny czas pogloru, inna geometria).
E2BO_POSITIONS = (("room_0", 50), ("office_0", 30))


def _energy(spec):
    """Calkowita energia spektrogramu - skalarny wskaznik 'ile echa' bez ksztaltu."""
    return float(np.sum(np.square(spec.astype(np.float64))))


def _rms(spec):
    return float(np.sqrt(np.mean(np.square(spec.astype(np.float64)))))


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
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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
    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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
REPLICA_MATERIAL_CONFIG = REPO_ROOT / "my-operations/replica_material_config.json"


@contextlib.contextmanager
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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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


# --- BLOK 1: czy threadCount zmienia ESTYMATOR, czy tylko tempo? -------------
#
# Dwie zmierzone liczby nie trzymaja sie razem fizyki:
#  (a) 2.57 s (1 watek) vs 0.2739 s (8 watkow) przy 5000 promieni = 9.4x
#      przyspieszenia na 8 watkach. Superliniowo - niemozliwe dla czystej
#      paralelizacji TEJ SAMEJ pracy.
#  (b) watki podnosza szum o 22-35%. Gdyby to byla ta sama estymata policzona
#      szybciej, szum bylby IDENTYCZNY, nie wyzszy.
# Obie anomalie wskazuja, ze threadCount moze ZMIENIAC to, co jest liczone (np.
# dzielic budzet promieni miedzy watki zamiast go zwielokrotniac). Wtedy 8 watkow
# daje gorsza jakosc szybciej, a nie te sama jakosc szybciej - i wpisanie
# threadCount>1 do generatora byloby cicha utrata jakosci.
#
# Kryterium glowne: ENERGIA (srednia po ~85 tys. komorek spektrogramu, o rzedy
# wielkosci mniej zaszumiona niz RMSE). RMSE sluzy jako miara wielkosci roznicy,
# po dekompozycji efekt = sqrt(RMSE^2 - sigma_1^2 - sigma_8^2) - porownywanie
# surowych RMSE dwoch zaszumionych estymat jest bledem (wystapil raz w Bloku B).

PRODUCTION_SENSOR_HEIGHT = 1.25  # kamera z pkl; patrz PKL_FORMAT.md i listener_height
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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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


# --- BLOK 2: czy tysiace renderow w JEDNEJ instancji ciekna? -----------------
#
# Wiemy, ze ~30 KONSTRUKCJI Simulatora w jednym procesie kladzie karte sprzetowo
# (wyciek GL/EGL na sim.close(), procedura odzysku przez PCI FLR w CLAUDE.md).
# To jednak inny wzorzec zuzycia zasobow niz tysiace RENDEROW w jednej instancji -
# a to wlasnie jest wzorzec produkcyjny ("jeden dlugo zyjacy Simulator na scene").
# Jesli render tez cieknie, architektura wymaga rotacji instancji w obrebie sceny
# (co jest bezpieczne dzieki werdyktowi BEZPIECZNY z e1_checkpoint_boundary_merge).
#
# Mierzymy trzy rzeczy naraz, bo wyciek moze sie ujawnic w kazdej z osobna:
#  - pamiec GPU (karta),
#  - RSS procesu (host - alokacje po stronie CPU tez potrafia rosnac),
#  - czas renderu (czesto rosnie WCZESNIEJ niz sama pamiec, np. przy narastajacej
#    liczbie zywych obiektow do przejrzenia).

GPUMEM_SCENE = "room_0"
GPUMEM_RENDERS = 3000
GPUMEM_SAMPLE_EVERY = 100
GPUMEM_RAYS = 500
GPUMEM_THREADS = 1


def _gpu_mem_mib():
    """Pamiec GPU zajeta lacznie (MiB) - z nvidia-smi, bo habitat trzyma kontekst
    przez EGL/GL, a nie jako 'compute app', wiec per-proces czesto nie widac."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


def _proc_rss_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def _largest_scene_render_count(n_renders_per_sample):
    """Ile renderow przypadnie na najwieksza scene - cel, do ktorego ekstrapolujemy."""
    biggest, best = None, -1
    for sd in sorted((REPO_ROOT / "my-operations/metadata/replica").iterdir()):
        pf = sd / "points.txt"
        if not pf.exists():
            continue
        n = sum(1 for _ in pf.open())
        if n > best:
            biggest, best = sd.name, n
    return biggest, best, best * 36 * n_renders_per_sample


def run_gpu_memory_scale():
    print("\n=== BLOK 2: pamiec GPU i RSS przy tysiacach runSimulation() ===")
    import librosa

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
    mc = str(REPLICA_MATERIAL_CONFIG)
    n_prod = int(os.environ.get("VE_N_RENDERS", "10"))  # N z Bloku 1
    big_scene, big_points, big_renders = _largest_scene_render_count(n_prod)
    print(f"  Najwieksza scena: {big_scene} ({big_points} lokalizacji) -> "
          f"{big_points} x 36 katow x N={n_prod} = {big_renders} renderow w jednej instancji")

    result = {"scene": GPUMEM_SCENE, "renders": GPUMEM_RENDERS, "sample_every": GPUMEM_SAMPLE_EVERY,
              "rays": GPUMEM_RAYS, "threads": GPUMEM_THREADS, "sensor_height": PRODUCTION_SENSOR_HEIGHT,
              "n_renders_per_sample_assumed": n_prod,
              "largest_scene": big_scene, "largest_scene_points": big_points,
              "largest_scene_renders": big_renders,
              "gpu_total_mib": None}

    points = pd.read_csv(_points_path(GPUMEM_SCENE), sep="\t", header=None, names=["id", "a", "b", "c"])
    n_points = len(points)
    baseline_gpu = _gpu_mem_mib()
    samples = []

    sim = build_sim(scene_name=GPUMEM_SCENE, indirect_ray_count=GPUMEM_RAYS, thread_count=GPUMEM_THREADS,
                    material_config=REPLICA_MATERIAL_CONFIG, sensor_height=PRODUCTION_SENSOR_HEIGHT)
    try:
        # Cache pozycji, zeby snap_point nie mieszal sie do pomiaru czasu renderu.
        pos_cache = {}
        for i in range(GPUMEM_RENDERS):
            # Wzorzec produkcyjny: przechodzimy po lokalizacjach i 36 katach, a nie
            # renderujemy w kolko jednego stanu - inny stan to inna praca silnika.
            pid = (i // 36) % n_points
            ang = float((i % 36) * 10)
            if pid not in pos_cache:
                pos_cache[pid] = load_point_position(sim, GPUMEM_SCENE, pid)
            t0 = time.perf_counter()
            _get_spec(sim, pos_cache[pid], ang, mc, chirp)
            dt = time.perf_counter() - t0
            if i % GPUMEM_SAMPLE_EVERY == 0 or i == GPUMEM_RENDERS - 1:
                samples.append({"render": i, "gpu_mib": _gpu_mem_mib(), "rss_mib": _proc_rss_mib(),
                                "s_per_render": dt})
                s = samples[-1]
                print(f"    render {i:5d}: GPU {s['gpu_mib']:.0f} MiB, RSS {s['rss_mib']:.0f} MiB, "
                      f"{s['s_per_render']:.3f} s", flush=True)
    finally:
        sim.close()

    result["baseline_gpu_mib"] = baseline_gpu
    result["samples"] = samples
    result["gpu_total_mib"] = 16303.0  # RTX 5070 Ti

    # --- analiza trendu -------------------------------------------------------
    # Pierwsze 5 probek (500 renderow) to faza rozgrzewki - alokacje buforow,
    # cache'e sceny. Trend liczymy DOPIERO po niej, inaczej rozgrzewka udaje wyciek.
    warm = [s for s in samples if s["render"] >= 500]
    x = np.array([s["render"] for s in warm], dtype=np.float64)
    trends = {}
    for key in ("gpu_mib", "rss_mib", "s_per_render"):
        y = np.array([s[key] for s in warm], dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else float("nan")
        trends[key] = {"per_1000_renders": slope * 1000.0,
                       "first": float(y[0]), "last": float(y[-1]),
                       "min": float(y.min()), "max": float(y.max()),
                       "std": float(y.std())}
    result["trends"] = trends

    _plot_gpu_memory(samples, result)

    gpu_rate = trends["gpu_mib"]["per_1000_renders"]
    rss_rate = trends["rss_mib"]["per_1000_renders"]
    time_rate = trends["s_per_render"]["per_1000_renders"]
    headroom = result["gpu_total_mib"] - trends["gpu_mib"]["last"]
    # Prog: rozrzut samego pomiaru nvidia-smi to kilkanascie MiB (desktop tez
    # alokuje), wiec za wyciek uznajemy dopiero tempo istotnie ponad ten szum.
    gpu_leaks = gpu_rate > 3 * trends["gpu_mib"]["std"] and gpu_rate > 10.0
    rss_leaks = rss_rate > 3 * trends["rss_mib"]["std"] and rss_rate > 10.0
    time_grows = time_rate > 0.02 * trends["s_per_render"]["first"]

    print("\n--- WERDYKT BLOK 2 ---")
    print(f"  GPU: {trends['gpu_mib']['first']:.0f} -> {trends['gpu_mib']['last']:.0f} MiB "
          f"(tempo {gpu_rate:+.1f} MiB / 1000 renderow, rozrzut {trends['gpu_mib']['std']:.1f})")
    print(f"  RSS: {trends['rss_mib']['first']:.0f} -> {trends['rss_mib']['last']:.0f} MiB "
          f"(tempo {rss_rate:+.1f} MiB / 1000 renderow, rozrzut {trends['rss_mib']['std']:.1f})")
    print(f"  czas/render: {trends['s_per_render']['first']:.3f} -> {trends['s_per_render']['last']:.3f} s "
          f"(tempo {time_rate:+.4f} s / 1000 renderow)")
    if not gpu_leaks and not rss_leaks and not time_grows:
        verdict = ("STABILNA - po fazie rozgrzewki ani pamiec GPU, ani RSS procesu, ani czas renderu nie "
                   f"rosna w tempie odrozniamym od szumu pomiaru na {GPUMEM_RENDERS} renderach. "
                   "Architektura 'jeden Simulator na scene' jest bezpieczna bez rotacji instancji.")
    else:
        parts = []
        if gpu_leaks:
            n_to_oom = headroom / gpu_rate * 1000.0
            parts.append(f"GPU rosnie {gpu_rate:.1f} MiB/1000 renderow; przy zapasie {headroom:.0f} MiB "
                         f"pamieci zabraknie po ~{n_to_oom:.0f} renderach "
                         f"({'PRZED' if n_to_oom < big_renders else 'PO'} koncu najwiekszej sceny, "
                         f"ktora wymaga {big_renders})")
        if rss_leaks:
            parts.append(f"RSS rosnie {rss_rate:.1f} MiB/1000 renderow")
        if time_grows:
            parts.append(f"czas renderu rosnie {time_rate:+.4f} s/1000 renderow")
        verdict = "ROSNIE - " + "; ".join(parts) + "."
    result["verdict"] = verdict
    print(f"\n  {verdict}")
    return result


def _plot_gpu_memory(samples, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [s["render"] for s in samples]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, key, label, color in ((axes[0], "gpu_mib", "pamiec GPU [MiB]", "tab:red"),
                                  (axes[1], "rss_mib", "RSS procesu [MiB]", "tab:blue"),
                                  (axes[2], "s_per_render", "czas renderu [s]", "tab:green")):
        y = [s[key] for s in samples]
        ax.plot(x, y, "o-", ms=3, lw=1.2, color=color)
        ax.axvline(500, color="k", ls=":", lw=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{result['scene']}, {result['renders']} renderow w JEDNEJ instancji "
                      f"({result['rays']} promieni / {result['threads']} watek)\n"
                      "linia kropkowana = koniec fazy rozgrzewki (500 renderow)", fontsize=10)
    axes[2].set_xlabel("numer renderu")
    fig.tight_layout()
    path = OUT_DIR / "gpu_memory_scale.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    result["plot_path"] = str(path)
    print(f"\n  Wykres zapisany: {path}")


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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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


# --- Podloga szumu w scenach, ktorych nigdy nie zmierzono ---------------------
#
# Po co: N_MAX = 40 odpowiada progowi sigma_1 = sqrt(40)*0.0644/3.5 = 0.1164.
# Powyzej tego progu regula adaptacyjna zazada wiecej niz 40 renderow, zostanie
# obcieta i probka moze nie osiagnac SNR 3.5 (GENERATOR_PARAMS.md §5 ogr. 6).
# Prog wybrano na podstawie udokumentowanego zakresu do sigma_1 = 0.1131, ale
# CLAUDE.md notuje szum render-do-renderu do 0.16 RMSE, czyli sigma_1 do 0.1131
# przy dwoch renderach — a 11 z 18 scen (1227 z 1740 lokalizacji, 71 %) nie ma
# ZADNEGO pomiaru. Na office_1 (mediana sigma_1 = 0.0766) juz jedna probka na 576
# dobila do limitu i nie dobila do progu. Scena z mediana 0.12 dawalaby
# wiekszosc probek przy limicie. To pytanie o poprawnosc zbioru, nie o harmonogram.
#
# Konfiguracja jest DOKLADNIE produkcyjna, inaczej pomiar nie odpowiadalby temu,
# co zobaczy generator:
#   - pozycja z generate_echo_dataset.load_scene_locations(): x,z z points.txt,
#     y z graph.pkl (NIE snap_point — ten daje ~0.21 m wyzej, patrz
#     GENERATOR_PARAMS.md §2 poprawka 2026-07-28)
#   - sensor 1.25 m, 500 promieni, 1 watek, materialy Repliki
#   - run_simulation=False (1 symulacja audio na render, §4.3)
#   - WARMUP_DISCARD renderow odrzuconych po konstrukcji Simulatora (§2)
#
# sigma_1 liczymy estymatorem WARIANCYJNYM po wszystkich renderach, nie
# polowkowym: tu chodzi o dokladnosc referencji, a polowkowy ma SD ~5 % z sufitem
# niezaleznym od liczby renderow (GENERATOR_PARAMS.md §3.4).

REMAINING_FRACTIONS = (0.20, 0.75)   # ta sama konwencja co NOISEFLOOR_FRACTIONS
REMAINING_M = 20                     # renderow na pozycje
REMAINING_ANGLE = 0.0


def _sigma_variance(specs):
    """sigma_1 = sqrt(srednia po komorkach z Var po renderach). n-1 st. swobody."""
    arr = np.stack(specs).astype(np.float64)
    return float(np.sqrt(np.var(arr, axis=0, ddof=1).mean()))


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

    chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)
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
                tra.phase3_echolocation(sim, positions[pids[0]], REMAINING_ANGLE, mc,
                                        run_simulation=False)
            first = True
            for pid in pids:
                specs = []
                for _ in range(REMAINING_M):
                    obs, _lp, _rot = tra.phase3_echolocation(
                        sim, positions[pid], REMAINING_ANGLE, mc if first else None,
                        run_simulation=False)
                    first = False
                    _echo, spec = tra.render_spectrogram(
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


EXPERIMENTS = {
    "noise_floor_remaining": run_noise_floor_remaining,
    "p0": run_p0,
    "noise_floor_orientation": run_noise_floor_orientation,
    "materials_verify": run_materials_verify,
    "signal_noise_recheck": run_signal_noise_recheck,
    "e2_thread_estimator": run_e2_thread_estimator,
    "e2_thread_effective_rays": run_e2_thread_effective_rays,
    "e2_thread_budget_confirm": run_e2_thread_budget_confirm,
    "gpu_memory_scale": run_gpu_memory_scale,
    "noise_floor_scenes": run_noise_floor_scenes,
    "e1": run_e1,
    "e2_rays_vs_renders": run_e2_rays_vs_renders,
    "e2_ray_bias": run_e2_ray_bias,
    "e2_bias_orientation": run_e2_bias_orientation,
    "e3_averaging_domain": run_e3_averaging_domain,
    "e4_ir_length": run_e4_ir_length,
    "listener_height": run_listener_height,
    "e1_checkpoint_boundary": run_e1_checkpoint_boundary,
    "e1_checkpoint_boundary_batch_a": run_e1_checkpoint_boundary_batch_a,
    "e1_checkpoint_boundary_batch_b": run_e1_checkpoint_boundary_batch_b,
    "e1_checkpoint_boundary_merge": run_e1_checkpoint_boundary_merge,
    "e1_extended": run_e1_extended,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=sorted(EXPERIMENTS.keys()),
        required=True,
        help="Ktore eksperymenty uruchomic (mozna kilka na raz).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            report = json.load(f)

    for name in args.exp:
        report[name] = EXPERIMENTS[name]()

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nRaport zapisany w: {REPORT_PATH}")
    print("\n=== PODSUMOWANIE ===")
    for name in args.exp:
        status = report[name].get("status", "-")
        print(f"  {name}: status={status}")


if __name__ == "__main__":
    main()
