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
import json
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
OUT_DIR = REPO_ROOT / "my-operations/Replica/diagnose_rlr_noise_out"  # gitignored, patrz .gitignore
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

    def __init__(self, materials_enabled, scene_path=SCENE_PATH, indirect_ray_count=None, thread_count=None):
        self.scene = str(scene_path)
        self.material_config = str(MATERIAL_CONFIG_PATH) if materials_enabled else None
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
              indirect_ray_count=None, thread_count=None):
    return tra.build_simulator(
        _Args(materials_enabled, _scene_path(scene_name), indirect_ray_count, thread_count)
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


EXPERIMENTS = {
    "p0": run_p0,
    "e1": run_e1,
    "e2_rays_vs_renders": run_e2_rays_vs_renders,
    "e2_ray_bias": run_e2_ray_bias,
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
