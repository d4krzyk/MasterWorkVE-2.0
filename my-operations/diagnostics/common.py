"""Wspolne dla wszystkich eksperymentow: sciezki, budowa Simulatora,
pozycje, pojedynczy render i estymatory."""

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
    """Namespace kompatybilny z echo_core.audio.build_simulator()."""

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
    return build_simulator(
        _Args(materials_enabled, _scene_path(scene_name), indirect_ray_count, thread_count,
              material_config, sensor_height)
    )

def load_point_position(sim, scene_name, point_id):
    """points.txt (danej sceny) -> pozycja habitat, zweryfikowana na navmeshu.

    Konwersja (a, b) -> (x=a, z=-b) i wysokosc z pathfindera (points.txt nie
    zapisuje y) zgodnie ze sprawdzonym empirycznie wzorem z
    my-operations/tools/rlr_minimal_example.py.
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
    obs, _listener_pos, _rot = phase3_echolocation(sim, position, angle_deg, material_config)
    return np.array(obs["audio_sensor"])

def _material_config_arg():
    return str(MATERIAL_CONFIG_PATH) if MATERIALS_ENABLED else None

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
    _echo, spec = render_spectrogram(rir, chirp)
    return spec

def _render_n_specs(sim, position, angle_deg, material_config, chirp, n):
    return [_get_spec(sim, position, angle_deg, material_config, chirp) for _ in range(n)]

def _residuals(specs):
    mean_spec = np.mean(specs, axis=0)
    return [s - mean_spec for s in specs]

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
        _echo, spec = render_spectrogram(mean_rir, chirp)
        return spec
    raise ValueError(f"nieznany estymator: {kind}")

def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

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

REPLICA_MATERIAL_CONFIG = REPO_ROOT / "my-operations/replica_material_config.json"

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
