"""Sciezki projektu — JEDYNE zrodlo prawdy o ukladzie katalogow.

Kolejna zmiana struktury repo ma wymagac edycji wylacznie tego pliku. Wszystko
wyprowadzone z polozenia pakietu, wiec dziala niezaleznie od katalogu roboczego
(inaczej niz smoke-tests/).
"""

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # my-operations/echo_core
MY_OPS = PACKAGE_DIR.parent                            # my-operations
REPO_ROOT = MY_OPS.parent

# Sciezka do skryptu generatora podana JAWNIE, a nie przez __file__: trafia do
# atrybutu `script` kazdego pliku HDF5 i do komunikatow o wznawianiu, wiec musi
# wskazywac generator, a nie modul, z ktorego akurat jest odczytywana.
SCRIPT_PATH = MY_OPS / "generate_echo_dataset.py"

HABITAT_SIM_PY = REPO_ROOT / "habitat-sim/src_python"
SCENE_ROOT = REPO_ROOT / "sound-spaces/data/scene_datasets/replica"
METADATA_ROOT = MY_OPS / "metadata/replica"
LOCATIONS_PKL = REPO_ROOT / "scenes_ve_metadata_locations/scene_observations_128.pkl"
MATERIAL_CONFIG = MY_OPS / "replica_material_config.json"
CHIRP_PATH = MY_OPS / "sweep_audio/3ms_sweep.wav"
SPEC_DOC = MY_OPS / "docs/GENERATOR_PARAMS.md"

OUT_ROOT = REPO_ROOT / "outputs/echoes_36deg"  # gitignored (patrz .gitignore)

# `import habitat_sim` wymaga src_python na sciezce; robimy to tutaj, zeby
# skrypt dzialal takze bez recznie ustawionego PYTHONPATH (istotne dla
# --status/--verify odpalanych ad hoc w drugim terminalu).
if str(HABITAT_SIM_PY) not in sys.path:
    sys.path.insert(0, str(HABITAT_SIM_PY))
if str(MY_OPS) not in sys.path:
    sys.path.insert(0, str(MY_OPS))


def scene_mesh(scene):
    return SCENE_ROOT / scene / "habitat/mesh_semantic.ply"


def points_txt(scene):
    return METADATA_ROOT / scene / "points.txt"


def graph_pkl(scene):
    return METADATA_ROOT / scene / "graph.pkl"


# --- uklad katalogu wyjsciowego --------------------------------------------
# Kazda scena dostaje wlasny podkatalog:
#
#   outputs/echoes_36deg/<scena>/
#       <scena>.h5          dataset
#       generate.log        log czytelny
#       decisions.jsonl     jedna linia na lokalizacje
#       progress.json       sidecar dla --status
#       verify/             PNG-i z --verify
#
# Plaska struktura przy 18 scenach dawalaby kilkadziesiat plikow w jednym
# katalogu — latwo pomylic scene, trudno skopiowac albo skasowac jedna bez
# ryzyka. Nazwa pliku HDF5 celowo powtarza nazwe katalogu: ten jeden plik
# opuszcza katalog sceny (kopia na maszyne treningowa, backup), wiec sama jego
# nazwa musi mowic, co to jest. Pozostale pliki maja nazwy generyczne, bo nigdy
# nie sa ogladane poza swoim katalogiem.
#
# WSZYSTKIE sciezki wyjsciowe wyprowadzamy z scene_dir() — kolejna zmiana ukladu
# ma wymagac edycji w jednym miejscu.
def scene_dir(scene):
    return OUT_ROOT / scene


def scene_h5(scene):
    return scene_dir(scene) / f"{scene}.h5"


def scene_log(scene):
    return scene_dir(scene) / "generate.log"


def scene_decisions(scene):
    return scene_dir(scene) / "decisions.jsonl"


def scene_progress(scene):
    # Sidecar czytany przez --status w trakcie generacji: plik HDF5 jest wtedy
    # otwarty do zapisu przez inny proces, a maly JSON zapisywany atomowo
    # (tmp + rename) zawsze da sie przeczytac bez ryzyka.
    return scene_dir(scene) / "progress.json"


def scene_verify_dir(scene):
    return scene_dir(scene) / "verify"


# Census sondy (--probe-only): lekki, osobny katalog. NIE trafia do katalogow
# scen, bo to nie jest czesc datasetu — to pomiar, ktory ma odpowiedziec na
# pytanie o N_MAX, a nie dane treningowe.
PROBE_CENSUS_ROOT = REPO_ROOT / "outputs/probe_census"


def probe_census_csv(scene):
    return PROBE_CENSUS_ROOT / f"{scene}.csv"


def probe_census_log(scene):
    return PROBE_CENSUS_ROOT / f"{scene}.log"


def scene_stdout(scene):
    # Surowe stdout/stderr procesu odpalonego przez echo_ctl.py — inne niz
    # generate.log, bo lapie takze komunikaty habitat-sim i ewentualny traceback
    # z momentu, gdy logger jeszcze nie istnieje.
    return scene_dir(scene) / "stdout.txt"

