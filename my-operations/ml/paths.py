"""Jedno miejsce, z ktorego wyprowadzane sa wszystkie sciezki fazy uczenia.

Ta sama zasada, co `echo_core/paths.py` w generatorze: skrypty NIE sklejaja
wlasnych sciezek. Kolejna zmiana ukladu katalogow to wtedy edycja w jednym
pliku, a nie polowanie na literaly po calym repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

# my-operations/ml/paths.py -> my-operations/ml -> my-operations -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

MY_OPERATIONS = REPO_ROOT / "my-operations"
PARIDA_ROOT = REPO_ROOT / "beyond-image-to-depth"
OUTPUTS = REPO_ROOT / "outputs"

# Katalogi datasetu -- po jednym na wariant geometrii (GENERATOR_PARAMS.md §4.5).
DATASET_DIRS = {
    "main": OUTPUTS / "echoes_36deg",
    "patched": OUTPUTS / "echoes_36deg_patched",
}

# Referencyjny zbior lokalizacji VisualEchoes (Gao i in., ECCV 2020).
SCENE_OBSERVATIONS_PKL = REPO_ROOT / "scenes_ve_metadata_locations" / "scene_observations_128.pkl"

# Wyjscia fazy uczenia. Wszystko ladu je w outputs/, nic nie powstaje w my-operations/.
ML_OUTPUTS = OUTPUTS / "ml"
SPLITS_DIR = ML_OUTPUTS / "splits"
BENCH_DIR = ML_OUTPUTS / "bench"
VERIFY_DIR = ML_OUTPUTS / "verify_loader"
RUNS_DIR = ML_OUTPUTS / "runs"

SPEC_DOC = MY_OPERATIONS / "docs" / "GENERATOR_PARAMS.md"


def scene_h5(scene: str, variant: str) -> Path:
    """Sciezka do pliku HDF5 danej sceny w danym wariancie.

    Wariant `patched` obejmuje tylko 10 scen nieszczelnych; pozostale 8 jest
    akustycznie szczelnych, ich siatka jest w obu wariantach TYM SAMYM plikiem,
    wiec przy threadCount=1 echa bylyby bit-identyczne (GENERATOR_PARAMS.md
    §4.5). Dlatego dla scen szczelnych fizycznie czytamy plik z `main` -- to nie
    jest obejscie braku danych, tylko konsekwencja tego, ze te dane sa te same.
    """
    root = DATASET_DIRS[variant]
    p = root / scene / f"{scene}.h5"
    if not p.exists() and variant == "patched":
        p = DATASET_DIRS["main"] / scene / f"{scene}.h5"
    return p


def add_parida_to_syspath() -> Path:
    """Wstawia `beyond-image-to-depth/` na sys.path, zeby dzialalo `from models import ...`.

    Repo Paridy uzywa importow wzglednych wobec swojego korzenia (`from models
    import networks`), a nie pakietowych, wiec bez tego nie da sie go uzyc jako
    biblioteki bez modyfikacji jego plikow -- a modyfikowac ich nie wolno.
    """
    p = str(PARIDA_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return PARIDA_ROOT
