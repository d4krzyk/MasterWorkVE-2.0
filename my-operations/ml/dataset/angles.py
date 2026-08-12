"""Filtr orientacji -- zmienna niezalezna calej pracy.

Zbior 36 orientacji co 10 stopni zostal wybrany m.in. dlatego, ze 36 dzieli sie
bez reszty przez 2, 3, 4, 6, 9, 12 i 18. Dzieki temu cala krzywa nasycenia
4 / 6 / 9 / 12 / 18 / 36 powstaje przez PODZBIORY juz wygenerowanych danych --
zaden punkt krzywej nie wymaga dogenerowania czegokolwiek na GPU, a wszystkie
punkty pochodza z DOKLADNIE tych samych renderow. To wyklucza tlumaczenie
roznicy miedzy punktami krzywej szumem generatora.

Skladnia `--angle-subset`:
    all         -> 36 orientacji (0..350 co 10)
    cardinal    -> {0, 90, 180, 270}, baseline VisualEchoes (Gao i in. 2020)
    every_N     -> co N-ta orientacja, N in {2,3,4,6,9,12,18} -> 18/12/9/6/4/3/2
    random_K    -> K orientacji losowanych NA LOKALIZACJE, z ziarna
    random_K_of_G -> K orientacji losowanych NA LOKALIZACJE, ale WYLACZNIE z
                     podsiatki G rownomiernie rozlozonych orientacji

KRZYWA PRZY STALYM BUDZECIE PROBEK (`random_K_of_G`). Krzywa nasycenia
`every_N` idzie na naturalnej licznosci (6 960 -> 62 640 probek), wiec rosnie
po DWOCH zmiennych naraz: roznorodnosci katowej i rozmiarze zbioru. To, co na
niej widac, jest w duzej mierze nasyceniem po rozmiarze zbioru -- zjawiskiem
znanym i nieciekawym. `random_4_of_G` trzyma licznosc STALA (4 probki na
lokalizacje, czyli 5 496 treningowych w kazdym punkcie) i zmienia wylacznie to,
z jak gestej siatki te 4 katy pochodza. Koncami tej krzywej sa dokladnie
istniejace warunki: `random_4_of_4` == `cardinal` (warunek A, wybor 4 z 4 jest
deterministyczny), a `random_4_of_36` == `random_4` (warunek D).

Roznica `cardinal` vs `every_9` jest zerowa co do zbioru katow (oba daja
{0,90,180,270}) i to jest zamierzone: `every_9` istnieje po to, zeby krzywa
`every_N` miala wlasny, spojny punkt przy 4 katach, a `cardinal` po to, zeby
warunek A dal sie nazwac w pracy tym, czym jest -- odtworzeniem baseline'u.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# Siatka orientacji zapisana w atrybucie `angles_deg` kazdego pliku HDF5.
ANGLE_STEP_DEG = 10
N_ANGLES = 36
CARDINAL_DEG = (0, 90, 180, 270)

# Dzielniki 36 dopuszczone dla every_N. 12 i 18 daja 3 i 2 katy -- zostawione,
# bo sa poprawne, choc nie wchodza do glownej macierzy eksperymentow.
EVERY_N_ALLOWED = (2, 3, 4, 6, 9, 12, 18)

_EVERY_RE = re.compile(r"^every_(\d+)$")
_RANDOM_RE = re.compile(r"^random_(\d+)$")
_RANDOM_OF_RE = re.compile(r"^random_(\d+)_of_(\d+)$")

# Dopuszczalne rozmiary podsiatki G dla random_K_of_G: liczby orientacji, ktore
# rownomiernie dziela 36 (te same, ktore daje every_N, plus pelne 36).
GRID_SIZES_ALLOWED = (2, 3, 4, 6, 9, 12, 18, 36)


class AngleSubsetError(ValueError):
    pass


def parse(spec: str) -> tuple[str, int]:
    """'every_3' -> ('every', 3). Waliduje od razu, zeby literowka w nazwie
    warunku nie ujawnila sie dopiero po godzinie treningu."""
    spec = spec.strip()
    if spec == "all":
        return "all", N_ANGLES
    if spec == "cardinal":
        return "cardinal", len(CARDINAL_DEG)
    m = _EVERY_RE.match(spec)
    if m:
        n = int(m.group(1))
        if n not in EVERY_N_ALLOWED:
            raise AngleSubsetError(
                f"every_{n}: N musi dzielic 36 bez reszty, dozwolone {EVERY_N_ALLOWED}"
            )
        return "every", n
    m = _RANDOM_OF_RE.match(spec)
    if m:
        k, g = int(m.group(1)), int(m.group(2))
        if g not in GRID_SIZES_ALLOWED:
            raise AngleSubsetError(
                f"random_{k}_of_{g}: G musi rownomiernie dzielic {N_ANGLES}, "
                f"dozwolone {GRID_SIZES_ALLOWED}")
        if not 1 <= k <= g:
            raise AngleSubsetError(f"random_{k}_of_{g}: K musi byc w zakresie 1..{g}")
        return "random_of", k
    m = _RANDOM_RE.match(spec)
    if m:
        k = int(m.group(1))
        if not 1 <= k <= N_ANGLES:
            raise AngleSubsetError(f"random_{k}: K musi byc w zakresie 1..{N_ANGLES}")
        return "random", k
    raise AngleSubsetError(
        f"nieznany --angle-subset {spec!r}; oczekiwano: "
        f"all | cardinal | every_N | random_K | random_K_of_G"
    )


def grid_size(spec: str) -> int:
    """Z ilu orientacji POCHODZA katy tego podzbioru (os krzywej stalego budzetu)."""
    m = _RANDOM_OF_RE.match(spec.strip())
    if m:
        return int(m.group(2))
    if spec.strip() == "cardinal":
        return len(CARDINAL_DEG)
    return N_ANGLES if spec.strip() in ("all",) or spec.strip().startswith("random_") \
        else angles_per_location(spec)


def angles_per_location(spec: str) -> int:
    """Ile orientacji przypadnie na jedna lokalizacje. Potrzebne, zeby policzyc
    oczekiwana licznosc zbioru PRZED zbudowaniem indeksu (weryfikacja Bloku 2)."""
    kind, n = parse(spec)
    if kind == "all":
        return N_ANGLES
    if kind == "cardinal":
        return len(CARDINAL_DEG)
    if kind == "every":
        return N_ANGLES // n
    return n  # random_K, random_K_of_G


def select_angles(spec: str, *, scene: str, location_id: int, seed: int) -> np.ndarray:
    """Zwraca posortowane katy [stopnie] wybrane dla danej lokalizacji.

    Dla wariantow deterministycznych (all/cardinal/every_N) argumenty
    scene/location_id/seed sa ignorowane -- ten sam zbior katow dostaje kazda
    lokalizacja. Dla random_K losowanie jest zakotwiczone w (seed, scene,
    location_id), a NIE w globalnym stanie RNG: dzieki temu podzbior jest
    identyczny niezaleznie od tego, w jakiej kolejnosci budowany jest indeks,
    ilu jest workerow DataLoadera i czy wczesniej cos innego losowalo.
    """
    kind, n = parse(spec)
    grid = np.arange(N_ANGLES, dtype=np.int64) * ANGLE_STEP_DEG

    if kind == "all":
        return grid
    if kind == "cardinal":
        return np.asarray(CARDINAL_DEG, dtype=np.int64)
    if kind == "every":
        return grid[::n].copy()

    # random_K / random_K_of_G -- losowanie zakotwiczone w (seed, scene, location)
    if kind == "random_of":
        g = grid_size(spec)
        pool = grid[::N_ANGLES // g]
    else:
        pool = grid
    h = hashlib.sha256(f"{seed}|{scene}|{location_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    chosen = rng.choice(pool.size, size=n, replace=False)
    return np.sort(pool[chosen])


def describe(spec: str) -> str:
    kind, n = parse(spec)
    per_loc = angles_per_location(spec)
    if kind == "random_of":
        g = grid_size(spec)
        return (f"{spec}: {per_loc} katow/lokalizacje losowanych z podsiatki {g} orientacji "
                f"(co {N_ANGLES // g * ANGLE_STEP_DEG} stopni); licznosc zbioru NIEZALEZNA od G")
    if kind == "random":
        return f"{spec}: {per_loc} katow/lokalizacje, losowane per lokalizacja (zalezne od --angle-seed)"
    if kind == "every":
        return f"{spec}: {per_loc} katow/lokalizacje, co {n * ANGLE_STEP_DEG} stopni, staly zbior"
    return f"{spec}: {per_loc} katow/lokalizacje, staly zbior"
