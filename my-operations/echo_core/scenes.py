"""Sceny Replica: kolejnosc generowania i zrodlo pozycji lokalizacji."""

import numpy as np

from .paths import LOCATIONS_PKL, graph_pkl, points_txt

# Kolejnosc scen — GENERATOR_PARAMS.md §4.2. Nie alfabetyczna: najpierw scena
# walidacyjna, potem komplet held-out (zeby dataloader mogl powstawac rownolegle
# do generacji), potem po jednej treningowej z kazdej rodziny, potem reszta
# rosnaco po liczbie lokalizacji.
SCENE_ORDER = (
    "office_1",                                        # walidacyjna generatora
    "apartment_2", "frl_apartment_5", "office_4",      # held-out
    "room_0", "office_0", "hotel_0",                   # po jednej z kazdej rodziny
    "room_1", "office_2", "room_2", "office_3",
    "frl_apartment_2", "frl_apartment_4", "frl_apartment_0", "frl_apartment_1",
    "frl_apartment_3", "apartment_1", "apartment_0",
)
HELD_OUT = ("apartment_2", "frl_apartment_5", "office_4")


# Lokalizacje: zbior z pkl, wspolrzedne z points.txt, wysokosc z graph.pkl
# ---------------------------------------------------------------------------
def load_scene_locations(scene):
    """-> (loc_ids: list[int], positions: dict[int, np.ndarray(3, float32)])

    Zrodla, zgodnie z GENERATOR_PARAMS.md §2 i PKL_FORMAT.md:
      * ZBIOR lokalizacji — klucze `scene_observations_128.pkl`, nie caly
        `points.txt`. Tylko ten zbior odpowiada probkowaniu z pracy Gao
        (VisualEchoes, ECCV 2020), wiec tylko on daje porownywalnosc.
      * x, z — z `points.txt`: x = a, z = -b.
      * y — z `graph.pkl` (`node["point"][1]`, pelna precyzja float32), a dla
        8 lokalizacji z calego zbioru, ktorych w grafie nie ma (room_0: 102,
        103, 111, 112, 120, 121; room_1: 45, 51) — stala sceny, bo `y` jest
        stale w obrebie sceny.

    DLACZEGO NIE `pathfinder.snap_point()`: zwraca on wysokosc powierzchni
    navmesha, ktora lezy ~0.21 m NAD `y` z grafu (mediana po 1740 lokalizacjach,
    maksimum 0.49 m; navmesh Repliki nie ma zapisanych NavMeshSettings, wiec
    recast odtwarza go z domyslna kwantyzacja). Zmierzone na office_1 przez
    porownanie piksel-po-pikselu z pkl: `y` z grafu daje RGB RMSE 0.0125 i
    99.98 % pikseli bit-identycznych, `snap_point` — RGB RMSE 50.05 i 36 %.
    Patrz GENERATOR_PARAMS.md §2 (poprawka 2026-07-28) i §5 ograniczenie 8.
    """
    import pandas as pd
    import pickle

    with open(LOCATIONS_PKL, "rb") as f:
        observations = pickle.load(f)
    if scene not in observations:
        raise KeyError(f"scena {scene!r} nie wystepuje w {LOCATIONS_PKL.name}")
    loc_ids = sorted({int(k[0]) for k in observations[scene].keys()})
    del observations  # 913 MB — nie trzymamy tego przez cala generacje

    with open(graph_pkl(scene), "rb") as f:
        graph = pickle.load(f)
    node_y = {int(n): float(d["point"][1]) for n, d in graph.nodes(data=True)}
    if not node_y:
        raise RuntimeError(f"graph.pkl sceny {scene} nie ma zadnego wezla z 'point'")
    y_scene = float(np.median(list(node_y.values())))

    points = pd.read_csv(points_txt(scene), sep="\t", header=None, names=["id", "a", "b", "c"])
    by_id = {int(r.id): (float(r.a), float(r.b)) for r in points.itertuples()}

    positions = {}
    for lid in loc_ids:
        if lid not in by_id:
            raise KeyError(f"{scene}: location_id={lid} z pkl nie ma odpowiednika w points.txt")
        a, b = by_id[lid]
        positions[lid] = np.array([a, node_y.get(lid, y_scene), -b], dtype=np.float32)
    return loc_ids, positions

