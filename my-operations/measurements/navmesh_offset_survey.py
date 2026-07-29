#!/usr/bin/env python3
"""DOWOD: navmesh Repliki lezy ~0.21 m nad `y` z graph.pkl — na wszystkich 1740 lokalizacjach.

Pytanie: skala rozbieznosci miedzy `pathfinder.snap_point()` a `y` z grafu — czy to
pojedyncze punkty, czy caly zbior?

Metoda: bez GPU. `habitat_sim.nav.PathFinder` laduje sam navmesh, wiec da sie
przemierzyc wszystkie sceny bez konstruowania Simulatora. Sprawdza takze, czy
wynik zalezy od punktu startowego `y_guess` (nie zalezy — snap_point rzutuje na
navmesh niezaleznie od startu).

Wynik (2026-07-29): mediana |dy| = 0.2125 m, maksimum 0.4901 m, |dy| > 1 cm
w 1738 z 1740 lokalizacji. Przyczyna: navmesh nie ma zapisanych NavMeshSettings,
wiec recast odtwarza go z domyslna kwantyzacja.

Raport: RAPORT_SESJI_2026-07-26_29.md §2.1 | Dokument: GENERATOR_PARAMS.md §2

Uruchomienie (bez GPU):
    python my-operations/measurements/navmesh_offset_survey.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion  # noqa: F401
import habitat_sim

from echo_core.paths import LOCATIONS_PKL, SCENE_ROOT, graph_pkl, points_txt
from echo_core.scenes import SCENE_ORDER


def main():
    obs = pickle.load(open(LOCATIONS_PKL, "rb"))
    print(f"  {'scena':<18}{'n':>5}{'max|dy|':>10}{'med|dy|':>10}{'dy>1cm':>8}{'max dxz':>10}")
    tot = big = 0
    all_dy = []
    for scene in SCENE_ORDER:
        graph = pickle.load(open(graph_pkl(scene), "rb"))
        ys = {int(n): d["point"][1] for n, d in graph.nodes(data=True)}
        y_scene = float(np.median(list(ys.values())))
        pts = pd.read_csv(points_txt(scene), sep="\t", header=None,
                          names=["id", "a", "b", "c"])
        locs = sorted({k[0] for k in obs[scene].keys()})

        pf = habitat_sim.nav.PathFinder()
        pf.load_nav_mesh(str(SCENE_ROOT / scene / "habitat/mesh_semantic.navmesh"))
        dy, dxz = [], []
        for lid in locs:
            r = pts[pts["id"] == lid].iloc[0]
            x, z = float(r["a"]), -float(r["b"])
            y = ys.get(lid, y_scene)          # 8 lokalizacji spoza grafu -> stala sceny
            sp = pf.snap_point([x, y, z])
            dy.append(abs(float(sp[1]) - y))
            dxz.append(float(np.hypot(sp[0] - x, sp[2] - z)))
        dy, dxz = np.array(dy), np.array(dxz)
        all_dy.append(dy)
        tot += len(locs)
        big += int((dy > 0.01).sum())
        print(f"  {scene:<18}{len(locs):>5}{dy.max():>10.4f}{np.median(dy):>10.4f}"
              f"{int((dy>0.01).sum()):>8}{dxz.max():>10.4f}")

    a = np.concatenate(all_dy)
    print(f"\n  RAZEM {tot} lokalizacji")
    print(f"  |dy|: mediana {np.median(a):.4f} m, maksimum {a.max():.4f} m")
    print(f"  |dy| > 1 cm w {big} lokalizacjach ({100*big/tot:.1f} %)")


if __name__ == "__main__":
    main()
