#!/usr/bin/env python3
"""DOWOD: wysokosc agenta `y` pochodzi z graph.pkl, nie z pathfinder.snap_point().

Pytanie: specyfikacja podawala `snap_point()`, PKL_FORMAT.md — `graph.pkl`. Ktore
zrodlo odtwarza referencyjny zbior `scene_observations_128.pkl`?

Metoda: renderuje lokalizacje i katy, ktore JUZ ISTNIEJA w pkl, obiema wersjami
`y` i porownuje piksel-po-pikselu. Rendering wizualny w habitat-sim jest
deterministyczny (inaczej niz audio), wiec kazda niezerowa roznica oznacza
rozbieznosc konfiguracji — test ma moc rozstrzygajaca.

Wynik (2026-07-28, office_1, 16 lokalizacji x 4 katy = 64 porownania):
    graph.pkl    RGB RMSE 0.0125 / max 0.0214, 99.982 % pikseli bit-identycznych
    snap_point   RGB RMSE 50.05  / max 75.28,  36.02 %

Raport: RAPORT_SESJI_2026-07-26_29.md §2.1 | Dokument: GENERATOR_PARAMS.md §2

Uruchomienie:
    conda activate habitat
    python my-operations/measurements/agent_height_vs_pkl.py [--scene office_1]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion  # noqa: F401
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from echo_core.params import CAMERA_HFOV, CAMERA_RESOLUTION, SENSOR_HEIGHT
from echo_core.paths import LOCATIONS_PKL, graph_pkl, points_txt, scene_mesh


def build_camera_only_sim(scene):
    """Simulator z samymi kamerami — pomiar dotyczy wylacznie obrazu."""
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene_mesh(scene))
    cfg.create_renderer = True
    cfg.enable_physics = False
    cfg.gpu_device_id = 0
    specs = []
    for uuid, stype in (("rgb", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type = uuid, stype
        s.resolution = list(CAMERA_RESOLUTION)
        s.position = [0.0, SENSOR_HEIGHT, 0.0]
        s.hfov = CAMERA_HFOV
        specs.append(s)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="office_1")
    args = ap.parse_args()

    ref = pickle.load(open(LOCATIONS_PKL, "rb"))[args.scene]
    graph = pickle.load(open(graph_pkl(args.scene), "rb"))
    ys = {int(n): d["point"][1] for n, d in graph.nodes(data=True)}
    pts = pd.read_csv(points_txt(args.scene), sep="\t", header=None,
                      names=["id", "a", "b", "c"])

    sim = build_camera_only_sim(args.scene)
    agent = sim.get_agent(0)

    def render(pos, ang):
        st = agent.get_state()
        st.position = np.array(pos, dtype=np.float32)
        st.rotation = quat_from_angle_axis(np.deg2rad(float(ang)), np.array([0., 1., 0.]))
        st.sensor_states = {}          # wyczyszczenie, zeby offset sensora policzyl sie od nowa
        agent.set_state(st, True)
        return sim.get_sensor_observations()

    locs = sorted({k[0] for k in ref.keys()})
    res = {v: {"rgb": [], "depth": [], "bit": []} for v in ("graph", "snap")}
    try:
        for lid in locs:
            r = pts[pts["id"] == lid].iloc[0]
            x, z = float(r["a"]), -float(r["b"])
            cand = {"graph": [x, ys[lid], z],
                    "snap": list(sim.pathfinder.snap_point([x, ys[lid], z]))}
            for ang in (0, 90, 180, 270):
                gt = ref[(lid, ang)]
                for name, pos in cand.items():
                    o = render(pos, ang)
                    d = res[name]
                    d["rgb"].append(float(np.sqrt(np.mean(
                        (o["rgb"].astype(np.float64) - gt["rgb"].astype(np.float64)) ** 2))))
                    d["depth"].append(float(np.sqrt(np.mean(
                        (o["depth"].astype(np.float64) - gt["depth"].astype(np.float64)) ** 2))))
                    d["bit"].append(float(np.mean(o["rgb"] == gt["rgb"])))
    finally:
        sim.close()

    print(f"\n=== {args.scene}: {len(locs)} lokalizacji x 4 katy = {len(locs)*4} porownan z pkl ===")
    print(f"{'wariant y':<12}{'RGB RMSE sr.':>14}{'RGB RMSE max':>14}"
          f"{'% bit-identycznych':>20}{'depth RMSE sr.':>16}")
    for name in ("graph", "snap"):
        d = res[name]
        print(f"{name:<12}{np.mean(d['rgb']):>14.4f}{np.max(d['rgb']):>14.4f}"
              f"{100*np.mean(d['bit']):>19.3f}%{np.mean(d['depth']):>16.6f}")


if __name__ == "__main__":
    main()
