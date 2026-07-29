#!/usr/bin/env python3
"""Rzeczywista powierzchnia [m2] per kategoria semantyczna Repliki.

DLACZEGO to liczymy, a nie bierzemy liczby obiektow: material akustyczny jest
przypisywany PER TROJKAT siatki semantycznej (AudioSensor::loadSemanticMesh()),
a wplyw materialu na pogłos jest proporcjonalny do POWIERZCHNI, nie do liczby
obiektow. 255 ksiazek to akustycznie nic, jedna sciana to wszystko. Liczba
obiektow (ani nawet pole bounding-boxa) nie jest wiec wlasciwa waga przy
decydowaniu, ktore kategorie warto mapowac starannie.

Format pliku: Replica `mesh_semantic.ply` (binary little-endian) ma
`element face` z lista uint32 o STALEJ dlugosci 4 (same quady) i wlasciwoscia
`object_id` (uint16) -> rekord ma zawsze 19 bajtow, wiec da sie go wczytac
jednym `np.frombuffer` bez parsowania sekwencyjnego.

Uruchomienie:  python my-operations/replica_semantic_area.py
"""

import collections
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPLICA_ROOT = REPO_ROOT / "sound-spaces/data/scene_datasets/replica"

FACE_DTYPE = np.dtype([("n", "u1"), ("v", "<u4", 4), ("object_id", "<u2")])
VERT_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])


def _header_counts(buf):
    head = buf[: buf.find(b"end_header\n") + len(b"end_header\n")].decode("ascii", "replace")
    counts = {}
    for line in head.splitlines():
        if line.startswith("element "):
            _, name, num = line.split()
            counts[name] = int(num)
    return len(head), counts


def scene_area_by_object(scene_dir):
    """object_id -> powierzchnia [m2] (quady dzielone na dwa trojkaty)."""
    buf = (scene_dir / "habitat/mesh_semantic.ply").read_bytes()
    off, counts = _header_counts(buf)
    nv, nf = counts["vertex"], counts["face"]
    verts = np.frombuffer(buf, dtype=VERT_DTYPE, count=nv, offset=off)
    faces = np.frombuffer(buf, dtype=FACE_DTYPE, count=nf, offset=off + nv * VERT_DTYPE.itemsize)
    if not np.all(faces["n"] == 4):
        raise RuntimeError(f"{scene_dir.name}: nie same quady, parser wymaga rozszerzenia")

    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    v = faces["v"]
    p0, p1, p2, p3 = xyz[v[:, 0]], xyz[v[:, 1]], xyz[v[:, 2]], xyz[v[:, 3]]
    area = 0.5 * (np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
                  + np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=1))

    out = collections.Counter()
    oid = faces["object_id"].astype(np.int64)
    sums = np.bincount(oid, weights=area)
    for i, a in enumerate(sums):
        if a > 0:
            out[i] = float(a)
    return out


def inventory():
    scenes = sorted(p for p in REPLICA_ROOT.iterdir() if (p / "habitat/info_semantic.json").exists())
    area = collections.Counter()
    objects = collections.Counter()
    scenes_with = collections.defaultdict(set)
    neg_area = 0.0
    total_area = 0.0
    per_scene = {}

    for sd in scenes:
        info = json.loads((sd / "habitat/info_semantic.json").read_text())
        # id -> nazwa kategorii; class_id == -1 nie ma kategorii (category() == null
        # w habitat-sim), wiec ZAWSZE dostaje material domyslny, niezaleznie od configu
        by_id = {}
        for o in info["objects"]:
            by_id[o["id"]] = "<class_id=-1>" if o["class_id"] == -1 else o["class_name"]
            objects["<class_id=-1>" if o["class_id"] == -1 else o["class_name"]] += 1
        a_obj = scene_area_by_object(sd)
        s_area = collections.Counter()
        for oid, a in a_obj.items():
            name = by_id.get(oid, "<brak w info_semantic>")
            s_area[name] += a
        for n, a in s_area.items():
            area[n] += a
            scenes_with[n].add(sd.name)
        total_area += sum(s_area.values())
        neg_area += s_area.get("<class_id=-1>", 0.0)
        per_scene[sd.name] = dict(s_area)
        print(f"  {sd.name:<18} {sum(s_area.values()):9.1f} m2, {len(s_area):3d} kategorii, "
              f"class_id=-1: {s_area.get('<class_id=-1>', 0.0) / sum(s_area.values()) * 100:5.2f}% pola",
              file=sys.stderr)

    return {"area": dict(area), "objects": dict(objects),
            "scenes_with": {k: sorted(v) for k, v in scenes_with.items()},
            "total_area": total_area, "neg_area": neg_area, "per_scene": per_scene,
            "n_scenes": len(scenes)}


def main():
    inv = inventory()
    area, total = inv["area"], inv["total_area"]
    print(f"\nScen: {inv['n_scenes']}, kategorii: {len(area)}, laczna powierzchnia: {total:.0f} m2")
    print(f"class_id=-1 (zawsze material domyslny): {inv['neg_area']:.1f} m2 = "
          f"{inv['neg_area'] / total * 100:.2f}% powierzchni, {inv['objects'].get('<class_id=-1>', 0)} obiektow")
    print(f"\n{'kategoria':<24}{'scen':>5}{'obiektow':>9}{'pole [m2]':>12}{'% pola':>9}{'skum.':>8}")
    cum = 0.0
    for name, a in sorted(area.items(), key=lambda kv: -kv[1]):
        cum += a
        print(f"{name:<24}{len(inv['scenes_with'][name]):>5}{inv['objects'].get(name, 0):>9}"
              f"{a:>12.1f}{a / total * 100:>8.2f}%{cum / total * 100:>7.1f}%")

    out = REPO_ROOT / "my-operations/replica_category_area.json"
    out.write_text(json.dumps({k: v for k, v in inv.items() if k != "per_scene"}, indent=2))
    print(f"\nZapisano: {out}")


if __name__ == "__main__":
    main()
