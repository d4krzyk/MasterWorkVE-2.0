#!/usr/bin/env python3
"""Dobor materialu laty pod zgodnosc z SoundSpaces 1.0 — sweep po materialach.

UWAGA METODOLOGICZNA, ktora musi isc razem z kazda liczba z tego skryptu:
to jest DOPASOWANIE, nie walidacja. Mamy RIR-y SoundSpaces 1.0 dla dwoch scen:
`office_1` (naturalnie zamknieta, laty jej nie dotycza) i `frl_apartment_2`. Sweep
dobiera material tak, zeby trafic w RT60 tej JEDNEJ sceny, wiec zgodnosc na koniec
jest tautologiczna — nie jest dowodem, ze symulacja zgadza sie z SS 1.0. Zeby to byla
walidacja, trzeba dociagnac RIR-y drugiej sceny otwartej i sprawdzic na niej material
dobrany tutaj. Dopoki to nie nastapi, wynik wolno opisac jako "material dobrany tak,
by trafic w SS 1.0", a nie "nasza symulacja zgadza sie z SS 1.0".

JAK DZIALA PODMIANA MATERIALU: lata dostaje `object_id` istniejacego obiektu wybranej
klasy, a material wynika z klasy przez replica_material_config.json. Sciany TAMTEGO
obiektu pozostaja niezmienione, wiec zmienia sie material dokladnie tych trojkatow,
ktore doklejamy — sweep nie ma zadnych efektow ubocznych na reszte sceny.

Punkt odniesienia: patch semantyczny (klasa `wall` = Gypsum Board, alfa 0.04 @ 1 kHz)
jest wyborem ZASADNYM, nie dopasowanym — wynika z tego, co lezy przy brzegu dziury.
Kazdy inny wiersz tabeli to juz dopasowanie.

Uruchomienie (1 + N konstrukcji Simulatora w jednym procesie; limit ~30 na proces):
    python my-operations/measurements/patch_material_sweep.py --scene frl_apartment_2
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from ceiling_patch_rt60 import (BANDS, OUT_DIR, PATCHED_ROOT, SS1_RT60, build,
                                measure, sample_pairs)
from echo_core.paths import REPO_ROOT, scene_mesh
from echo_core.scenes import load_scene_locations

# Drabinka klas rozpieta na calym dostepnym zakresie pochlaniania. W nawiasie material
# i alfa @ 1 kHz z replica_material_config.json. `wall` to wybor semantyczny (domyslny).
LADDER = [
    ("pillar",       "Concrete",       0.02),
    ("wall",         "Gypsum Board",   0.04),
    ("door",         "wood, Thick",    0.06),
    ("floor",        "Wood Floor",     0.07),
    ("handrail",     "Steel",          0.10),
    ("bottle",       "Glass",          0.12),
    ("indoor-plant", "Foliage",        0.17),
    ("mat",          "Carpet",         0.20),
    ("rug",          "Carpet, Heavy",  0.37),
    ("curtain",      "Curtain",        0.75),
]


def run_variant(scene, cls, pairs, positions, n_renders, warmup, keep):
    suffix = f"__{cls}"
    mesh = PATCHED_ROOT / (scene + suffix) / "habitat/mesh_semantic.ply"
    r = subprocess.run(
        [sys.executable, str(HERE / "patch_scene_holes.py"), "--scene", scene,
         "--patch-class", cls, "--suffix", suffix, "--force"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode != 0 or not mesh.exists():
        print(f"    [{cls}] latanie NIEUDANE:\n      " +
              "\n      ".join(r.stderr.strip().splitlines()[-4:]))
        return None
    sim = build(mesh)
    try:
        res, n_rir = measure(sim, pairs, positions, n_renders, warmup, cls)
    finally:
        sim.close()
    if not keep:
        shutil.rmtree(mesh.parent.parent, ignore_errors=True)   # ~84 MB na wariant
    return {fc: (float(np.median(res[fc])) if res[fc] else np.nan) for fc in BANDS}, n_rir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="frl_apartment_2")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--renders", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--dmin", type=float, default=1.0)
    ap.add_argument("--dmax", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--keep-meshes", action="store_true")
    ap.add_argument("--classes", help="ogranicz drabinke, np. 'wall,rug' (do szybkiego testu)")
    args = ap.parse_args()

    ladder = LADDER
    if args.classes:
        want = {c.strip() for c in args.classes.split(",")}
        ladder = [e for e in LADDER if e[0] in want]

    if args.scene not in SS1_RT60:
        sys.exit(f"brak wartosci SS 1.0 dla {args.scene}")
    ss1 = SS1_RT60[args.scene]
    mid_bands = [b for b in BANDS if int(b) in (500, 1000, 2000)]

    loc_ids, positions = load_scene_locations(args.scene)
    pairs, dst = sample_pairs(positions, loc_ids, args.pairs, args.dmin, args.dmax, args.seed)
    print(f"\n  {args.scene}: {len(pairs)} par, odleglosc srednia {dst[0]:.2f} m "
          f"({dst[1]:.2f}-{dst[2]:.2f}), {args.renders} renderow/pare, rozgrzewka {args.warmup}\n")

    rows = []
    print("  --- siatka oryginalna (bez laty)")
    sim = build(scene_mesh(args.scene))
    try:
        res, n_rir = measure(sim, pairs, positions, args.renders, args.warmup, "oryginalna")
    finally:
        sim.close()
    med = {fc: (float(np.median(res[fc])) if res[fc] else np.nan) for fc in BANDS}
    rows.append({"wariant": "BEZ LATY", "material": "-", "alfa_1k": None,
                 "rt60": med, "rir_s": round(n_rir / 44100, 2)})

    for cls, matname, alpha in ladder:
        print(f"  --- lata jako '{cls}' ({matname}, alfa@1k = {alpha})")
        out = run_variant(args.scene, cls, pairs, positions, args.renders,
                          args.warmup, args.keep_meshes)
        if out is None:
            continue
        med, n_rir = out
        rows.append({"wariant": cls, "material": matname, "alfa_1k": alpha,
                     "rt60": med, "rir_s": round(n_rir / 44100, 2)})

    # --- tabela ---------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"  SWEEP MATERIALU LATY — {args.scene}, RT60 [s] wobec SoundSpaces 1.0")
    print("=" * 88)
    print(f"  {'wariant':<14}{'material':<16}{'a@1k':>6}{'RIR':>6}"
          + "".join(f"{int(fc):>7}" for fc in BANDS) + f"{'/SS1':>8}")
    best, best_err = None, None
    for r in rows:
        ratios = [r["rt60"][fc] / ss1[int(fc)] for fc in mid_bands
                  if np.isfinite(r["rt60"][fc])]
        ratio = float(np.median(ratios)) if ratios else np.nan
        r["ratio_ss1"] = round(ratio, 4) if np.isfinite(ratio) else None
        a = f"{r['alfa_1k']:.2f}" if r["alfa_1k"] is not None else "-"
        print(f"  {r['wariant']:<14}{r['material']:<16}{a:>6}{r['rir_s']:>6.2f}"
              + "".join(f"{r['rt60'][fc]:>7.3f}" for fc in BANDS)
              + f"{ratio:>7.2f}x")
        if r["wariant"] != "BEZ LATY" and np.isfinite(ratio):
            err = abs(np.log(ratio))          # 0.5x i 2.0x to ten sam blad co do wielkosci
            if best_err is None or err < best_err:
                best, best_err = r, err
    print(f"\n  {'SoundSpaces 1.0':<30}      " + "".join(f"{ss1[int(fc)]:>7.3f}" for fc in BANDS))
    if best:
        print(f"\n  NAJBLIZEJ SS 1.0: lata jako '{best['wariant']}' ({best['material']}, "
              f"alfa@1k = {best['alfa_1k']}) -> {best['ratio_ss1']:.2f}x")
        sem = next((r for r in rows if r["wariant"] == "wall"), None)
        if sem:
            print(f"  Wybor SEMANTYCZNY (wall / Gypsum Board)              -> {sem['ratio_ss1']:.2f}x")
        print("\n  Przypomnienie: powyzsze to DOPASOWANIE do jednej sceny, nie walidacja —"
              "\n  patrz naglowek skryptu.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.scene}_material_sweep.json"
    out.write_text(json.dumps({
        "scene": args.scene, "pairs": len(pairs), "renders_per_pair": args.renders,
        "warmup": args.warmup, "dist_mean_m": round(dst[0], 3),
        "ss1_rt60": ss1, "is_fit_not_validation": True,
        "variants": [{**r, "rt60": {str(int(k)): round(v, 4) for k, v in r["rt60"].items()}}
                     for r in rows],
    }, indent=2, ensure_ascii=False))
    print(f"\n  zapisano: {out}")


if __name__ == "__main__":
    main()
