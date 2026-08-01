#!/usr/bin/env python3
"""Zgodnosc miedzysilnikowa SS 2.0 vs SS 1.0: sceny SZCZELNE kontra OTWARTE.

PO CO, skoro §2.13 juz porownywalo RT60 z SS 1.0: tamten wynik (0.96x) powstal przez
DOBRANIE materialu laty pod SS 1.0 na jednej scenie, wiec byl tautologiczny. Ten pomiar
jest odwrotny — nie ma w nim ANI JEDNEGO dopasowanego parametru:

  * na scenach SZCZELNYCH geometria jest cala, wiec nie ma czego latac ani dobierac.
    Zgodnosc (albo jej brak) z SS 1.0 jest tu czysta wlasnoscia dwoch silnikow.
  * na scenie OTWARTEJ porownujemy geometrie oryginalna z zalatana SEMANTYCZNIE
    (material dziedziczony z otoczenia dziury, sufit na wysokosci ocalalego fragmentu) —
    zadnego strojenia.

Teza, ktora to ma rozstrzygnac: jesli na scenach szczelnych oba silniki zgadzaja sie
w granicach kilkunastu procent, to odchylenie na `frl_apartment_*` jest wlasnoscia SIATKI,
a nie silnika — i to zdanie nie wymaga zadnej lataniny ani dopasowania.

UWAGA o statusie SS 1.0: to baseline, nie prawda podstawowa. Paper SoundSpaces 2.0
(arXiv 2206.08312) raportuje, ze 2.0 zgadza sie z pomiarami rzeczywistymi LEPIEJ niz 1.0
(blad stosunku direct-to-reverberant 11.0 dB -> 0.98 dB). Rozbieznosc nie oznacza wiec
automatycznie bledu po naszej stronie.

METODA — identyczna po obu stronach:
  * pary zrodlo-odbiornik filtrowane po odleglosci 1-3 m (bez tego porownanie miedzy
    scenami jest obciazone: w malych scenach losowe pary sa blizsze niz w duzych);
  * zrodlo ODDALONE, nie wspollokowane — echolokacja daje zanik wielonachyleniowy,
    ktory przesuwa okno dopasowania T20 w inna czesc krzywej niz w RIR-ach SS 1.0;
  * ten sam estymator: Schroeder -> T20 w oknie -5..-25 dB -> RT60, pasma oktawowe;
  * usrednianie po renderach w domenie energii, filtr w domenie cisnienia osobno dla
    kazdego renderu.

Podzial scen na szczelne/otwarte pochodzi z POMIARU ucieczki promieni
(outputs/measurements/ray_escape/summary.csv), nie z heurystyki.

Uruchomienie — jedna scena na proces (limit ~30 konstrukcji Simulatora na proces):
    python my-operations/measurements/cross_engine_rt60.py --all
    python my-operations/measurements/cross_engine_rt60.py --summary    # bez GPU
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from ceiling_patch_rt60 import PATCHED_ROOT, build, measure, sample_pairs
from echo_core.paths import REPO_ROOT, scene_mesh
from echo_core.scenes import load_scene_locations
from soundspaces1_rt60 import BANDS, RIR_ROOT, scene_rt60

OUT_DIR = REPO_ROOT / "outputs/measurements/cross_engine"
ESCAPE_SUMMARY = REPO_ROOT / "outputs/measurements/ray_escape/summary.csv"
MID = (500, 1000, 2000)


def scene_class(scene):
    """-> 'szczelna' / 'nieszczelna bokiem' / 'BEZ SUFITU' z pomiaru ucieczki promieni."""
    if not ESCAPE_SUMMARY.exists():
        sys.exit(f"brak {ESCAPE_SUMMARY} — najpierw ray_escape_survey.py --all")
    for r in csv.DictReader(open(ESCAPE_SUMMARY)):
        if r["scene"] == scene:
            return r["class"]
    return "?"


def available_scenes():
    """Sceny, dla ktorych mamy pobrane RIR-y SoundSpaces 1.0."""
    if not RIR_ROOT.is_dir():
        sys.exit(f"brak {RIR_ROOT} — najpierw fetch_soundspaces1_rirs.sh")
    return sorted(d.name for d in RIR_ROOT.iterdir()
                  if d.is_dir() and any(d.glob("*/*.wav")))


def measure_ours(scene, mesh, pairs, positions, n_renders, warmup, label):
    sim = build(mesh)
    try:
        res, n_rir = measure(sim, pairs, positions, n_renders, warmup, label)
    finally:
        sim.close()
    return ({fc: (float(np.median(res[fc])) if res[fc] else float("nan")) for fc in BANDS},
            n_rir)


def run_one(scene, variant, args):
    loc_ids, positions = load_scene_locations(scene)
    pairs, dst = sample_pairs(positions, loc_ids, args.pairs, args.dmin, args.dmax, args.seed)
    mesh = (PATCHED_ROOT / scene / "habitat/mesh_semantic.ply" if variant == "patched"
            else scene_mesh(scene))
    if not Path(mesh).exists():
        sys.exit(f"brak siatki {mesh}")
    med, n_rir = measure_ours(scene, mesh, pairs, positions, args.renders, args.warmup,
                              f"{scene}/{variant}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{scene}__{variant}.json"
    out.write_text(json.dumps({
        "scene": scene, "variant": variant, "class": scene_class(scene),
        "pairs": len(pairs), "renders_per_pair": args.renders, "warmup": args.warmup,
        "dist_mean_m": round(dst[0], 3), "rir_s": round(n_rir / 44100, 3),
        "rt60": {str(int(fc)): round(med[fc], 4) for fc in BANDS},
    }, indent=2))
    print(f"  {scene}/{variant}: RT60@1k = {med[1000.0]:.3f} s, RIR {n_rir/44100:.2f} s "
          f"-> {out.name}")


def summary(args):
    files = sorted(OUT_DIR.glob("*.json")) if OUT_DIR.is_dir() else []
    if not files:
        sys.exit(f"brak wynikow w {OUT_DIR} — najpierw --all")
    ours = {}
    for f in files:
        d = json.loads(f.read_text())
        ours[(d["scene"], d["variant"])] = d

    # soundspaces1_rt60.scene_rt60 losuje pary przez rng.sample(), czyli oczekuje
    # random.Random, a nie generatora numpy (ten ma .choice, nie .sample).
    import random
    rng = random.Random(args.seed)
    scenes = sorted({s for s, _ in ours})
    print("\n" + "=" * 104)
    print("  ZGODNOSC MIEDZYSILNIKOWA — SoundSpaces 2.0 (nasze) vs 1.0, BEZ dopasowanych parametrow")
    print("=" * 104)
    print(f"  {'scena':<17}{'typ':<20}{'wariant':<10}"
          + "".join(f"{b:>7}" for b in (250, 500, 1000, 2000)) + f"{'/SS1':>8}{'RIR':>7}")
    print("  " + "-" * 102)

    rows = []
    for s in scenes:
        try:
            ss1, n_ok, _fs, _dur, _nf, _dst = scene_rt60(s, args.ss1_pairs, rng,
                                                         args.dmin, args.dmax)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  {s:<17}SS 1.0 niedostepne: {e}")
            continue
        cls = scene_class(s)
        print(f"  {s:<17}{cls:<20}{'SS 1.0':<10}"
              + "".join(f"{ss1[float(b)]:>7.3f}" for b in (250, 500, 1000, 2000))
              + f"{'—':>8}{'—':>7}")
        for variant in ("original", "patched"):
            d = ours.get((s, variant))
            if not d:
                continue
            r = [d["rt60"][str(b)] / ss1[float(b)] for b in MID
                 if np.isfinite(ss1[float(b)])]
            ratio = float(np.median(r)) if r else float("nan")
            print(f"  {'':<17}{'':<20}{variant:<10}"
                  + "".join(f"{d['rt60'][str(b)]:>7.3f}" for b in (250, 500, 1000, 2000))
                  + f"{ratio:>7.2f}x{d['rir_s']:>7.2f}")
            rows.append({"scene": s, "class": cls, "variant": variant,
                         "ratio_ss1": round(ratio, 4),
                         "rt60": d["rt60"], "ss1": {str(b): round(ss1[float(b)], 4)
                                                    for b in (250, 500, 1000, 2000)}})
        print("  " + "-" * 102)

    def agg(pred, name):
        v = [r["ratio_ss1"] for r in rows if pred(r) and np.isfinite(r["ratio_ss1"])]
        if not v:
            return None
        med = float(np.median(v))
        print(f"  {name:<52} n={len(v):<3} mediana {med:.2f}x  "
              f"zakres {min(v):.2f}-{max(v):.2f}x")
        return {"name": name, "n": len(v), "median": round(med, 4),
                "min": round(min(v), 4), "max": round(max(v), 4)}

    print("\n  AGREGATY (mediana stosunku nasze/SS 1.0 po pasmach 500 Hz - 2 kHz)")
    a = [agg(lambda r: r["class"] == "szczelna" and r["variant"] == "original",
             "sceny SZCZELNE, geometria oryginalna (0 wolnych parametrow)"),
         agg(lambda r: r["class"] != "szczelna" and r["variant"] == "original",
             "sceny nieszczelne, geometria oryginalna"),
         agg(lambda r: r["class"] != "szczelna" and r["variant"] == "patched",
             "sceny nieszczelne, ZALATANE semantycznie (0 wolnych parametrow)")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps({"per_scene": rows,
                               "aggregates": [x for x in a if x],
                               "ss1_pairs": args.ss1_pairs,
                               "is_fit": False}, indent=2, ensure_ascii=False))
    print(f"\n  zapisano: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene")
    g.add_argument("--all", action="store_true")
    g.add_argument("--summary", action="store_true")
    ap.add_argument("--variant", default="original", choices=("original", "patched"))
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--ss1-pairs", type=int, default=80)
    ap.add_argument("--renders", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--dmin", type=float, default=1.0)
    ap.add_argument("--dmax", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260729)
    args = ap.parse_args()

    if args.summary:
        return summary(args)
    if args.scene:
        return run_one(args.scene, args.variant, args)

    todo = []
    for s in available_scenes():
        todo.append((s, "original"))
        if (PATCHED_ROOT / s / "habitat/mesh_semantic.ply").exists():
            todo.append((s, "patched"))
    print(f"  {len(todo)} pomiarow, kazdy w osobnym procesie "
          f"(sceny z RIR-ami SS 1.0: {', '.join(available_scenes())})\n")
    failed = []
    for s, v in todo:
        r = subprocess.run([sys.executable, __file__, "--scene", s, "--variant", v,
                            "--pairs", str(args.pairs), "--renders", str(args.renders),
                            "--warmup", str(args.warmup)],
                           cwd=str(REPO_ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            failed.append(f"{s}/{v}")
            print(f"  {s}/{v}: BLAD\n    " +
                  "\n    ".join(r.stderr.strip().splitlines()[-5:]))
        else:
            print(r.stdout.rstrip())
    if failed:
        # nie polykamy bledu: brak sceny w agregacie musi byc widoczny
        print(f"\n  NIEPOWODZENIE: {', '.join(failed)}")
    summary(args)


if __name__ == "__main__":
    main()
