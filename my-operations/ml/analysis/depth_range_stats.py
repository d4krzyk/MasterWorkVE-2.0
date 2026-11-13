#!/usr/bin/env python
"""Jaka czesc glebi lezy POZA zasiegiem okna echa? Odsetek waznych pikseli z glebia > 10,29 m.

    python my-operations/ml/analysis/depth_range_stats.py [--out DIR] [--dry-run]

Zero GPU. Czyta glebie (metry, float32) prosto z HDF5 wariantu `main`
(`outputs/echoes_36deg/<scena>/<scena>.h5`, zbior `depth`) -- te same wiersze, ktore
wybiera `EchoH5Dataset` (indeks budowany przez sama klase, nie odtwarzany tutaj),
podzial z `ml/dataset/splits.py` (plik `outputs/ml/splits/replica_locations.json`,
bez autobudowania). Piksel wazny = depth > 0, tak jak w metrykach.

Zbiory:
  * test@36 -- lokalizacje `test` x 36 orientacji (ogolem i per scena held-out);
  * train   -- lokalizacje `train` x 36 orientacji (warunek B), oraz podzbiory tego
               samego zbioru wybrane przez `angles.select_angles`: `cardinal`
               (warunek A) i `random_4` z angle_seed=0 (warunek D/ED).
Podzbiory A i ED nie wymagaja drugiego odczytu: sa podzbiorem wierszy B, wiec
liczniki per probka sa liczone raz (z 36 orientacji) i sumowane po kluczu
(scena, lokalizacja, kat).

Czytanie strumieniowe: jedna scena naraz, blokami po BLOCK wierszy (posortowane
indeksy -> sekwencyjny odczyt chunkow 1:1 z probka). W pamieci zostaja tylko
liczniki per probka (4 liczby calkowite).

Zapisuje liczniki pikseli (licznik, mianownik), nie tylko odsetki. Wynik:
`<DIR>/depth_range_stats.json` i `.md`; `--dry-run` -> `<DIR>/dryrun/` na 2
lokalizacjach na scene. Nie nadpisuje (blad, jesli plik wynikowy istnieje).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import MAX_DEPTH_REPLICA, DatasetConfig, EchoH5Dataset  # noqa: E402
from ..dataset.splits import HELDOUT_SCENES, Splits, load_splits  # noqa: E402

DEFAULT_OUT = paths.ML_OUTPUTS / "overnight_2026-11"
VARIANT = "main"

# Promien DWUSTRONNY okna echa: okno odpowiedzi impulsowej ma 60 ms, a dzwiek
# musi dojsc do sciany i wrocic, wiec najdalsza powierzchnia "widoczna" w echu
# lezy w odleglosci c * t / 2 = 343 m/s * 0,060 s / 2 = 10,29 m. Glebia powyzej
# tej wartosci nie ma odpowiednika w spektrogramie wejsciowym.
ECHO_WINDOW_RANGE_M = 10.29
# Progi kontekstowe (dla porownania), plus glowny.
THRESHOLDS_M = (8.0, ECHO_WINDOW_RANGE_M, 12.0)
BLOCK = 256
EXPECTED_TEST36 = 6588


def _kol(t: float) -> str:
    return f"gt_{t:g}m"


def _dataset(mode: str, subset: str, splits: Splits) -> EchoH5Dataset:
    return EchoH5Dataset(DatasetConfig(variant=VARIANT, mode=mode, angle_subset=subset,
                                       angle_seed=0, augment=False, return_meta=False),
                         splits=splits)


def _key(ds: EchoH5Dataset) -> np.ndarray:
    """Klucz probki niezalezny od kolejnosci: (nazwa sceny w liscie ds.scenes, loc, kat)."""
    return (ds.index_scene.astype(np.int64) * 10_000_000
            + ds.index_loc.astype(np.int64) * 1000 + ds.index_angle.astype(np.int64))


def per_sample_counts(ds: EchoH5Dataset) -> dict[str, np.ndarray]:
    """Liczniki per probka: n_valid oraz n(depth > prog) dla kazdego progu. Strumieniowo."""
    n = len(ds)
    cols = {"n_valid": np.zeros(n, dtype=np.int64),
            **{_kol(t): np.zeros(n, dtype=np.int64) for t in THRESHOLDS_M}}
    n_max = 0
    for si, scene in enumerate(ds.scenes):
        idx = np.flatnonzero(ds.index_scene == si)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(ds.index_row[idx], kind="stable")]
        rows = ds.index_row[idx]
        with h5py.File(ds.scene_paths[si], "r") as f:
            dset = f["depth"]
            for b in range(0, idx.size, BLOCK):
                sl = slice(b, b + BLOCK)
                r = rows[sl]
                depth = dset[r.tolist()]                 # (k, 128, 128) metry
                valid = depth > 0
                cols["n_valid"][idx[sl]] = valid.reshape(len(r), -1).sum(1)
                for t in THRESHOLDS_M:
                    cols[_kol(t)][idx[sl]] = (depth > t).reshape(len(r), -1).sum(1)
                n_max += int((depth > MAX_DEPTH_REPLICA).sum())
        print(f"    {scene}: {idx.size} probek", flush=True)
    cols["_n_gt_max_depth_total"] = np.array([n_max])
    return cols


def _agregat(cols: dict[str, np.ndarray], sel: np.ndarray | None = None) -> dict:
    s = slice(None) if sel is None else sel
    mian = int(cols["n_valid"][s].sum())
    out = {"n_samples": int(cols["n_valid"][s].size), "n_valid_pixels": mian, "progi": {}}
    for t in THRESHOLDS_M:
        lic = int(cols[_kol(t)][s].sum())
        out["progi"][f"{t:g}"] = {"licznik": lic, "mianownik": mian,
                                  "odsetek_pct": (100.0 * lic / mian) if mian else None}
    return out


def _subset_sel(full: EchoH5Dataset, sub: EchoH5Dataset) -> np.ndarray:
    """Pozycje probek podzbioru `sub` w indeksie `full` (dopasowanie po kluczu)."""
    if full.scenes != sub.scenes:
        raise RuntimeError("rozne listy scen w zbiorze pelnym i podzbiorze")
    kf, ks = _key(full), _key(sub)
    order = np.argsort(kf, kind="stable")
    pos = np.searchsorted(kf[order], ks)
    if np.any(pos >= kf.size) or not np.array_equal(kf[order][pos], ks):
        raise RuntimeError("podzbior zawiera probki spoza zbioru 36 orientacji")
    return order[pos]


def _okrojone(splits: Splits, k: int) -> Splits:
    """Dry-run: pierwsze k lokalizacji kazdej sceny w kazdym podzbiorze."""
    cut = lambda d: {s: v[:k] for s, v in d.items()}  # noqa: E731
    return Splits(train=cut(splits.train), val=cut(splits.val), test=cut(splits.test),
                  meta={**splits.meta, "dry_run_locations_per_scene": k})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"katalog wynikow (domyslnie {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="2 lokalizacje na scene, wyniki do <DIR>/dryrun/")
    args = ap.parse_args(argv)
    t0 = time.time()

    out_dir = args.out / "dryrun" if args.dry_run else args.out
    f_json, f_md = out_dir / "depth_range_stats.json", out_dir / "depth_range_stats.md"
    istnieje = [str(p) for p in (f_json, f_md) if p.exists()]
    if istnieje:
        print("BLAD: plik(i) wynikowe juz istnieja, nie nadpisuje:\n  " + "\n  ".join(istnieje),
              file=sys.stderr)
        return 2

    splits = load_splits(variant=VARIANT, autobuild=False)
    fingerprint = splits.meta.get("location_fingerprint")
    if args.dry_run:
        splits = _okrojone(splits, 2)

    P: dict = {"opis": "odsetek waznych pikseli glebi (depth > 0) z glebia > prog; wariant main",
               "prog_glowny_m": ECHO_WINDOW_RANGE_M,
               "prog_glowny_uzasadnienie": "promien dwustronny okna echa 60 ms: 343 m/s * 0,060 s / 2",
               "progi_m": list(THRESHOLDS_M), "max_depth_replica": MAX_DEPTH_REPLICA,
               "wariant": VARIANT, "split_fingerprint": fingerprint, "dry_run": bool(args.dry_run)}

    print("test@36 (lokalizacje test x 36 orientacji):")
    ds_t = _dataset("test", "all", splits)
    ct = per_sample_counts(ds_t)
    test = {"ogolem": _agregat(ct), "per_scena": {}}
    for si, sc in enumerate(ds_t.scenes):
        test["per_scena"][sc] = _agregat(ct, np.flatnonzero(ds_t.index_scene == si))
    test["sceny"] = list(ds_t.scenes)
    test["sceny_zgodne_z_held_out"] = sorted(ds_t.scenes) == sorted(HELDOUT_SCENES)
    test["n_samples_oczekiwane_pelny_przebieg"] = EXPECTED_TEST36
    test["n_samples_zgodne_6588"] = (len(ds_t) == EXPECTED_TEST36)
    test["n_pikseli_powyzej_max_depth"] = int(ct["_n_gt_max_depth_total"][0])
    P["test@36"] = test

    print("train (lokalizacje train x 36 orientacji, warunek B):")
    ds_b = _dataset("train", "all", splits)
    cb = per_sample_counts(ds_b)
    train = {"B_all_36": _agregat(cb)}
    for lab, spec in (("A_cardinal", "cardinal"), ("ED_random_4_angle_seed0", "random_4")):
        sel = _subset_sel(ds_b, _dataset("train", spec, splits))
        train[lab] = _agregat(cb, sel)
    train["B_all_36_per_scena"] = {sc: _agregat(cb, np.flatnonzero(ds_b.index_scene == si))
                                   for si, sc in enumerate(ds_b.scenes)}
    train["n_pikseli_powyzej_max_depth"] = int(cb["_n_gt_max_depth_total"][0])
    train["uwaga"] = ("A i ED to podzbiory wierszy B wybrane przez angles.select_angles "
                      "(cardinal; random_4 z angle_seed=0, jak warunek ED/D)")
    P["train"] = train
    P["czas_s"] = round(time.time() - t0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(f_json, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(P, indent=2, ensure_ascii=False))
    with open(f_md, "x", encoding="utf-8") as fh:
        fh.write(render_md(P))

    print(f"zapisano: {f_json}\n          {f_md}")
    g = test["ogolem"]
    print(f"test@36: n_samples={g['n_samples']} (6588? {test['n_samples_zgodne_6588']}), "
          f">{ECHO_WINDOW_RANGE_M} m: {g['progi'][f'{ECHO_WINDOW_RANGE_M:g}']['odsetek_pct']:.3f} %")
    for lab in ("B_all_36", "A_cardinal", "ED_random_4_angle_seed0"):
        v = train[lab]
        print(f"train {lab}: n_samples={v['n_samples']}, >{ECHO_WINDOW_RANGE_M} m: "
              f"{v['progi'][f'{ECHO_WINDOW_RANGE_M:g}']['odsetek_pct']:.3f} %")
    print(f"czas: {P['czas_s']} s")
    return 0


def render_md(P: dict) -> str:
    progi = [f"{t:g}" for t in P["progi_m"]]
    L = ["# Głębia poza zasięgiem okna echa", "",
         "Generowane przez `my-operations/ml/analysis/depth_range_stats.py`. Nie edytować ręcznie.",
         f"Odsetek WAŻNYCH pikseli (depth > 0) z głębią > próg. Próg główny **{P['prog_glowny_m']} m** "
         f"= {P['prog_glowny_uzasadnienie']}; progi 8 i 12 m jako kontekst. Wariant `{P['wariant']}`, "
         f"split `{P['split_fingerprint']}`.", ""]
    if P["dry_run"]:
        L += ["**DRY-RUN** — 2 lokalizacje na scenę; liczby NIE są wynikiem.", ""]
    hdr = "| zbiór | próbek | ważnych pikseli | " + " | ".join(f"> {p} m" for p in progi) + " |"
    sep = "|---|---|---|" + "---|" * len(progi)

    def row(name, a):
        cells = []
        for p in progi:
            v = a["progi"][p]
            pct = "—" if v["odsetek_pct"] is None else f"{v['odsetek_pct']:.3f} %"
            cells.append(f"{pct} ({v['licznik']}/{v['mianownik']})")
        return f"| {name} | {a['n_samples']} | {a['n_valid_pixels']} | " + " | ".join(cells) + " |"

    t = P["test@36"]
    L += ["## test@36", "", f"Liczba próbek: {t['ogolem']['n_samples']} "
          f"(oczekiwane w pełnym przebiegu {t['n_samples_oczekiwane_pelny_przebieg']}; "
          f"zgodne: {t['n_samples_zgodne_6588']}). Sceny: {', '.join(t['sceny'])}.", "", hdr, sep,
          row("test@36 ogółem", t["ogolem"])]
    for sc, a in t["per_scena"].items():
        L.append(row(sc, a))
    tr = P["train"]
    L += ["", "## Zbiór treningowy", "", tr["uwaga"] + ".", "", hdr, sep,
          row("B (36 orientacji)", tr["B_all_36"]), row("A (cardinal)", tr["A_cardinal"]),
          row("ED (random_4, angle_seed=0)", tr["ED_random_4_angle_seed0"]), "",
          "Per scena treningowa (B, 36 orientacji):", "", hdr, sep]
    for sc, a in tr["B_all_36_per_scena"].items():
        L.append(row(sc, a))
    L += ["", f"Pikseli > max_depth ({P['max_depth_replica']} m): test@36 {t['n_pikseli_powyzej_max_depth']}, "
              f"train B {tr['n_pikseli_powyzej_max_depth']}.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
