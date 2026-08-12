#!/usr/bin/env python
"""CLI strony DANYCH fazy uczenia: podzialy, weryfikacja, benchmark.

    python my-operations/ml/echo_data.py --make-splits
    python my-operations/ml/echo_data.py --stats --angle-subset every_4
    python my-operations/ml/echo_data.py --verify-loader --geometry main
    python my-operations/ml/echo_data.py --bench --workers 0,2,4,8

Zadna z tych operacji nie trenuje modelu. `--bench` jako jedyna dotyka GPU
(mikrobenchmark forward+backward) i mozna ja pominac flaga `--no-model`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Uruchamiany jako skrypt (`python my-operations/ml/echo_data.py`), wiec pakiet
# `ml` musi byc widoczny na sciezce -- inaczej dzialaloby tylko `python -m`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.dataset"

from . import angles as angles_mod  # noqa: E402
from .. import paths  # noqa: E402
from .echo_h5_dataset import DatasetConfig, EchoH5Dataset, expected_n_samples  # noqa: E402
from .splits import build_splits, load_splits, save_splits, split_path  # noqa: E402

GEOMETRIES = ("main", "patched")


def cmd_make_splits(args) -> int:
    fp = split_path()
    if fp.exists() and not args.force:
        splits = load_splits()
        print(f"Podzial juz istnieje: {fp}")
        print(f"  odcisk={splits.meta.get('location_fingerprint')} "
              f"zbudowany={splits.meta.get('built_utc')}")
        print("  (--force zeby przeliczyc; UWAGA: zmienia zbior testowy, "
              "co uniewaznia porownanie z wczesniejszymi przebiegami)")
    else:
        splits = build_splits(args.geometry)
        fp = save_splits(splits)
        print(f"Zapisano podzial: {fp}")
    for mode in ("train", "val", "test"):
        per = splits.locations(mode)
        print(f"  {mode:5s}: {len(per):2d} scen, {splits.n_locations(mode):4d} lokalizacji"
              f"  -> {expected_n_samples(splits, mode, 'all'):6d} probek przy 36 katach")
    print(f"  odcisk podzialu: {splits.meta['location_fingerprint']}")
    return 0


def cmd_stats(args) -> int:
    splits = load_splits(variant=args.geometry)
    subs = args.angle_subset.split(",") if args.angle_subset else [
        "cardinal", "every_6", "every_4", "every_3", "every_2", "all", "random_4"]
    print(f"wariant geometrii: {args.geometry}")
    print(f"{'subset':<12}{'kat/lok':>8}{'train':>9}{'val':>8}{'test':>8}{'RAZEM':>9}")
    print("-" * 54)
    for sub in subs:
        try:
            per = angles_mod.angles_per_location(sub)
        except angles_mod.AngleSubsetError as e:
            print(f"{sub:<12} BLAD: {e}")
            continue
        row = [expected_n_samples(splits, m, sub) for m in ("train", "val", "test")]
        print(f"{sub:<12}{per:>8}{row[0]:>9}{row[1]:>8}{row[2]:>8}{sum(row):>9}")
    if args.build_index:
        print("\nfaktyczna dlugosc datasetu (buduje indeks, czyta tylko metadane):")
        for sub in subs:
            ds = EchoH5Dataset(DatasetConfig(variant=args.geometry, mode="train", angle_subset=sub),
                               splits=splits)
            print(f"  train/{sub:<12} n={len(ds)}")
    return 0


def cmd_verify(args) -> int:
    from .verify_loader import run_all

    checks, fp = run_all(
        variant=args.geometry,
        full_depth_scan=not args.no_full_depth_scan,
        quick=args.quick,
    )
    print()
    print("=" * 72)
    print(f"WERYFIKACJA DATALOADERA -- wariant {args.geometry}")
    print("=" * 72)
    for c in checks:
        print(c.line())
    print("-" * 72)

    depth = next((c for c in checks if c.name.startswith("zakres glebi")), None)
    if depth is not None:
        r = depth.detail["_RAZEM"]
        print(f"glebia: max={r['global_max_m']} m wobec max_depth={r['max_depth_modelu_m']} m; "
              f"pikseli powyzej: {r['pikseli_powyzej_max_depth']} "
              f"({r['procent_powyzej']} %), zerowych: "
              f"{r['procent_zerowych_(maskowanych_w_stracie)']} %")
    counts = next((c for c in checks if c.name.startswith("licznosc")), None)
    if counts is not None:
        print("licznosc (RAZEM train+val+test):")
        for sub, row in counts.detail.items():
            t = row["total"]
            flag = "ok" if t["got"] == t["want"] else "ZLE"
            print(f"   {sub:<12} {row['angles_per_location']:>2} kat/lok  "
                  f"{t['got']:>6} (oczek. {t['want']:>6})  {flag}")
    ok = all(c.passed for c in checks)
    print("-" * 72)
    print(f"WYNIK: {'WSZYSTKO PRZESZLO' if ok else 'SA BLEDY'}   raport: {fp}")
    return 0 if ok else 1


def cmd_bench(args) -> int:
    from ..checks import bench

    workers = [int(x) for x in args.workers.split(",")]
    payload: dict = {"geometry": args.geometry, "batch_size": args.batch_size,
                     "angle_subset": args.angle_subset or "all"}

    print("=" * 78)
    print("BLOK 3: PRZEPUSTOWOSC DATALOADERA (bez modelu)")
    print("=" * 78)
    print(f"{'workers':>8}{'prefetch':>10}{'batches':>9}{'probek':>9}{'sek':>9}"
          f"{'probek/s':>11}{'wsad/s':>9}")
    print("-" * 78)
    loader_results = []
    for nw in workers:
        r = bench.bench_loader(
            variant=args.geometry, num_workers=nw, batch_size=args.batch_size,
            max_batches=None if args.full_epoch else args.max_batches,
            angle_subset=args.angle_subset or "all", augment=not args.no_augment,
        )
        loader_results.append(r)
        print(f"{r.num_workers:>8}{str(r.prefetch_factor):>10}{r.batches:>9}{r.samples:>9}"
              f"{r.seconds:>9.1f}{r.samples_per_s:>11.1f}{r.batches_per_s:>9.2f}")
    payload["loader"] = [vars(r) for r in loader_results]

    if args.prefetch:
        print()
        print("wariant: wiecej prefetch_factor przy najlepszej liczbie workerow")
        best_nw = max(loader_results, key=lambda r: r.samples_per_s).num_workers
        pf_results = []
        for pf in [int(x) for x in args.prefetch.split(",")]:
            r = bench.bench_loader(
                variant=args.geometry, num_workers=best_nw, batch_size=args.batch_size,
                prefetch_factor=pf, max_batches=args.max_batches,
                angle_subset=args.angle_subset or "all", augment=not args.no_augment,
            )
            pf_results.append(r)
            print(f"  workers={best_nw} prefetch={pf:<3} -> {r.samples_per_s:8.1f} probek/s")
        payload["prefetch"] = [vars(r) for r in pf_results]

    if args.uncompressed_scene:
        print()
        print("wariant: ten sam zbior BEZ kompresji (jedna scena)")
        best_nw = max(loader_results, key=lambda r: r.samples_per_s).num_workers
        gz, un, sizes = bench.bench_uncompressed(
            args.uncompressed_scene, variant=args.geometry, num_workers=best_nw,
            batch_size=args.batch_size, max_batches=args.max_batches,
            augment=not args.no_augment,
        )
        print(f"  gzip         : {gz.samples_per_s:8.1f} probek/s  ({sizes['gzip_MB']} MB)")
        print(f"  bez kompresji: {un.samples_per_s:8.1f} probek/s  ({sizes['uncompressed_MB']} MB)")
        print(f"  przyspieszenie {sizes['przyspieszenie']}x, "
              f"rozmiar x{sizes['wspolczynnik_kompresji']}")
        payload["uncompressed"] = {"gzip": vars(gz), "uncompressed": vars(un), **sizes}

    if not args.no_model:
        print()
        print("=" * 78)
        print("BLOK 3: JEDNA ITERACJA TRENINGU (dane syntetyczne, sam GPU)")
        print("=" * 78)
        print(f"{'AMP':>6}{'batch':>8}{'ms/iter':>10}{'probek/s':>11}{'peak GB':>10}")
        print("-" * 78)
        model_results = []
        for amp in (False, True):
            r = bench.bench_model(batch_size=args.batch_size, amp=amp, iters=args.model_iters)
            model_results.append(r)
            print(f"{str(amp):>6}{r.batch_size:>8}{r.ms_per_iter:>10.2f}"
                  f"{r.samples_per_s:>11.1f}{r.peak_mem_gb:>10.2f}")
        payload["model"] = [vars(r) for r in model_results]

        print()
        print("=" * 78)
        print(f"SZACUNEK CZASU PRZEBIEGU ({args.steps} krokow gradientu)")
        print("=" * 78)
        best_loader = max(loader_results, key=lambda r: r.samples_per_s)
        est = {}
        for r in model_results:
            e = bench.estimate_run_hours(r.ms_per_iter, best_loader.samples_per_s,
                                         args.batch_size, args.steps)
            est[f"amp={r.amp}"] = e
            print(f"  AMP={str(r.amp):<5} gpu={e['s_na_krok_gpu']:.4f}s/krok  "
                  f"io={e['s_na_krok_io']:.4f}s/krok  -> {e['s_na_krok_efektywnie']:.4f}s/krok  "
                  f"= {e['godzin_bez_walidacji']:.2f} h   [waskie gardlo: {e['waskie_gardlo']}]")
        payload["estimate"] = est
        payload["best_loader"] = vars(best_loader)

    fp = bench.save_report(payload, f"bench_{args.geometry}")
    print()
    print(f"raport: {fp}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry", choices=GEOMETRIES, default="main",
                   help="wariant geometrii sceny (GENERATOR_PARAMS.md §4.5)")
    p.add_argument("--angle-subset", default=None,
                   help="all | cardinal | every_N | random_K (lista po przecinku dla --stats)")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-splits", action="store_true", help="zbuduj/pokaz podzial train/val/test")
    g.add_argument("--stats", action="store_true", help="licznosci probek per podzbior katow")
    g.add_argument("--verify-loader", action="store_true", help="pelna weryfikacja (Blok 2)")
    g.add_argument("--bench", action="store_true", help="benchmark przepustowosci (Blok 3)")

    p.add_argument("--force", action="store_true", help="--make-splits: przelicz mimo istniejacego pliku")
    p.add_argument("--build-index", action="store_true", help="--stats: potwierdz licznosc budujac indeks")
    p.add_argument("--quick", action="store_true", help="--verify-loader: skrocona wersja")
    p.add_argument("--no-full-depth-scan", action="store_true",
                   help="--verify-loader: nie skanuj calej glebi (szybciej, ale %% powyzej max_depth bedzie z probki)")

    p.add_argument("--workers", default="0,2,4,8", help="--bench: lista num_workers")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=300, help="--bench: ile wsadow na konfiguracje")
    p.add_argument("--full-epoch", action="store_true", help="--bench: pelna epoka zamiast --max-batches")
    p.add_argument("--prefetch", default=None, help="--bench: lista prefetch_factor do sprawdzenia, np. 2,4,8")
    p.add_argument("--uncompressed-scene", default=None,
                   help="--bench: nazwa sceny do przepisania bez kompresji i porownania")
    p.add_argument("--no-model", action="store_true", help="--bench: pomin mikrobenchmark GPU")
    p.add_argument("--no-augment", action="store_true", help="--bench: bez augmentacji PIL")
    p.add_argument("--model-iters", type=int, default=30)
    p.add_argument("--steps", type=int, default=40000, help="liczba krokow gradientu do szacunku czasu")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.make_splits:
        return cmd_make_splits(args)
    if args.stats:
        return cmd_stats(args)
    if args.verify_loader:
        return cmd_verify(args)
    if args.bench:
        return cmd_bench(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
