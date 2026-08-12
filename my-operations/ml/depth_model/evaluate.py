#!/usr/bin/env python
"""BLOK 2: protokol ewaluacji. Wszystko z JEDNEGO checkpointu, zero dodatkowych przebiegow.

    # ewaluacja jednego przebiegu (GPU: jeden przelot po zbiorze testowym)
    python my-operations/ml/evaluate.py --run-dir outputs/ml/runs/A_seed0

    # porownanie sparowane dwoch warunkow -- czyta zapisane tabele, ZERO GPU
    python my-operations/ml/evaluate.py --compare A_seed0 B_seed0

    # kontrola permutacyjna echa na GOTOWYM checkpoincie (Blok 1.2, wersja darmowa)
    python my-operations/ml/evaluate.py --run-dir outputs/ml/runs/B_seed0 --shuffle-echo

Kluczowa wlasnosc: wszystkie punkty 2.1-2.6 licza sie z tej samej tabeli
statystyk per probka (`metrics.SampleTable`), zbieranej w jednym przelocie po
zbiorze testowym. Tabela idzie na dysk (`samples_test36.npz`), wiec kazde
pozniejsze grupowanie, bootstrap i porownanie miedzy warunkami nie dotyka juz
ani GPU, ani zbioru danych.

CO SIE RAPORTUJE I DLACZEGO:

2.1  DWA ZBIORY TESTOWE.  `test@36` jest PODSTAWOWY -- agent moze stac zwrocony
     dowolnie i to jest sytuacja docelowa. Warunek `cardinal` jest tu oceniany
     na 32 katach, ktorych nigdy nie widzial, i to jest wlasnie mierzone.
     `test@4` to kolumna dodatkowa, dla zgodnosci z ukladem Gao/Paridy.

2.2  RMSE W FUNKCJI ODLEGLOSCI KATOWEJ od najblizszego kata OBECNEGO W ZBIORZE
     TRENINGOWYM warunku. Definicja jest globalna (zbior katow wystepujacych w
     calym zbiorze treningowym), nie per lokalizacja -- i to jest cala roznica
     miedzy `cardinal` a `random_4`: pierwszy nie widzial kata 10 stopni NIGDY,
     drugi widzial go w innych punktach. Jesli istnieje luka generalizacji
     katowej, bedzie widoczna jako monotoniczna krzywa.

2.3  BOOTSTRAP SPAROWANY PO LOKALIZACJACH. Patrz `metrics.bootstrap_paired_by_location`.

2.4  ROZBICIE PER SCENA Z LICZBA WAZNYCH PIKSELI. `frl_apartment_5` ma 14,24 %
     dziur, `office_4` 0,0012 % -- RMSE liczone na 86 % kadru i na 100 % kadru to
     nie jest ta sama wielkosc.

2.5  `office_4` JAKO SONDA TRANSFERU GEOMETRII. Scena jest szczelna, wiec w obu
     wariantach serwowana z tego samego pliku i jej probki testowe sa
     BIT-IDENTYCZNE (potwierdzone w `geometry_check.py`, kontrola 0.2). Roznica
     wyniku na `office_4` miedzy modelem trenowanym na `main` a trenowanym na
     `patched` mierzy czysto transfer, przy danych testowych trzymanych
     doslownie stalych.

2.6  STRATYFIKACJA OTWARTE / SZCZELNE. Jesli efekt gestosci rozni sie miedzy
     scenami z dziura a szczelnymi juz w samym `main`, odpowiedz o wplywie
     geometrii jest dostepna BEZ uruchamiania wariantu `patched`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.depth_model"

from ..dataset import angles as angles_mod  # noqa: E402
from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import DatasetConfig, EchoH5Dataset  # noqa: E402
from ..matrix.experiments import CONDITIONS_BY_ID  # noqa: E402
from ..analysis.geometry_check import PATCHED_SCENES, SEALED_SCENES  # noqa: E402
from .metrics import (  # noqa: E402
    STRATA,
    SampleStatsCollector,
    SampleTable,
    bootstrap_paired_by_location,
    min_distance_to_grid,
)
from ..dataset.splits import load_splits  # noqa: E402

# Zbiory testowe raportowane zawsze oba, z tego samego checkpointu (2.1).
TEST_SETS = {"test@36": "all", "test@4": "cardinal"}


def _eval_dir(run_id: str) -> Path:
    return paths.ML_OUTPUTS / "eval" / run_id


# ------------------------------------------------------- siatka kątów treningowych


def training_angle_grid(condition, splits) -> np.ndarray:
    """Zbior katow, ktore w OGOLE wystepuja w zbiorze treningowym warunku.

    Dla `cardinal`/`every_N` to staly zbior. Dla `random_4` katy sa losowane per
    lokalizacja, wiec suma po 1 374 lokalizacjach niemal na pewno pokrywa cala
    siatke 36 -- i to jest merytoryczna roznica miedzy warunkiem A a D: A nie
    widzial kata 10 stopni ANI RAZU, D widzial go w innych punktach.
    """
    seen: set[int] = set()
    for scene, loc_ids in splits.locations("train").items():
        for loc in loc_ids:
            sel = angles_mod.select_angles(condition.angle_subset, scene=scene,
                                           location_id=int(loc), seed=condition.angle_seed)
            seen.update(int(a) for a in sel)
            if len(seen) == angles_mod.N_ANGLES:
                return np.arange(0, 360, angles_mod.ANGLE_STEP_DEG)
    return np.asarray(sorted(seen), dtype=np.int64)


# ------------------------------------------------------------------ zbieranie


def load_model(run_dir: Path, device, splits):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cond_id = cfg["condition"]["id"]
    cond = CONDITIONS_BY_ID[cond_id]
    fast = bool(cfg.get("spec", {}).get("extra", {}).get("fast_bilinear", False))

    from .train_condition import build_model
    model, nets = build_model(cond, device, splits, fast_bilinear=fast)
    loaded = {}
    for k, net in nets.items():
        p = run_dir / f"best_{k}.pth"
        if not p.exists():
            raise FileNotFoundError(f"brak wag {p} -- przebieg nie zapisal jeszcze checkpointu")
        net.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        loaded[k] = str(p.name)
    model.eval()
    return model, cond, cfg, loaded


@torch.no_grad()
def collect_table(model, cond, splits, *, angle_subset: str, device, batch_size: int,
                  num_workers: int, amp: bool, edge_threshold: float,
                  mask_variant: str | None, shuffle_echo_seed: int | None) -> SampleTable:
    # KONTROLA PERMUTACYJNA (Blok 1.2, wersja bez treningu): echo brane z LOSOWO
    # INNEJ probki tego samego zbioru, obraz i glebia na miejscu.
    #
    # Permutacja pochodzi z `EchoH5Dataset`, a nie jest robiona tutaj drugi raz:
    # tamta wersja gwarantuje bijekcje ORAZ wyklucza echo z TEJ SAMEJ lokalizacji
    # (co przy 36 orientacjach na lokalizacje jest istotne -- echo z innego kata
    # tej samej pozycji nadal niesie pelna informacje o polozeniu). Druga,
    # rownolegla implementacja tej samej rzeczy mogla by sie po cichu rozjechac
    # z warunkiem `SE`, a wtedy wersja darmowa i wersja trenowana mierzylyby
    # dwie rozne rzeczy.
    ds = EchoH5Dataset(
        DatasetConfig(variant=cond.geometry, mode="test", angle_subset=angle_subset,
                      angle_seed=cond.angle_seed, augment=False, mask_variant=mask_variant,
                      shuffle_echo_seed=shuffle_echo_seed),
        splits=splits)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)
    col = SampleStatsCollector(ds.scenes, edge_threshold)

    n_fixed = 0
    if ds.index_echo_src is not None:
        n_fixed = int((ds.index_echo_src == np.arange(len(ds))).sum())

    for batch in loader:
        dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(dev)
        col.update(out["depth_predicted"].float(), out["depth_gt"].float(),
                   scene_idx=dev["scene_idx"], location_id=dev["location_id"],
                   angle_deg=dev["angle_deg"],
                   valid_ref=dev.get("valid_ref"))
    info = {"n_samples": len(ds), "shuffled_echo": ds.index_echo_src is not None,
            "shuffle_fixed_points": n_fixed}
    ds.close()
    return col.table(), info


# ------------------------------------------------------------------- raporty


def analyse(tab: SampleTable, *, train_grid: np.ndarray, label: str) -> dict:
    """Wszystkie grupowania Bloku 2 z jednej tabeli."""
    res: dict = {"label": label, "n_samples": len(tab),
                 "n_locations": int(np.unique(tab.location_key()).size),
                 "train_angle_grid": train_grid.tolist(),
                 "overall": tab.aggregate()}

    # 2.2 -- RMSE w funkcji odleglosci katowej od siatki treningowej.
    dist = min_distance_to_grid(tab.cols["angle_deg"], train_grid)
    buckets = {}
    for d in sorted(set(dist.tolist())):
        rows = np.flatnonzero(dist == d)
        a = tab.aggregate(rows)
        buckets[f"{d:.0f}"] = {
            "n_samples": int(rows.size),
            **{st: {"RMSE": a[st]["RMSE"], "MAE": a[st]["MAE"],
                    "DELTA1": a[st]["DELTA1"], "n_pixels": a[st]["n_pixels"]}
               for st in STRATA},
        }
    res["by_angular_distance"] = buckets
    if len(buckets) > 1:
        ks = sorted(buckets, key=float)
        r = [buckets[k]["all"]["RMSE"] for k in ks]
        res["angular_gap"] = {
            "rmse_at_0deg": r[0], "rmse_at_max": r[-1],
            "max_distance_deg": float(ks[-1]),
            "delta_rmse": r[-1] - r[0],
            "relative_pct": 100.0 * (r[-1] - r[0]) / r[0] if r[0] else float("nan"),
            "monotonic": bool(all(x <= y + 1e-12 for x, y in zip(r, r[1:]))),
        }

    # 2.4 -- per scena, z liczba i odsetkiem waznych pikseli.
    per_scene = {}
    for si, name in enumerate(tab.scene_names):
        rows = np.flatnonzero(tab.cols["scene_idx"] == si)
        if rows.size:
            per_scene[name] = tab.aggregate(rows)
    res["per_scene"] = per_scene

    # 2.5 -- sonda transferu geometrii. Wyodrebniona jawnie, bo jej wartosc
    # bierze sie z tego, ze dane testowe sa doslownie te same w obu wariantach.
    if "office_4" in per_scene:
        res["office_4_probe"] = {
            "uwaga": "scena szczelna: plik HDF5 wspolny dla obu wariantow geometrii, "
                     "probki testowe bit-identyczne (geometry_check.py, kontrola 0.2)",
            **per_scene["office_4"],
        }

    # 2.6 -- stratyfikacja otwarte / szczelne wewnatrz wariantu.
    groups = {"otwarte_z_dziura": PATCHED_SCENES, "szczelne": SEALED_SCENES}
    strat = {}
    for gname, scenes in groups.items():
        sids = [i for i, n in enumerate(tab.scene_names) if n in scenes]
        rows = np.flatnonzero(np.isin(tab.cols["scene_idx"], sids))
        if rows.size:
            strat[gname] = {"scenes": [tab.scene_names[i] for i in sids
                                       if (tab.cols["scene_idx"] == i).any()],
                            **tab.aggregate(rows)}
    res["by_geometry_group"] = strat
    return res


def _assert_same_protocol(run_a: str, run_b: str, *, force: bool = False) -> None:
    """Oba przebiegi musza pochodzic z tego samego protokolu walidacji i maski.

    DLACZEGO TO ISTNIEJE. Na dysku leza checkpointy sprzed i po decyzji z
    2026-08-11 §1 (walidacja na wlasnym podzbiorze katow wobec pelnych 36).
    Checkpoint wybrany innym kryterium to inny model, a `--compare` porownywalo
    je bez slowa ostrzezenia -- roznica wygladalaby jak efekt warunku, a bylaby
    efektem zmiany protokolu. To samo dotyczy `mask_mode`: metryki liczone na
    innej masce nie sa ta sama wielkoscia.
    """
    def read(run: str) -> dict:
        p = paths.RUNS_DIR / run / "status.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    sa, sb = read(run_a), read(run_b)
    if not sa or not sb:
        return  # brak status.json (np. przebieg pretekstowy) -- nie ma czego porownac
    diffs = []
    for key, default in (("val_angle_subset", "<sprzed 2026-08-11>"), ("mask_mode", "intersection")):
        va, vb = sa.get(key, default), sb.get(key, default)
        if va != vb:
            diffs.append(f"{key}: {run_a}={va!r} vs {run_b}={vb!r}")
    if diffs and not force:
        raise SystemExit(
            "BLAD: przebiegi uzyly ROZNYCH protokolow, porownanie bylo by mylace:\n  "
            + "\n  ".join(diffs)
            + "\nPrzelicz starszy przebieg albo swiadomie wymus flaga --force-compare."
        )
    if diffs:
        print("UWAGA: rozne protokoly, wymuszone --force-compare:\n  " + "\n  ".join(diffs))


def compare(tab_a: SampleTable, tab_b: SampleTable, name_a: str, name_b: str,
            *, n_boot: int, seed: int) -> dict:
    """2.3 -- roznica z przedzialem ufnosci, sparowana po lokalizacjach."""
    out: dict = {"a": name_a, "b": name_b, "n_boot": n_boot}
    for st in STRATA:
        out[st] = bootstrap_paired_by_location(tab_a, tab_b, stratum=st,
                                               metric="RMSE", n_boot=n_boot, seed=seed)
    # Ten sam bootstrap na grupach 2.6 -- czy efekt rozni sie miedzy scenami
    # z dziura a szczelnymi.
    for gname, scenes in (("otwarte_z_dziura", PATCHED_SCENES), ("szczelne", SEALED_SCENES)):
        sids = [i for i, n in enumerate(tab_a.scene_names) if n in scenes]
        rows = np.flatnonzero(np.isin(tab_a.cols["scene_idx"], sids))
        if rows.size:
            out[f"group.{gname}"] = bootstrap_paired_by_location(
                tab_a, tab_b, stratum="all", metric="RMSE", n_boot=n_boot,
                seed=seed, rows_a=rows, rows_b=rows)
    return out


# ------------------------------------------------------------------ sterowanie


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, help="katalog przebiegu (outputs/ml/runs/<id>)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="porownaj dwa JUZ ZEWALUOWANE przebiegi po id (zero GPU)")
    ap.add_argument("--test-set", choices=sorted(TEST_SETS) + ["both"], default="both")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--edge-threshold", type=float, default=0.10)
    ap.add_argument("--intersection-mask", action="store_true",
                    help="punktuj wylacznie piksele wazne w OBU wariantach geometrii "
                         "(wymagane przy porownaniu main vs patched, patrz Blok 0.5)")
    ap.add_argument("--shuffle-echo", action="store_true",
                    help="Blok 1.2 (wersja darmowa): echo z losowo innej probki")
    ap.add_argument("--shuffle-seed", type=int, default=1234)
    ap.add_argument("--force-compare", action="store_true",
                    help="porownaj mimo roznych protokolow walidacji/maski (patrz _assert_same_protocol)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.compare:
        a, b = args.compare
        # `test@4` sluzy do porownan SPAROWANYCH na katach, ktore oba modele
        # widzialy -- patrz poprawka 4.1 raportu 2026-08-11: zestawianie
        # `EA@test4` z `EB@test36` bylo porownaniem przez DWA ROZNE zbiory.
        label = args.test_set if args.test_set != "both" else "test@36"
        _assert_same_protocol(a, b, force=args.force_compare)
        ta = SampleTable.load(_eval_dir(a) / f"samples_{label}.npz")
        tb = SampleTable.load(_eval_dir(b) / f"samples_{label}.npz")
        res = compare(ta, tb, a, b, n_boot=args.n_boot, seed=args.boot_seed)
        res["test_set"] = label
        suffix = "" if label == "test@36" else f"_{label.replace('@', '')}"
        out = paths.ML_OUTPUTS / "eval" / f"compare_{a}_vs_{b}{suffix}.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                       encoding="utf-8")
        print(f"POROWNANIE SPAROWANE  {a}  vs  {b}   (bootstrap po lokalizacjach)")
        for st in STRATA:
            r = res[st]
            print(f"  {st:8s} dRMSE = {r['delta_point']:+.5f}  "
                  f"95% CI [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]  "
                  f"{'ISTOTNE' if r['ci_excludes_zero'] else 'nieistotne'}  "
                  f"(n_lok={r['n_locations']})")
        print(f"\nzapisano: {out}")
        return 0

    if not args.run_dir:
        ap.error("podaj --run-dir albo --compare")

    run_id = args.run_dir.name
    device = torch.device("cuda")
    amp = not args.no_amp
    t0 = time.perf_counter()

    splits = load_splits()
    model, cond, cfg, loaded = load_model(args.run_dir, device, splits)
    grid = training_angle_grid(cond, splits)
    mask_variant = ("main" if cond.geometry == "patched" else "patched") if args.intersection_mask else None

    out_dir = _eval_dir(run_id + ("_shuffled_echo" if args.shuffle_echo else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"EWALUACJA {run_id}   warunek {cond.id} ({cond.model}, geometria {cond.geometry})")
    print(f"  siatka katow treningowych: {grid.tolist() if grid.size <= 8 else str(grid.size) + ' katow (pelna)'}")
    print(f"  maska przeciecia: {mask_variant or 'wylaczona'}")
    print(f"  permutacja echa : {'TAK, ziarno ' + str(args.shuffle_seed) if args.shuffle_echo else 'nie'}")
    print("=" * 74)

    payload: dict = {
        "run_id": run_id, "condition": cond.id, "model": cond.model,
        "geometry": cond.geometry, "angle_subset": cond.angle_subset,
        "seed": cfg["spec"]["seed"], "weights": loaded,
        "split_fingerprint": splits.meta.get("location_fingerprint"),
        "intersection_mask_variant": mask_variant,
        "shuffled_echo": bool(args.shuffle_echo),
        "shuffle_seed": args.shuffle_seed if args.shuffle_echo else None,
        "edge_threshold_m_per_px": args.edge_threshold,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_sets": {},
    }

    wanted = TEST_SETS if args.test_set == "both" else {args.test_set: TEST_SETS[args.test_set]}
    for label, subset in wanted.items():
        tab, info = collect_table(
            model, cond, splits, angle_subset=subset, device=device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            amp=amp, edge_threshold=args.edge_threshold, mask_variant=mask_variant,
            shuffle_echo_seed=args.shuffle_seed if args.shuffle_echo else None)
        tab.save(out_dir / f"samples_{label}.npz")
        payload["test_sets"][label] = analyse(tab, train_grid=grid, label=label)
        payload["test_sets"][label]["collect_info"] = info

        o = payload["test_sets"][label]["overall"]
        print(f"\n  {label}:  n={len(tab)}  RMSE {o['all']['RMSE']:.5f}  "
              f"krawedzie {o['edge']['RMSE']:.5f}  gladkie {o['smooth']['RMSE']:.5f}  "
              f"d1 {o['all']['DELTA1']:.4f}  waznych px {o['all']['valid_pixel_fraction']*100:.2f} %")
        bd = payload["test_sets"][label]["by_angular_distance"]
        if len(bd) > 1:
            print("      RMSE wg odleglosci katowej od siatki treningowej:")
            for k in sorted(bd, key=float):
                print(f"        {k:>4s} st.  n={bd[k]['n_samples']:5d}  "
                      f"RMSE {bd[k]['all']['RMSE']:.5f}  krawedzie {bd[k]['edge']['RMSE']:.5f}")
            g = payload["test_sets"][label]["angular_gap"]
            print(f"      luka {g['delta_rmse']:+.5f} ({g['relative_pct']:+.2f} %), "
                  f"monotoniczna: {g['monotonic']}")
        print("      per scena (RMSE / % waznych px):")
        for s, v in payload["test_sets"][label]["per_scene"].items():
            print(f"        {s:18s} {v['all']['RMSE']:.5f}   "
                  f"{v['all']['valid_pixel_fraction']*100:6.2f} %   n={v['n_samples']}")

    payload["seconds"] = round(time.perf_counter() - t0, 1)
    (out_dir / "eval.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nzapisano: {out_dir / 'eval.json'}  ({payload['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
