#!/usr/bin/env python
"""KONTROLA RGB-ONLY: sama siec obrazowa `RGBDepthNet` na TYCH SAMYCH parach co A / D / B.

    # pelny przebieg (trening 40 000 krokow + ewaluacja test@36 i test@4)
    python my-operations/ml/depth_model/train_rgbonly.py --condition A --seed 0

    # weryfikacja rownosci par (lokalizacja, kat) z train_condition.py -- BEZ GPU
    python my-operations/ml/depth_model/train_rgbonly.py --check-pairs

    # proba generalna: kilkadziesiat krokow, 1 walidacja na czesci, ewaluacja na 2 wsadach
    python my-operations/ml/depth_model/train_rgbonly.py --condition A --seed 0 --dry-run

PO CO. Pelny model (RGB + echo + material + uwaga) w warunkach A (`cardinal`),
D (`random_4`, angle_seed=0) i B (`all`) trenuje sie przez `train_condition.py`.
Bez kontroli "sam obraz na tych samych danych" nie wiadomo, ile z wyniku
pelnego modelu wnosi galaz audio, a ile sama siec obrazowa, ktora w warunku B
widzi po prostu 9x wiecej kadrow niz w A. Ta kontrola daje dokladnie te liczbe:
ten sam `RGBDepthNet` Paridy, ta sama strata, ten sam optymalizator, ten sam
budzet krokow i TE SAME pary (lokalizacja, kat) -- rozni sie wylacznie tym, ze
nie ma echa, materialu ani uwagi.

CO JEST WSPOLNE Z `train_condition.py` (importowane, nie kopiowane):
  * `set_seed`, `InfiniteLoader`, `evaluate` (walidacja z `val@4` z tych samych
    predykcji), `VAL_ANGLE_SUBSET` = "all" -- walidacja ZAWSZE na pelnych 36 katach
    (6 588 probek), checkpoint wybierany po najlepszym RMSE walidacyjnym `all`;
  * hiperparametry z `RunSpec` (Adam, lr 1e-4, beta1 0,9, wd 5e-4), budzet
    `TOTAL_STEPS`, `BATCH_SIZE`, AMP + GradScaler, `LogDepthLoss` na pikselach
    `depth_gt != 0`, skala `MAX_DEPTH_REPLICA`;
  * `DatasetConfig` zbioru treningowego budowany z tych samych pol warunku
    (`geometry`, `angle_subset`, `angle_seed`) -- rownosc par sprawdza `--check-pairs`.
Model `RGBOnlyModel` pochodzi z `pretext_model/transfer.py` (warunek `scratch`).
Ewaluacja testowa to `evaluate.collect_table` + `evaluate.analyse`, czyli ta
sama agregacja (`SampleTable.aggregate`) i ten sam format `eval.json`, co dla
pelnego modelu.

CZEGO TU CELOWO NIE MA:
  * `checkpoint.pt` z momentami Adama -- zapisujemy wylacznie `best_rgbdepth.pth`;
    konsekwencja: przebiegu nie da sie wznowic (`--resume` nie istnieje).
    Przebieg trwa kilkanascie minut, wiec wznawianie nie jest potrzebne;
  * `fast_bilinear` -- dotyczy wylacznie `attentionNet`, ktorej tu nie ma;
  * nadpisywania: istniejacy katalog przebiegu = blad (brak `--force`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.depth_model"

from .. import paths  # noqa: E402
from ..dataset import angles as angles_mod  # noqa: E402
from ..dataset.echo_h5_dataset import (  # noqa: E402
    MAX_DEPTH_REPLICA,
    DatasetConfig,
    EchoH5Dataset,
    build_dataloader,
)
from ..dataset.splits import load_splits  # noqa: E402
from ..matrix.experiments import BATCH_SIZE, CONDITIONS_BY_ID, TOTAL_STEPS, RunSpec  # noqa: E402
from ..pretext_model.transfer import RGBOnlyModel  # noqa: E402
from .evaluate import TEST_SETS, analyse, collect_table, training_angle_grid  # noqa: E402
from .metrics import SampleStatsCollector  # noqa: E402
from .train_condition import VAL_ANGLE_SUBSET, InfiniteLoader, set_seed  # noqa: E402
from .train_condition import evaluate as validate  # noqa: E402

# Warunki, dla ktorych ta kontrola ma sens: trzy warunki glowne macierzy.
RGBONLY_CONDITIONS = ("A", "D", "B")

# Oczekiwane licznosci -- twarde liczby z config.json przebiegow pelnego modelu
# (`outputs/ml/runs/{A,D,B}_seed0/config.json`). Rozjazd = inny zbior, przebieg
# nie startuje.
EXPECTED_N = {"train": {"A": 5_496, "D": 5_496, "B": 49_464},
              "val": 6_588, "test@36": 6_588, "test@4": 732}

DEFAULT_OUT_ROOT = paths.ML_OUTPUTS / "overnight_2026-11" / "rgbonly"

# Dry-run: ile krokow, ile pierwszych krokow pominac w pomiarze czasu (start
# workerow DataLoadera, autotuning cuDNN), ile probek walidacji, ile wsadow testu.
DRY_STEPS = 60
DRY_WARMUP = 20
DRY_VAL_SAMPLES = 320
DRY_TEST_BATCHES = 2


# ------------------------------------------------------------ konfiguracje danych


def train_data_config(cond) -> DatasetConfig:
    """Zbior treningowy kontroli RGB-only -- z TYCH SAMYCH pol warunku, z ktorych
    `train_condition.py` buduje zbior pelnego modelu. Permutacja echa nie wchodzi
    w gre (A/D/B jej nie maja, a siec i tak echa nie czyta)."""
    if getattr(cond, "shuffle_echo_seed", None) is not None:
        raise ValueError(f"warunek {cond.id} ma permutacje echa -- kontrola RGB-only jej nie obsluguje")
    return DatasetConfig(variant=cond.geometry, mode="train",
                         angle_subset=cond.angle_subset, angle_seed=cond.angle_seed)


def val_data_config(cond) -> DatasetConfig:
    return DatasetConfig(variant=cond.geometry, mode="val", angle_subset=VAL_ANGLE_SUBSET,
                         angle_seed=cond.angle_seed, augment=False)


def reference_train_config_from_train_condition(cond) -> DatasetConfig:
    """DOSLOWNIE wyrazenie z `train_condition.main()` (zbior treningowy). Nie da
    sie go zaimportowac, bo siedzi w ciele `main()`; trzymane osobno od
    `train_data_config`, zeby porownanie bylo porownaniem dwoch sciezek, a nie
    jednej funkcji z sama soba."""
    shuffle_seed = getattr(cond, "shuffle_echo_seed", None)
    return DatasetConfig(variant=cond.geometry, mode="train",
                         angle_subset=cond.angle_subset, angle_seed=cond.angle_seed,
                         shuffle_echo_seed=shuffle_seed)


def sample_keys(ds: EchoH5Dataset) -> list[tuple[str, int, int]]:
    """(scena, location_id, kat) kazdej probki -- po NAZWIE sceny, nie indeksie."""
    return [(ds.scenes[int(s)], int(l), int(a))
            for s, l, a in zip(ds.index_scene, ds.index_loc, ds.index_angle)]


def keys_fingerprint(keys) -> str:
    h = hashlib.sha256()
    for s, l, a in sorted(keys):
        h.update(f"{s}|{l}|{a}\n".encode())
    return h.hexdigest()[:16]


def _independent_keys(splits, mode: str, angle_subset: str, angle_seed: int) -> set:
    """Trzecia, niezalezna od `EchoH5Dataset` droga: podzial x `select_angles`."""
    out = set()
    for scene, locs in splits.locations(mode).items():
        for loc in locs:
            for a in angles_mod.select_angles(angle_subset, scene=scene,
                                              location_id=int(loc), seed=angle_seed):
                out.add((scene, int(loc), int(a)))
    return out


def _hdf5_rows_match(ds: EchoH5Dataset) -> bool:
    """Czy wiersz HDF5 wskazany przez indeks naprawde niesie deklarowane
    (location_id, kat). Czyta wylacznie male tablice metadanych."""
    for si, p in enumerate(ds.scene_paths):
        sel = np.flatnonzero(ds.index_scene == si)
        if sel.size == 0:
            continue
        with h5py.File(p, "r") as f:
            loc = f["location_id"][:].astype(np.int64)
            ang = f["angle_deg"][:].astype(np.int64)
        rows = ds.index_row[sel]
        if not (np.array_equal(loc[rows], ds.index_loc[sel].astype(np.int64))
                and np.array_equal(ang[rows], ds.index_angle[sel].astype(np.int64))):
            return False
    return True


def check_pairs(cond_ids, splits) -> dict:
    """Rownosc zbiorow (scena, lokalizacja, kat): RGB-only vs train_condition.py."""
    report: dict = {"split_fingerprint": splits.meta.get("location_fingerprint"),
                    "conditions": {}, "ok": True}
    for cid in cond_ids:
        cond = CONDITIONS_BY_ID[cid]
        mine = EchoH5Dataset(train_data_config(cond), splits=splits)
        ref = EchoH5Dataset(reference_train_config_from_train_condition(cond), splits=splits)
        km, kr = sample_keys(mine), sample_keys(ref)
        sm, sr = set(km), set(kr)
        indep = _independent_keys(splits, "train", cond.angle_subset, cond.angle_seed)
        full_cfg = paths.RUNS_DIR / f"{cid}_seed0" / "config.json"
        n_full_run = (json.loads(full_cfg.read_text(encoding="utf-8")).get("n_train")
                      if full_cfg.exists() else None)
        r = {
            "angle_subset": cond.angle_subset, "angle_seed": cond.angle_seed,
            "geometry": cond.geometry,
            "n_rgbonly": len(km), "n_train_condition": len(kr),
            "n_expected": EXPECTED_N["train"][cid],
            "n_from_full_model_run_config": n_full_run,
            "no_duplicates": len(sm) == len(km) and len(sr) == len(kr),
            "sets_identical": sm == sr,
            "order_identical": km == kr,
            "identical_to_split_x_select_angles": sm == indep,
            "hdf5_rows_match": _hdf5_rows_match(mine),
            "fingerprint": keys_fingerprint(km),
            "n_locations": len({(s, l) for s, l, _ in km}),
        }
        r["ok"] = bool(r["sets_identical"] and r["no_duplicates"]
                       and r["identical_to_split_x_select_angles"] and r["hdf5_rows_match"]
                       and r["n_rgbonly"] == r["n_expected"]
                       and (n_full_run is None or n_full_run == r["n_rgbonly"]))
        report["conditions"][cid] = r
        report["ok"] &= r["ok"]
        mine.close(); ref.close()

    # Walidacja i test sa wspolne dla wszystkich warunkow (angle_seed nie gra roli
    # dla `all`/`cardinal`), wiec wystarczy je policzyc raz.
    cond = CONDITIONS_BY_ID[cond_ids[0]]
    counts = {"val": len(EchoH5Dataset(val_data_config(cond), splits=splits))}
    for label, subset in TEST_SETS.items():
        counts[label] = len(EchoH5Dataset(
            DatasetConfig(variant=cond.geometry, mode="test", angle_subset=subset,
                          angle_seed=cond.angle_seed, augment=False), splits=splits))
    report["eval_counts"] = counts
    report["eval_counts_ok"] = all(counts[k] == EXPECTED_N[k] for k in counts)
    report["ok"] &= report["eval_counts_ok"]
    return report


def print_check(rep: dict) -> None:
    print(f"ROWNOSC PAR (scena, lokalizacja, kat): RGB-only vs train_condition.py   "
          f"[odcisk podzialu {rep['split_fingerprint']}]")
    for cid, r in rep["conditions"].items():
        print(f"  {cid} ({r['angle_subset']}, seed {r['angle_seed']}): n={r['n_rgbonly']} "
              f"(train_condition {r['n_train_condition']}, oczekiwane {r['n_expected']}, "
              f"przebieg pelny {r['n_from_full_model_run_config']})  "
              f"zbiory identyczne={r['sets_identical']}  kolejnosc={r['order_identical']}  "
              f"=podzial x select_angles: {r['identical_to_split_x_select_angles']}  "
              f"HDF5 ok={r['hdf5_rows_match']}  odcisk {r['fingerprint']}  -> "
              f"{'OK' if r['ok'] else 'BLAD'}")
    c = rep["eval_counts"]
    print(f"  walidacja {c['val']}, test@36 {c['test@36']}, test@4 {c['test@4']}  -> "
          f"{'OK' if rep['eval_counts_ok'] else 'BLAD'}")
    print(f"WYNIK: {'ZGODNE' if rep['ok'] else 'NIEZGODNE'}")


# ------------------------------------------------------------------ ewaluacja


@torch.no_grad()
def _collect_limited(model, cond, splits, *, angle_subset, device, batch_size, num_workers,
                     amp, edge_threshold, max_batches):
    """Tylko dla --dry-run: to samo co `evaluate.collect_table` (bez maski i
    permutacji), ale przerwane po `max_batches` wsadach."""
    ds = EchoH5Dataset(DatasetConfig(variant=cond.geometry, mode="test",
                                     angle_subset=angle_subset, angle_seed=cond.angle_seed,
                                     augment=False), splits=splits)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)
    col = SampleStatsCollector(ds.scenes, edge_threshold)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(dev)
        col.update(out["depth_predicted"].float(), out["depth_gt"].float(),
                   scene_idx=dev["scene_idx"], location_id=dev["location_id"],
                   angle_deg=dev["angle_deg"], valid_ref=dev.get("valid_ref"))
    info = {"n_samples": len(ds), "shuffled_echo": False, "shuffle_fixed_points": 0,
            "dry_run_batches": max_batches}
    ds.close()
    return col.table(), info


def run_test_eval(model, net, cond, splits, run_dir: Path, args, amp: bool, device,
                  run_id: str, best_step: int) -> dict:
    """test@36 i test@4 z `best_rgbdepth.pth` -- format jak `evaluate.py`."""
    t0 = time.perf_counter()
    net.load_state_dict(torch.load(run_dir / "best_rgbdepth.pth", map_location=device,
                                   weights_only=True))
    model.eval()
    grid = training_angle_grid(cond, splits)
    payload: dict = {
        "run_id": run_id, "condition": cond.id, "model": "rgb_only",
        "control": "RGBDepthNet (RGB2Depth) bez echa, materialu i uwagi",
        "geometry": cond.geometry, "angle_subset": cond.angle_subset,
        "angle_seed": cond.angle_seed, "seed": args.seed,
        "weights": {"rgbdepth": "best_rgbdepth.pth"}, "best_step": best_step,
        "split_fingerprint": splits.meta.get("location_fingerprint"),
        "intersection_mask_variant": None, "mask_mode": None,
        "shuffled_echo": False, "shuffle_seed": None,
        "edge_threshold_m_per_px": args.edge_threshold,
        "dry_run": bool(args.dry_run),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_sets": {},
    }
    for label, subset in TEST_SETS.items():
        kw = dict(angle_subset=subset, device=device, batch_size=args.batch_size,
                  num_workers=args.num_workers, amp=amp, edge_threshold=args.edge_threshold)
        if args.dry_run:
            tab, info = _collect_limited(model, cond, splits, max_batches=DRY_TEST_BATCHES, **kw)
        else:
            tab, info = collect_table(model, cond, splits, mask_variant=None,
                                      shuffle_echo_seed=None, **kw)
        tab.save(run_dir / f"samples_{label}.npz")
        payload["test_sets"][label] = analyse(tab, train_grid=grid, label=label)
        payload["test_sets"][label]["collect_info"] = info
        o = payload["test_sets"][label]["overall"]
        print(f"  {label}: n={len(tab)}  RMSE {o['all']['RMSE']:.5f}  "
              f"krawedzie {o['edge']['RMSE']:.5f}  d1 {o['all']['DELTA1']:.4f}")
    payload["seconds"] = round(time.perf_counter() - t0, 1)
    (run_dir / "eval.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return payload


# ------------------------------------------------------------------- sterowanie


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", choices=RGBONLY_CONDITIONS)
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--validation-freq", type=int, default=1000)
    p.add_argument("--display-freq", type=int, default=100)
    p.add_argument("--edge-threshold", type=float, default=0.10)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--dry-run", action="store_true",
                   help=f"{DRY_STEPS} krokow, 1 walidacja na {DRY_VAL_SAMPLES} probkach, ewaluacja "
                        f"na {DRY_TEST_BATCHES} wsadach; wynik do <out-root>/dryrun/")
    p.add_argument("--check-pairs", action="store_true",
                   help="tylko porownanie par z train_condition.py dla A, D, B (bez GPU)")
    args = p.parse_args(argv)

    splits = load_splits()
    fp = splits.meta.get("location_fingerprint")

    if args.check_pairs:
        rep = check_pairs(list(RGBONLY_CONDITIONS), splits)
        print_check(rep)
        out = args.out_root / "check_pairs.json"
        if out.exists():
            print(f"(nie zapisano: {out} juz istnieje)")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            rep["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"zapisano: {out}")
        return 0 if rep["ok"] else 3

    if args.condition is None or args.seed is None:
        p.error("podaj --condition i --seed (albo --check-pairs)")

    cond = CONDITIONS_BY_ID[args.condition]
    spec = RunSpec(condition=cond.id, seed=args.seed, total_steps=args.steps,
                   batch_size=args.batch_size, num_workers=args.num_workers,
                   amp=not args.no_amp, validation_freq=args.validation_freq,
                   display_freq=args.display_freq, edge_threshold_m=args.edge_threshold,
                   tag="rgbonly")
    run_id = f"RGBONLY_{cond.id}_seed{args.seed}"
    if args.dry_run:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = args.out_root / "dryrun" / f"{run_id}_{stamp}"
        spec.total_steps = DRY_STEPS
        spec.display_freq = 10
        spec.validation_freq = DRY_STEPS
    else:
        run_dir = args.out_root / run_id
    if run_dir.exists():
        print(f"BLAD: {run_dir} juz istnieje -- nic nie nadpisuje.")
        return 2

    # Rownosc par sprawdzana PRZED treningiem, dla tego warunku. Tanie (same
    # metadane), a zly zbior to stracony przebieg.
    rep = check_pairs([cond.id], splits)
    print_check(rep)
    if not rep["ok"]:
        print("BLAD: zbior treningowy nie zgadza sie z train_condition.py -- przerwane.")
        return 3

    run_dir.mkdir(parents=True, exist_ok=False)
    amp = spec.amp
    device = torch.device("cuda")

    # Kolejnosc jak w train_condition.py: ziarno -> siec -> optymalizator -> dane.
    set_seed(args.seed)
    paths.add_parida_to_syspath()
    from models import criterion
    from models.models import ModelBuilder

    net = ModelBuilder().build_rgbdepth().to(device)
    model = RGBOnlyModel(net, MAX_DEPTH_REPLICA).to(device)
    # Jedna grupa parametrow = jedna siec; hiperparametry wprost z RunSpec,
    # tak jak `train_condition.build_optimizer` dla wariantu "adam".
    optimizer = torch.optim.Adam([{"params": net.parameters(), "lr": spec.lr}],
                                 betas=(spec.beta1, 0.999), weight_decay=spec.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = criterion.LogDepthLoss()

    train_cfg, val_cfg = train_data_config(cond), val_data_config(cond)

    def _val_part(ds):
        # Tylko --dry-run: walidacja "na czesci" -- pierwsze probki indeksu.
        n = DRY_VAL_SAMPLES
        ds.index_scene, ds.index_row = ds.index_scene[:n], ds.index_row[:n]
        ds.index_loc, ds.index_angle = ds.index_loc[:n], ds.index_angle[:n]

    train_loader, train_ds = build_dataloader(train_cfg, batch_size=spec.batch_size,
                                              num_workers=spec.num_workers, splits=splits)
    val_loader, val_ds = build_dataloader(val_cfg, batch_size=spec.batch_size,
                                          num_workers=max(2, spec.num_workers // 2),
                                          splits=splits, shuffle=False, drop_last=False,
                                          dataset_filter=_val_part if args.dry_run else None)
    print(f"{run_id}{'  [DRY-RUN]' if args.dry_run else ''}: train={len(train_ds)} "
          f"val={len(val_ds)} (walidacja '{VAL_ANGLE_SUBSET}')  wsadow/epoke={len(train_loader)}  "
          f"rownowaznik epok={spec.total_steps * spec.batch_size / len(train_ds):.1f}")
    print(f"  katalog: {run_dir}")

    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "control": "RGB-only (RGBDepthNet Paridy, bez echa/materialu/uwagi)",
        "condition": dataclasses.asdict(cond), "spec": vars(spec),
        "dry_run": bool(args.dry_run), "argv": sys.argv,
        "train_data": dataclasses.asdict(train_cfg), "val_data": dataclasses.asdict(val_cfg),
        "val_angle_subset": VAL_ANGLE_SUBSET, "max_depth": MAX_DEPTH_REPLICA,
        "loss": "LogDepthLoss (Parida), piksele depth_gt != 0",
        "optimizer": {"name": "adam", "lr": spec.lr, "betas": [spec.beta1, 0.999],
                      "weight_decay": spec.weight_decay},
        "split_fingerprint": fp,
        "train_pairs_fingerprint": rep["conditions"][cond.id]["fingerprint"],
        "pairs_check": rep,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "full_model_run": str(paths.RUNS_DIR / f"{cond.id}_seed{args.seed}"),
        "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    metrics_fp = run_dir / "metrics.jsonl"
    train_log = run_dir / "train_loss.csv"
    train_log.write_text("step,loss,epoch,samples_per_s\n", encoding="utf-8")

    stop = {"flag": False}

    def _sig(signum, frame):
        print(f"\nsygnal {signum}: koncze po biezacym kroku (bez ewaluacji testowej)")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    stream = InfiniteLoader(train_loader)
    model.train()
    step = 0
    best = {"val36": {"rmse": float("inf"), "step": 0},
            "val4": {"rmse": float("inf"), "step": 0}}
    running, t0 = [], time.perf_counter()
    t_train_start = time.perf_counter()
    t_timed, n_timed, t_val = None, 0, 0.0

    while step < spec.total_steps and not stop["flag"]:
        if step == DRY_WARMUP and args.dry_run:
            torch.cuda.synchronize()
            t_timed = time.perf_counter()
        batch = stream.next()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(batch)
            dp, dg = out["depth_predicted"], out["depth_gt"]
            m = dg != 0
            loss = loss_fn(dp[m], dg[m])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running.append(float(loss.detach()))
        step += 1

        if step % spec.display_freq == 0:
            dt = time.perf_counter() - t0
            sps = spec.display_freq * spec.batch_size / dt
            avg = float(np.mean(running))
            print(f"  krok {step:6d}/{spec.total_steps}  loss {avg:.5f}  "
                  f"epoka {stream.epoch}  {sps:.1f} probek/s")
            with train_log.open("a", encoding="utf-8") as f:
                f.write(f"{step},{avg:.6f},{stream.epoch},{sps:.1f}\n")
            running, t0 = [], time.perf_counter()

        if step % spec.validation_freq == 0 or step == spec.total_steps:
            if args.dry_run and t_timed is not None:
                torch.cuda.synchronize()
                n_timed = step - DRY_WARMUP
                t_timed = time.perf_counter() - t_timed
            tv = time.perf_counter()
            res = validate(model, val_loader, loss_fn, device, val_ds.scenes,
                           spec.edge_threshold_m, amp)
            t_val += time.perf_counter() - tv
            rec = {"step": step, "epoch": stream.epoch, "split": "val", **res}
            with metrics_fp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            o = res["all"]
            r4 = res.get("val4") or {}
            print(f"  [val {step}] loss {res['loss']:.4f}  RMSE@36 {o['RMSE']:.4f}  "
                  f"RMSE@4 {r4.get('all', {}).get('RMSE', float('nan')):.4f}  "
                  f"d1 {o['DELTA1']:.4f}")
            # Wybor checkpointu: najlepsze RMSE walidacyjne `all` (pelne 36 katow),
            # jak w train_condition.py. `val@4` tylko notowane (bez osobnych wag).
            if o["RMSE"] < best["val36"]["rmse"]:
                best["val36"] = {"rmse": o["RMSE"], "step": step}
                torch.save(net.state_dict(), run_dir / "best_rgbdepth.pth")
                (run_dir / "best.json").write_text(
                    json.dumps({**rec, "best_step": step}, indent=2, ensure_ascii=False,
                               default=str), encoding="utf-8")
                print(f"           -> nowy najlepszy val@36 RMSE {o['RMSE']:.5f}")
            if r4 and r4["all"]["RMSE"] < best["val4"]["rmse"]:
                best["val4"] = {"rmse": r4["all"]["RMSE"], "step": step}
            t0 = time.perf_counter()

    train_seconds = time.perf_counter() - t_train_start
    finished = step >= spec.total_steps
    status = {
        "run_id": run_id, "step": step, "total_steps": spec.total_steps,
        "finished": finished, "interrupted": stop["flag"], "dry_run": bool(args.dry_run),
        "val_angle_subset": VAL_ANGLE_SUBSET,
        "best_val_rmse": best["val36"]["rmse"], "best_step": best["val36"]["step"],
        "best_val4_rmse": best["val4"]["rmse"], "best_step_val4": best["val4"]["step"],
        "best_step_fraction_of_budget": (best["val36"]["step"] / spec.total_steps
                                         if spec.total_steps else 0.0),
        "epochs": stream.epoch, "train_seconds": round(train_seconds, 1),
        "validation_seconds": round(t_val, 1), "eval_done": False,
    }
    if args.dry_run and n_timed:
        status["timing"] = {"s_per_step": t_timed / n_timed, "timed_steps": n_timed,
                            "warmup_steps": DRY_WARMUP,
                            "val_seconds_per_sample": t_val / max(len(val_ds), 1)}

    train_ds.close()
    val_ds.close()

    if finished and best["val36"]["step"] > 0:
        print(f"EWALUACJA TESTOWA z best_rgbdepth.pth (krok {best['val36']['step']})")
        payload = run_test_eval(model, net, cond, splits, run_dir, args, amp, device,
                                run_id, best["val36"]["step"])
        status["eval_done"] = True
        status["eval_seconds"] = payload["seconds"]
        status["test36_rmse"] = payload["test_sets"]["test@36"]["overall"]["all"]["RMSE"]
        status["test4_rmse"] = payload["test_sets"]["test@4"]["overall"]["all"]["RMSE"]

    status["ended_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"\nKONIEC: krok {step}/{spec.total_steps}, najlepszy val RMSE "
          f"{best['val36']['rmse']:.5f} na kroku {best['val36']['step']}"
          f"{'' if status['eval_done'] else '  (BEZ ewaluacji testowej)'}")
    if "timing" in status:
        t = status["timing"]
        print(f"  pomiar: {t['s_per_step']:.4f} s/krok (kroki {DRY_WARMUP + 1}-{step}), "
              f"walidacja {t['val_seconds_per_sample'] * 1000:.2f} ms/probke")
    return 0 if finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
