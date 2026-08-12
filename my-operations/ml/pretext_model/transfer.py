#!/usr/bin/env python
"""MODEL 2, etap 2: zadanie docelowe RGB2Depth BEZ AUDIO W CZASIE TESTU.

    python my-operations/ml/pretext_model/transfer.py --init scratch --seed 0
    python my-operations/ml/pretext_model/transfer.py --init outputs/ml/pretext_model/pretext_K36_seed0/best_encoder.pth --seed 0

To jest miejsce, w ktorym powstaje LICZBA DO PRACY. Warunki do zestawienia,
wszystkie z wlasnych renderow:

    inicjalizacja enkodera        odniesienie u Gao (Replica)
    Scratch                       0,360
    pretrening K=4                0,332
    pretrening K=12               -- (nasze)
    pretrening K=36               -- (nasze)
    pretrening K=36 @ 16 par/lok. -- (nasze, kontrola)

Kolumna "odniesienie" NIE JEST baseline'em do przepisania do wlasnej tabeli --
silnik akustyczny jest inny (RLRAudioPropagation wobec silnika Gao), sceny
przetworzone inaczej, a `geometry_check.py` pokazal, ze same warianty geometrii
zmieniaja energie pozna o 46 %. Sluzy wylacznie do sprawdzenia, czy odtwarzamy
wlasciwy PORZADEK warunkow i rzad wielkosci efektu.

Siec zadania docelowego to `RGBDepthNet` Paridy, strata `LogDepthLoss`,
maskowanie `depth_gt != 0`, skala `max_depth` -- czyli dokladnie ten sam uklad,
co w Modelu 1, tylko bez galezi audio, materialu i uwagi. Rozni sie WYLACZNIE
inicjalizacja enkodera, wiec kazda roznica wyniku jest przypisywalna
pretreningowi.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.pretext_model"

from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import MAX_DEPTH_REPLICA, DatasetConfig, build_dataloader  # noqa: E402
from ..depth_model.metrics import SampleStatsCollector  # noqa: E402
from ..dataset.splits import load_splits  # noqa: E402
from .model import load_pretrained_encoder  # noqa: E402
from .train_pretext import set_seed  # noqa: E402

DEFAULT_STEPS = 40_000
DEFAULT_BATCH = 32


class RGBOnlyModel(torch.nn.Module):
    """RGB2Depth bez audio -- sam `RGBDepthNet` Paridy ze skala `max_depth`.

    Interfejs celowo identyczny z `AudioVisualModel` i `EchoOnlyModel`, zeby
    petla treningowa, walidacja i metryki byly wspolne z Modelem 1 i zadna
    roznica wyniku nie mogla pochodzic z innego kodu wokol sieci.
    """

    def __init__(self, net_rgbdepth, max_depth: float):
        super().__init__()
        self.net_rgbdepth = net_rgbdepth
        self.max_depth = max_depth

    def forward(self, x):
        img_depth, _ = self.net_rgbdepth(x["img"])
        scaled = img_depth * self.max_depth
        return {"img_depth": scaled, "depth_predicted": scaled,
                "audio_depth": None, "attention": None, "depth_gt": x["depth"]}


def runs_dir() -> Path:
    return paths.ML_OUTPUTS / "pretext_transfer"


@torch.no_grad()
def evaluate(model, loader, device, amp, loss_fn, scene_names, edge_threshold) -> dict:
    model.eval()
    col = SampleStatsCollector(scene_names, edge_threshold)
    losses = []
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(b)
        dp, dg = out["depth_predicted"].float(), out["depth_gt"].float()
        m = dg != 0
        if bool(m.any()):
            losses.append(float(loss_fn(dp[m], dg[m])))
        col.update(dp, dg, scene_idx=b["scene_idx"], location_id=b["location_id"],
                   angle_deg=b["angle_deg"])
    model.train()
    tab = col.table()
    res = tab.aggregate()
    res["loss"] = float(np.mean(losses)) if losses else float("nan")
    res["per_scene"] = {n: tab.aggregate(np.flatnonzero(tab.cols["scene_idx"] == i))
                        for i, n in enumerate(scene_names)
                        if (tab.cols["scene_idx"] == i).any()}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True,
                    help="'scratch' albo sciezka do best_encoder.pth z pretreningu")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", default=None, help="nazwa warunku w raporcie (domyslnie z --init)")
    ap.add_argument("--variant", default="main", choices=("main", "patched"))
    ap.add_argument("--angle-subset", default="all",
                    help="podzbior katow zadania DOCELOWEGO (domyslnie pelne 36)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--validation-freq", type=int, default=1000)
    ap.add_argument("--display-freq", type=int, default=100)
    ap.add_argument("--edge-threshold", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    scratch = args.init.lower() == "scratch"
    label = args.label or ("scratch" if scratch else Path(args.init).parent.name)
    rid = f"transfer_{label}_seed{args.seed}"
    out_dir = runs_dir() / rid
    splits = load_splits(variant=args.variant)
    amp = not args.no_amp
    device = torch.device("cuda")

    print("=" * 74)
    print(f"MODEL 2 -- zadanie docelowe RGB2Depth (bez audio)   {label}  ziarno {args.seed}")
    print("=" * 74)

    set_seed(args.seed)
    paths.add_parida_to_syspath()
    from models.models import ModelBuilder
    from models import criterion

    net = ModelBuilder().build_rgbdepth()
    transfer_report = {"init": "scratch"}
    if not scratch:
        transfer_report = load_pretrained_encoder(net, args.init)
        print(f"  przeniesiono {transfer_report['n_loaded']} kluczy enkodera "
              f"z {transfer_report['n_encoder_keys_in_file']} w pliku "
              f"(niezgodnych ksztaltow: {transfer_report['n_shape_mismatch']})")
        if not transfer_report["ok"]:
            print("  BLAD: przeniesienie nie doszlo do skutku -- przerwane, "
                  "bo wynik bylby nieodrozninalny od 'scratch'.")
            return 2
    else:
        print("  enkoder losowy (Scratch) -- warunek odniesienia")

    model = RGBOnlyModel(net, MAX_DEPTH_REPLICA).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = criterion.LogDepthLoss()

    train_loader, train_ds = build_dataloader(
        DatasetConfig(variant=args.variant, mode="train", angle_subset=args.angle_subset),
        batch_size=args.batch_size, num_workers=args.num_workers, splits=splits)
    val_loader, val_ds = build_dataloader(
        DatasetConfig(variant=args.variant, mode="val", angle_subset=args.angle_subset,
                      augment=False),
        batch_size=args.batch_size, num_workers=max(2, args.num_workers // 2),
        splits=splits, shuffle=False, drop_last=False)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  "
          f"rownowaznik epok={args.steps * args.batch_size / len(train_ds):.1f}")

    if args.dry_run:
        train_ds.close(); val_ds.close()
        print("\n--dry-run: nic nie uruchomiono.")
        return 0
    if out_dir.exists() and not args.force:
        print(f"\nBLAD: {out_dir} juz istnieje. Uzyj --force.")
        train_ds.close(); val_ds.close()
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps({
        "run_id": rid, "label": label, "seed": args.seed, "init": args.init,
        "transfer": transfer_report, "variant": args.variant,
        "angle_subset": args.angle_subset, "steps": args.steps,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "amp": amp, "n_train": len(train_ds), "n_val": len(val_ds),
        "split_fingerprint": splits.meta.get("location_fingerprint"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    metrics_fp = out_dir / "metrics.jsonl"
    best = {"rmse": float("inf"), "step": 0}
    step, epoch = 0, 0
    it = iter(train_loader)
    running, t0 = [], time.perf_counter()
    model.train()

    while step < args.steps and not stop["flag"]:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            it = iter(train_loader)
            batch = next(it)
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(b)
            dp, dg = out["depth_predicted"], out["depth_gt"]
            m = dg != 0
            loss = loss_fn(dp[m], dg[m])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running.append(float(loss.detach()))
        step += 1

        if step % args.display_freq == 0:
            dt = time.perf_counter() - t0
            print(f"  krok {step:6d}/{args.steps}  loss {np.mean(running):.5f}  "
                  f"epoka {epoch}  {args.display_freq * args.batch_size / dt:.1f} probek/s")
            running, t0 = [], time.perf_counter()

        if step % args.validation_freq == 0 or step == args.steps:
            r = evaluate(model, val_loader, device, amp, loss_fn, val_ds.scenes,
                         args.edge_threshold)
            rec = {"step": step, "epoch": epoch, "split": "val", **r}
            with metrics_fp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(f"  [val {step}] loss {r['loss']:.4f}  RMSE {r['all']['RMSE']:.5f}  "
                  f"krawedzie {r['edge']['RMSE']:.5f}  d1 {r['all']['DELTA1']:.4f}")
            if r["all"]["RMSE"] < best["rmse"]:
                best = {"rmse": r["all"]["RMSE"], "step": step}
                torch.save(net.state_dict(), out_dir / "best_rgbdepth.pth")
                (out_dir / "best.json").write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                print(f"           -> nowy najlepszy RMSE {best['rmse']:.5f}")
            t0 = time.perf_counter()

    frac = best["step"] / args.steps if args.steps else 0.0
    (out_dir / "status.json").write_text(json.dumps({
        "run_id": rid, "label": label, "step": step, "total_steps": args.steps,
        "finished": step >= args.steps, "interrupted": stop["flag"],
        "best_val_rmse": best["rmse"], "best_step": best["step"],
        "best_step_fraction_of_budget": frac,
        "budget_ceiling_warning": bool(frac >= 0.9 and step >= args.steps),
        "epochs": epoch, "ended_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    train_ds.close()
    val_ds.close()
    print(f"\nKONIEC: krok {step}/{args.steps}, najlepszy val RMSE {best['rmse']:.5f} "
          f"na kroku {best['step']}")
    return 0 if step >= args.steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
