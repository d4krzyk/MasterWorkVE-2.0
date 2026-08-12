#!/usr/bin/env python
"""MODEL 2, etap 1: pretrening zadania orientacyjnego.

    python my-operations/ml/pretext_model/train_pretext.py --k 4  --seed 0
    python my-operations/ml/pretext_model/train_pretext.py --k 36 --seed 0
    python my-operations/ml/pretext_model/train_pretext.py --k 36 --pairs-per-location 16 --seed 0
    python my-operations/ml/pretext_model/train_pretext.py --k 4 --steps 40 --smoke   # przebieg dymny

KONTROLA PRZY ROWNEJ LICZBIE PAR (`--pairs-per-location 16`). Ta sama logika, co
warunek D w Modelu 1: pretrening K=36 ma 81x wiecej par niz K=4, wiec porownanie
K=4 vs K=36 mieszaloby rozdzielczosc katowa zadania z rozmiarem zbioru. Wariant
K=36 podprobkowany do 16 par na lokalizacje rozdziela to:

    K36 - K36@16par   izoluje ILOSC DANYCH
    K36@16par - K4    izoluje SAMA ROZDZIELCZOSC KATOWA ZADANIA

BUDZET W KROKACH, NIE EPOKACH -- z dokladnie tego samego powodu, co w Modelu 1
(`experiments.py`): przy stalej liczbie epok wariant K=36 dostalby 81x wiecej
krokow optymalizacji i wygralby z tego powodu, a nie z powodu rozdzielczosci.
"""

from __future__ import annotations

import argparse
import json
import random
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
from ..dataset.splits import load_splits  # noqa: E402
from . import K_VARIANTS, PAIRS_PER_LOCATION_CONTROL  # noqa: E402
from .metrics import PretextEvaluator  # noqa: E402
from .model import OrientationPretextNet  # noqa: E402
from .pairs import PairConfig, build_pair_loader  # noqa: E402

AUDIO_SHAPE = (2, 257, 166)
# Ten sam budzet, co Model 1 -- zeby "krok" znaczyl to samo w obu modelach.
DEFAULT_STEPS = 40_000
DEFAULT_BATCH = 32


def run_id(k: int, seed: int, pairs_per_location: int | None) -> str:
    tag = f"K{k}" + (f"_p{pairs_per_location}" if pairs_per_location else "")
    return f"pretext_{tag}_seed{seed}"


def runs_dir() -> Path:
    return paths.ML_OUTPUTS / "pretext"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, k, amp, loss_fn) -> dict:
    model.eval()
    ev = PretextEvaluator(k)
    losses = []
    for batch in loader:
        b = {kk: v.to(device, non_blocking=True) for kk, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(b)
            losses.append(float(loss_fn(logits.float(), b["label"])))
        ev.update(logits, b["label"])
    model.train()
    r = ev.result()
    r["loss"] = float(np.mean(losses)) if losses else float("nan")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, required=True, choices=sorted(set(K_VARIANTS) | {2, 6, 9, 18}))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pairs-per-location", type=int, default=None,
                    help=f"kontrola 4.4: stala liczba par na lokalizacje "
                         f"(dla K=36 uzyj {PAIRS_PER_LOCATION_CONTROL})")
    ap.add_argument("--pair-seed", type=int, default=0)
    ap.add_argument("--val-pairs-per-location", type=int, default=16,
                    help="ile par walidacyjnych na lokalizacje (0 = wszystkie K^2). "
                         "Przy K=36 wszystkie to 237 168 par na KAZDA walidacje, czyli "
                         "wielokrotnie drozej niz sam trening miedzy walidacjami. Stala "
                         "wartosc dla wszystkich K ma tez te zalete, ze zbior walidacyjny "
                         "ma to samo n niezaleznie od wariantu")
    ap.add_argument("--variant", default="main", choices=("main", "patched"))
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--validation-freq", type=int, default=1000)
    ap.add_argument("--display-freq", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="krotki przebieg sprawdzajacy kod")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rid = run_id(args.k, args.seed, args.pairs_per_location)
    out_dir = runs_dir() / rid
    splits = load_splits(variant=args.variant)
    amp = not args.no_amp

    tr_cfg = PairConfig(variant=args.variant, mode="train", n_classes=args.k,
                        pairs_per_location=args.pairs_per_location, pair_seed=args.pair_seed)
    val_ppl = (None if args.val_pairs_per_location <= 0
               else min(args.val_pairs_per_location, args.k * args.k))
    va_cfg = PairConfig(variant=args.variant, mode="val", n_classes=args.k,
                        pairs_per_location=val_ppl, pair_seed=args.pair_seed,
                        augment=False)

    print("=" * 74)
    print(f"MODEL 2 -- pretrening orientacji   K={args.k}  ziarno {args.seed}")
    print("=" * 74)

    train_loader, train_ds = build_pair_loader(
        tr_cfg, batch_size=args.batch_size, num_workers=args.num_workers, splits=splits)
    val_loader, val_ds = build_pair_loader(
        va_cfg, batch_size=args.batch_size, num_workers=max(2, args.num_workers // 2),
        splits=splits, shuffle=False, drop_last=False)

    s = train_ds.summary()
    for kk, vv in s.items():
        print(f"  {kk:22s}: {vv}")
    print(f"  {'par walidacyjnych':22s}: {len(val_ds)}")
    print(f"  {'rownowaznik epok':22s}: "
          f"{args.steps * args.batch_size / max(len(train_ds), 1):.2f}")
    print(f"  {'odcisk podzialu':22s}: {splits.meta.get('location_fingerprint')}")

    if args.dry_run:
        train_ds.close(); val_ds.close()
        print("\n--dry-run: nic nie uruchomiono.")
        return 0
    if out_dir.exists() and not args.force:
        print(f"\nBLAD: {out_dir} juz istnieje. Uzyj --force.")
        train_ds.close(); val_ds.close()
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda")
    model = OrientationPretextNet(args.k, audio_shape=AUDIO_SHAPE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss()   # PLASKA cross-entropy -- patrz 4.6

    (out_dir / "config.json").write_text(json.dumps({
        "run_id": rid, "K": args.k, "seed": args.seed, "variant": args.variant,
        "pairs_per_location": args.pairs_per_location, "pair_seed": args.pair_seed,
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "amp": amp,
        "loss": "CrossEntropyLoss (plaska, wersja podstawowa -- porownywalna z Gao)",
        "train": s, "n_val_pairs": len(val_ds),
        "split_fingerprint": splits.meta.get("location_fingerprint"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    metrics_fp = out_dir / "metrics.jsonl"
    best = {"maae": float("inf"), "step": 0}
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
        b = {kk: v.to(device, non_blocking=True) for kk, v in batch.items()}

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(b)
            loss = loss_fn(logits, b["label"])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running.append(float(loss.detach()))
        step += 1

        if step % args.display_freq == 0:
            dt = time.perf_counter() - t0
            print(f"  krok {step:6d}/{args.steps}  loss {np.mean(running):.5f}  "
                  f"epoka {epoch}  {args.display_freq * args.batch_size / dt:.1f} par/s")
            running, t0 = [], time.perf_counter()

        if step % args.validation_freq == 0 or step == args.steps:
            r = evaluate(model, val_loader, device, args.k, amp, loss_fn)
            rec = {"step": step, "epoch": epoch, "split": "val", **r}
            with metrics_fp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fine = r["by_true_shift"][f"fine_le_20deg"]
            print(f"  [val {step}] loss {r['loss']:.4f}  MAAE {r['MAAE_deg']:.2f} st. "
                  f"(losowo 90)  top1 {r['top1']*100:.1f} % (losowo {r['top1_chance']*100:.1f} %)  "
                  f"+/-30 st. {r['acc_within_30deg']*100:.1f} %  "
                  f"MAAE drobne {fine['MAAE_deg']:.2f}")
            if r["MAAE_deg"] < best["maae"]:
                best = {"maae": r["MAAE_deg"], "step": step, **{k2: v2 for k2, v2 in r.items()
                                                                if k2 != "confusion_matrix"}}
                torch.save(model.encoder_state_dict(), out_dir / "best_encoder.pth")
                torch.save(model.state_dict(), out_dir / "best_full.pth")
                (out_dir / "best.json").write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                print(f"           -> nowe najlepsze MAAE {r['MAAE_deg']:.3f} st., zapisano enkoder")
            t0 = time.perf_counter()

    (out_dir / "status.json").write_text(json.dumps({
        "run_id": rid, "step": step, "total_steps": args.steps,
        "finished": step >= args.steps, "interrupted": stop["flag"],
        "best_maae_deg": best["maae"], "best_step": best["step"],
        "best_step_fraction_of_budget": best["step"] / args.steps if args.steps else 0.0,
        "epochs": epoch, "ended_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    train_ds.close()
    val_ds.close()
    print(f"\nKONIEC: krok {step}/{args.steps}, najlepsze MAAE {best['maae']:.3f} st. "
          f"na kroku {best['step']}")
    print(f"  enkoder do przeniesienia: {out_dir / 'best_encoder.pth'}")
    return 0 if step >= args.steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
