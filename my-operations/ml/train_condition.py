#!/usr/bin/env python
"""Uruchamia JEDEN warunek przy JEDNYM ziarnie.

    python my-operations/ml/train_condition.py --condition A --seed 0
    python my-operations/ml/train_condition.py --condition B --seed 1 --resume
    python my-operations/ml/train_condition.py --condition A --seed 0 --dry-run

Jedna jednostka pracy na wywolanie, tak jak `generate_echo_dataset.py` mial
jedna scene na proces. Powody sa te same:
  * przebieg trwa kilkanascie godzin, wiec musi dac sie wznowic bez utraty
    calosci i musi przezyc zerwane SSH (`echo_ctl.py`-owy `start_new_session`);
  * kazdy warunek ma wlasny, izolowany stan CUDA -- zaden wyciek pamieci ani
    zawieszony kontekst z warunku A nie moze skazic warunku B;
  * kolejnoscia i czasem uruchomien steruje czlowiek przez `exp_ctl.py`, a nie
    petla po macierzy.

CO SIE TU NIE ZMIENIA WOBEC PARIDY: architektura sieci, funkcja straty
(`LogDepthLoss`), optymalizator (Adam, lr 1e-4, wd 5e-4), maskowanie `depth_gt != 0`
i skala `max_depth`. Zmienna niezalezna to gestosc katowa; podmiana czegokolwiek
z powyzszych uniemozliwilaby przypisanie efektu.

CO SIE ZMIENIA I DLACZEGO:
  1. `build_audiodepth()` dostaje JAWNIE `audio_shape=[2,257,166]`. `train.py`
     Paridy wola je bez argumentu, czyli z domyslnym ksztaltem mp3d [2,257,121],
     i na Replice wywala sie na niezgodnosci kanalow w `conv1x1`
     (3808 vs 2464). To jest blad w oryginale, nie zmiana metody.
  2. Budzet to KROKI, nie epoki (patrz `experiments.py`).
  3. Walidacja liczy metryki stratyfikowane i per scena (patrz `metrics.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "ml"

from . import paths  # noqa: E402
from .echo_h5_dataset import MAX_DEPTH_REPLICA, DatasetConfig, build_dataloader  # noqa: E402
from .experiments import (  # noqa: E402
    BATCH_SIZE,
    CONDITIONS_BY_ID,
    MODEL_ECHO,
    TOTAL_STEPS,
    RunSpec,
)
from .metrics import Evaluator  # noqa: E402
from .splits import load_splits  # noqa: E402

AUDIO_SHAPE = (2, 257, 166)


class EchoOnlyModel(torch.nn.Module):
    """ECHO2DEPTH: sama galaz audio, bez obrazu, materialu i uwagi.

    Nie jest to nowa architektura -- to DOKLADNIE `net_audiodepth` Paridy
    (`SimpleAudioDepthNet`), wyjete z fuzji. Ta sama siec, ten sam mnoznik
    `max_depth`, ten sam ksztalt wyjscia; interfejs celowo identyczny z
    `AudioVisualModel`, zeby petla treningowa i ewaluacja byly wspolne i zeby
    zadna roznica wyniku nie mogla pochodzic z innego kodu wokol.
    """

    def __init__(self, net_audiodepth, max_depth: float):
        super().__init__()
        self.net_audiodepth = net_audiodepth
        self.max_depth = max_depth

    def forward(self, x):
        audio_depth, audio_feat = self.net_audiodepth(x["audio"])
        scaled = audio_depth * self.max_depth
        return {
            "audio_depth": scaled,
            "depth_predicted": scaled,   # w tym warunku predykcja = wyjscie audio
            "img_depth": None,
            "attention": None,
            "depth_gt": x["depth"],
        }


def set_seed(seed: int) -> None:
    """Ziarno wszystkiego, co losuje.

    Nie ustawiamy `torch.use_deterministic_algorithms(True)`: czesc jader cuDNN
    nie ma deterministycznych odpowiednikow, a wymuszenie ich spowolniloby
    przebieg. Cel ziarna to POROWNYWALNY ROZRZUT miedzy warunkami (kazdy warunek
    startuje z tej samej inicjalizacji dla danego ziarna), a nie bitowa
    odtwarzalnosc pojedynczego przebiegu.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(condition, device, splits, fast_bilinear: bool = False):
    paths.add_parida_to_syspath()
    from models.models import ModelBuilder

    b = ModelBuilder()
    # audio_shape JAWNIE -- patrz naglowek modulu, punkt 1.
    net_audiodepth = b.build_audiodepth(audio_shape=list(AUDIO_SHAPE)).to(device)

    if condition.model == MODEL_ECHO:
        model = EchoOnlyModel(net_audiodepth, MAX_DEPTH_REPLICA).to(device)
        nets = {"audiodepth": net_audiodepth}
        return model, nets

    from models.audioVisual_model import AudioVisualModel

    net_rgbdepth = b.build_rgbdepth().to(device)
    net_attention = b.build_attention().to(device)
    net_material = b.build_material_property().to(device)

    if fast_bilinear:
        # Podmiana `nn.Bilinear` na tozsamy matematycznie `BilinearEinsum`.
        # NIE jest to zmiana architektury: te same parametry, ta sama funkcja,
        # te same gradienty (roznica wzgledna ~1e-6, czyli szum float32) --
        # zmienia sie tylko kolejnosc operacji. Zmierzone: 1513 -> 78 ms/iter.
        # Domyslnie wylaczone; szczegoly w `fast_bilinear.py`.
        from .fast_bilinear import swap_bilinear
        n = swap_bilinear(net_attention)
        print(f"  fast-bilinear  : podmieniono {n} warstw nn.Bilinear (tozsame matematycznie)")

    class _Opt:
        max_depth = MAX_DEPTH_REPLICA

    model = AudioVisualModel(
        (net_rgbdepth, net_audiodepth, net_attention, net_material), _Opt()
    ).to(device)
    nets = {
        "rgbdepth": net_rgbdepth,
        "audiodepth": net_audiodepth,
        "attention": net_attention,
        "material": net_material,
    }
    return model, nets


def build_optimizer(nets: dict, spec: RunSpec):
    groups = [{"params": n.parameters(), "lr": spec.lr} for n in nets.values()]
    if spec.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=spec.beta1, weight_decay=spec.weight_decay)
    return torch.optim.Adam(groups, betas=(spec.beta1, 0.999), weight_decay=spec.weight_decay)


class InfiniteLoader:
    """Nieskonczony strumien wsadow z liczeniem epok.

    Budzet jest w krokach, a nie epokach, wiec petla treningowa nie moze byc
    petla po epokach. Licznik epok jest zachowany wylacznie do raportu -- w
    pracy trzeba podac, ile razy kazdy warunek obszedl swoj zbior, bo to
    rozne ryzyko przeuczenia.
    """

    def __init__(self, loader):
        self.loader = loader
        self.epoch = 0
        self._it = iter(loader)

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self.epoch += 1
            self._it = iter(self.loader)
            return next(self._it)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, scene_names, edge_threshold, amp):
    model.eval()
    ev = Evaluator(scene_names, edge_threshold)
    losses = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(batch)
        dp = out["depth_predicted"].float()
        dg = out["depth_gt"].float()
        m = dg != 0
        if bool(m.any()):
            losses.append(float(loss_fn(dp[m], dg[m])))
        ev.update(dp, dg, batch.get("scene_idx"))
    model.train()
    res = ev.result()
    res["loss"] = float(np.mean(losses)) if losses else float("nan")
    return res


def save_checkpoint(path: Path, nets, optimizer, scaler, step, best_rmse, epoch, spec):
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "step": step, "best_rmse": best_rmse, "epoch": epoch,
        "nets": {k: v.state_dict() for k, v in nets.items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "spec": vars(spec),
    }, tmp)
    tmp.rename(path)  # atomowo: przerwanie w trakcie zapisu nie zostawia polowki


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS_BY_ID))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--validation-freq", type=int, default=1000)
    p.add_argument("--display-freq", type=int, default=100)
    p.add_argument("--checkpoint-freq", type=int, default=2000)
    p.add_argument("--edge-threshold", type=float, default=0.10)
    p.add_argument("--fast-bilinear", action="store_true",
                   help="tozsamy matematycznie, ~19x szybszy zamiennik nn.Bilinear w attentionNet "
                        "(16.8 h -> 0.86 h na przebieg); patrz fast_bilinear.py")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="nadpisz istniejacy katalog przebiegu")
    p.add_argument("--dry-run", action="store_true",
                   help="pokaz konfiguracje i licznosci, NIE trenuj (nie dotyka GPU)")
    args = p.parse_args(argv)

    cond = CONDITIONS_BY_ID[args.condition]
    spec = RunSpec(
        condition=cond.id, seed=args.seed, total_steps=args.steps,
        batch_size=args.batch_size, num_workers=args.num_workers,
        amp=not args.no_amp, validation_freq=args.validation_freq,
        display_freq=args.display_freq, edge_threshold_m=args.edge_threshold,
        extra={"fast_bilinear": args.fast_bilinear},
    )
    run_dir = spec.run_dir()
    splits = load_splits(variant=cond.geometry)

    n_train = cond.n_train_samples(splits)
    print("=" * 74)
    print(f"WARUNEK {cond.id}  ziarno {args.seed}   [{cond.group}]")
    print("=" * 74)
    print(f"  izoluje        : {cond.isolates}")
    print(f"  angle_subset   : {cond.angle_subset}")
    print(f"  geometria      : {cond.geometry}")
    print(f"  model          : {cond.model}")
    print(f"  probek train   : {n_train}")
    print(f"  krokow         : {spec.total_steps} (batch {spec.batch_size})")
    print(f"  rownowaznik ep.: {cond.epochs_equivalent(splits, spec.total_steps, spec.batch_size):.1f}")
    print(f"  AMP            : {spec.amp}")
    print(f"  katalog        : {run_dir}")
    print(f"  odcisk podzialu: {splits.meta.get('location_fingerprint')}")

    if args.dry_run:
        print("\n--dry-run: nic nie uruchomiono.")
        return 0

    ckpt_path = run_dir / "checkpoint.pt"
    if run_dir.exists() and not (args.resume or args.force):
        print(f"\nBLAD: {run_dir} juz istnieje. Uzyj --resume albo --force.")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda")
    model, nets = build_model(cond, device, splits, fast_bilinear=args.fast_bilinear)
    optimizer = build_optimizer(nets, spec)
    scaler = torch.amp.GradScaler("cuda", enabled=spec.amp)

    paths.add_parida_to_syspath()
    from models import criterion
    loss_fn = criterion.LogDepthLoss()

    train_loader, train_ds = build_dataloader(
        DatasetConfig(variant=cond.geometry, mode="train",
                      angle_subset=cond.angle_subset, angle_seed=cond.angle_seed),
        batch_size=spec.batch_size, num_workers=spec.num_workers, splits=splits,
    )
    val_loader, val_ds = build_dataloader(
        DatasetConfig(variant=cond.geometry, mode="val",
                      angle_subset=cond.angle_subset, angle_seed=cond.angle_seed,
                      augment=False),
        batch_size=spec.batch_size, num_workers=max(2, spec.num_workers // 2),
        splits=splits, shuffle=False, drop_last=False,
    )
    print(f"  train={len(train_ds)} val={len(val_ds)} wsadow/epoke={len(train_loader)}")

    step, best_rmse, start_epoch = 0, float("inf"), 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        for k, n in nets.items():
            n.load_state_dict(ck["nets"][k])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best_rmse, start_epoch = ck["step"], ck["best_rmse"], ck["epoch"]
        print(f"  WZNOWIONO od kroku {step} (best RMSE {best_rmse:.5f})")

    (run_dir / "config.json").write_text(json.dumps({
        "condition": vars(cond) if hasattr(cond, "__dict__") else cond.__dict__,
        "spec": vars(spec),
        "split_fingerprint": splits.meta.get("location_fingerprint"),
        "n_train": len(train_ds), "n_val": len(val_ds),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    metrics_fp = run_dir / "metrics.jsonl"
    train_log = run_dir / "train_loss.csv"
    if not train_log.exists():
        train_log.write_text("step,loss,epoch,samples_per_s\n", encoding="utf-8")

    stop = {"flag": False}

    def _sig(signum, frame):
        # Przerwanie ma zapisac checkpoint, a nie zgubic kilkanascie godzin.
        print(f"\nsygnal {signum}: konczę po biezacym kroku i zapisuje checkpoint")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    stream = InfiniteLoader(train_loader)
    stream.epoch = start_epoch
    model.train()
    running, t0 = [], time.perf_counter()

    while step < spec.total_steps and not stop["flag"]:
        batch = stream.next()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=spec.amp):
            out = model(batch)
            dp, dg = out["depth_predicted"], out["depth_gt"]
            m = dg != 0
            loss = loss_fn(dp[m], dg[m])
        optimizer.zero_grad(set_to_none=True)
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
            res = evaluate(model, val_loader, loss_fn, device, val_ds.scenes,
                           spec.edge_threshold_m, spec.amp)
            rec = {"step": step, "epoch": stream.epoch, "split": "val", **res}
            with metrics_fp.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            o, e, s = res["overall"], res["edge"], res["smooth"]
            print(f"  [val {step}] loss {res['loss']:.4f}  RMSE {o['RMSE']:.4f}  "
                  f"krawedzie {e['RMSE']:.4f}  gladkie {s['RMSE']:.4f}  "
                  f"d1 {o['DELTA1']:.4f}  (krawedzi {res['edge_pixel_fraction']*100:.1f}% px)")

            # Wybor checkpointu po RMSE walidacyjnym, nie po ostatnim kroku:
            # warunki roznia sie rownowaznikiem epok, wiec roznie sie przeuczaja.
            if o["RMSE"] < best_rmse:
                best_rmse = o["RMSE"]
                for k, n in nets.items():
                    torch.save(n.state_dict(), run_dir / f"best_{k}.pth")
                (run_dir / "best.json").write_text(json.dumps(rec, indent=2,
                                                             ensure_ascii=False, default=str),
                                                   encoding="utf-8")
                print(f"           -> nowy najlepszy RMSE {best_rmse:.5f}, zapisano wagi")
            t0 = time.perf_counter()

        if step % args.checkpoint_freq == 0:
            save_checkpoint(ckpt_path, nets, optimizer, scaler, step, best_rmse,
                            stream.epoch, spec)

    save_checkpoint(ckpt_path, nets, optimizer, scaler, step, best_rmse, stream.epoch, spec)
    (run_dir / "status.json").write_text(json.dumps({
        "step": step, "total_steps": spec.total_steps,
        "finished": step >= spec.total_steps, "interrupted": stop["flag"],
        "best_val_rmse": best_rmse, "epochs": stream.epoch,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    train_ds.close()
    val_ds.close()
    print(f"\nKONIEC: krok {step}/{spec.total_steps}, najlepszy val RMSE {best_rmse:.5f}")
    return 0 if step >= spec.total_steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
