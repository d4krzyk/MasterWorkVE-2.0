#!/usr/bin/env python
"""BLOK 1.1: czy `--fast-bilinear` dokłada cokolwiek ponad wlasny szum frameworka.

    python my-operations/ml/determinism_check.py --steps 2000
    python my-operations/ml/determinism_check.py --steps 200 --smoke   # sprawdzenie kodu

PROBLEM Z "JEDNYM PRZEBIEGIEM KONTROLNYM". Raport 2026-08-05 §7 sugerowal
porownanie jednego przebiegu `nn.Bilinear` z jednym przebiegiem `BilinearEinsum`.
To nie rozstrzyga niczego: z n=1 vs n=1 nie da sie orzec o NIEROZROZNIALNOSCI,
bo nie wiadomo, ile te dwa przebiegi rozjechalyby sie i BEZ podmiany. Trening
na GPU nie jest deterministyczny -- atomiki cuDNN, redukcje w innej kolejnosci
przy roznym rozkladzie blokow, AMP -- wiec dwa uruchomienia tego samego kodu z
tym samym ziarnem TEZ sie rozjezdzaja.

ROZWIAZANIE: ta sama logika, co kontrola negatywna przy `hfov=70` w fazie
generowania -- zmierzyc podstawienie PRZECIWKO WLASNEJ PODLODZE SZUMU:

    para 1   nn.Bilinear    vs nn.Bilinear      -> podloga (szum frameworka)
    para 2   BilinearEinsum vs BilinearEinsum   -> podloga wariantu szybkiego
    para 3   nn.Bilinear    vs BilinearEinsum   -> efekt podstawienia

KRYTERIUM DECYZYJNE: jesli para 3 rozjezdza sie NIE BARDZIEJ niz para 1,
podstawienie nie wnosi nic ponad niedeterminizm, ktory i tak jest w kazdym
przebiegu -- i `--fast-bilinear` moze isc na domyslnie wlaczone.

Rozbieznosc mierzona jest w dwoch niezaleznych wielkosciach, na kilku krokach
posrednich (nie tylko na koncu -- interesuje nas, czy para 3 SLEDZI pare 1 przez
caly przebieg, czy odrywa sie w pewnym momencie):
  * wagi: ||w_a - w_b||_2 / ||w_a||_2, po wszystkich sieciach i osobno per siec
  * walidacyjne RMSE na STALYM podzbiorze zbioru walidacyjnego (porownanie
    sparowane -- ten sam podzbior we wszystkich przebiegach)

KONTROLA KOLEJNOSCI DANYCH. Zeby porownanie mialo sens, wszystkie przebiegi
musza zobaczyc te same probki w tej samej kolejnosci. Nie jest to zakladane:
skrypt liczy skrot ciagu (scene, location_id, angle_deg) po wszystkich krokach
i porownuje go miedzy przebiegami. Rozny skrot uniewaznia caly pomiar.

KOSZT. `nn.Bilinear` to 1,51 s/krok (raport §3.8), wiec 2 przebiegi po 2 000
krokow to ~100 min; wariant szybki dokłada ~5 min. Tego nie da sie skrocic bez
skrocenia przebiegu, bo pomiar POLEGA na tym, ze wolna wersja tez biegnie.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.checks"

from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import DatasetConfig, build_dataloader  # noqa: E402
from ..matrix.experiments import BATCH_SIZE, CONDITIONS_BY_ID, RunSpec  # noqa: E402
from ..depth_model.metrics import MetricAccumulator  # noqa: E402
from ..dataset.splits import load_splits  # noqa: E402
from ..depth_model.train_condition import build_model, build_optimizer, set_seed  # noqa: E402

# Kroki, na ktorych robimy zdjecie wag. Krok 0 jest kluczowy: obie wersje MUSZA
# startowac z bit-identycznych wag, inaczej caly pomiar mierzy inna inicjalizacje.
DEFAULT_CHECKPOINTS = (0, 100, 250, 500, 1000, 2000)


def _out_dir() -> Path:
    return paths.ML_OUTPUTS / "determinism"


def _snap_dir() -> Path:
    return _out_dir() / "_snapshots"


# ------------------------------------------------------------------ porownania


def state_diff(sd_a: dict, sd_b: dict) -> dict:
    """Rozbieznosc dwoch kompletow wag, globalnie i per siec."""
    per_net = {}
    tot_d2 = 0.0
    tot_r2 = 0.0
    tot_max = 0.0
    identical = True
    for net in sorted(sd_a):
        d2 = 0.0
        r2 = 0.0
        mx = 0.0
        for k in sd_a[net]:
            a = sd_a[net][k].to(torch.float64)
            b = sd_b[net][k].to(torch.float64)
            d = a - b
            d2 += float((d ** 2).sum())
            r2 += float((a ** 2).sum())
            mx = max(mx, float(d.abs().max()) if d.numel() else 0.0)
        per_net[net] = {
            "l2_diff": float(np.sqrt(d2)),
            "l2_ref": float(np.sqrt(r2)),
            "rel_l2": float(np.sqrt(d2 / r2)) if r2 > 0 else 0.0,
            "max_abs_diff": mx,
            "identical": d2 == 0.0,
        }
        identical = identical and d2 == 0.0
        tot_d2 += d2
        tot_r2 += r2
        tot_max = max(tot_max, mx)
    return {
        "rel_l2": float(np.sqrt(tot_d2 / tot_r2)) if tot_r2 > 0 else 0.0,
        "l2_diff": float(np.sqrt(tot_d2)),
        "l2_ref": float(np.sqrt(tot_r2)),
        "max_abs_diff": tot_max,
        "bit_identical": identical,
        "per_net": per_net,
    }


# -------------------------------------------------------------------- przebieg


@torch.no_grad()
def eval_subset(model, loader, device, amp) -> dict:
    """RMSE na STALYM podzbiorze walidacji.

    Podzbior, nie caly zbior: pelna walidacja pelnego modelu to ~200 wsadow,
    a przy `nn.Bilinear` kazdy z nich kosztuje. Porownanie jest SPAROWANE --
    wszystkie przebiegi licza na dokladnie tych samych probkach -- wiec do
    pomiaru ROZNICY miedzy przebiegami podzbior wystarcza; nie jest to liczba
    do raportowania jako jakosc modelu.
    """
    model.eval()
    acc = MetricAccumulator()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(batch)
        acc.update(out["depth_predicted"].float(), out["depth_gt"].float())
    model.train()
    r = acc.result()
    return {"RMSE": r["RMSE"], "RMSE_per_sample": r["RMSE_per_sample"],
            "n_samples": r["n_samples"]}


def build_everything(condition: str, seed: int, fast: bool, device, amp: bool,
                     batch_size: int, num_workers: int, val_batches: int):
    """Model + optymalizator + loadery. Wspolne dla treningu i odtwarzania."""
    cond = CONDITIONS_BY_ID[condition]
    spec = RunSpec(condition=condition, seed=seed, batch_size=batch_size,
                   num_workers=num_workers, amp=amp)
    splits = load_splits(variant=cond.geometry)

    set_seed(seed)
    model, nets = build_model(cond, device, splits, fast_bilinear=fast)
    optimizer = build_optimizer(nets, spec)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader, train_ds = build_dataloader(
        DatasetConfig(variant=cond.geometry, mode="train",
                      angle_subset=cond.angle_subset, angle_seed=cond.angle_seed),
        batch_size=batch_size, num_workers=num_workers, splits=splits, generator=gen)

    from ..dataset.echo_h5_dataset import EchoH5Dataset
    val_ds = EchoH5Dataset(
        DatasetConfig(variant=cond.geometry, mode="val", angle_subset=cond.angle_subset,
                      angle_seed=cond.angle_seed, augment=False), splits=splits)
    n_val_want = val_batches * batch_size
    stride = max(1, len(val_ds) // n_val_want)
    sub_idx = list(range(0, len(val_ds), stride))[:n_val_want]
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, sub_idx), batch_size=batch_size, shuffle=False,
        num_workers=max(2, num_workers // 2), pin_memory=True, persistent_workers=True)
    return model, nets, optimizer, scaler, train_loader, train_ds, val_loader, val_ds


def replay_reference(tag: str, *, fast: bool, steps: int, seed: int, condition: str,
                     checkpoints: tuple[int, ...], val_batches: int, amp: bool,
                     num_workers: int, batch_size: int, device: torch.device) -> dict:
    """Odtwarza `evals` i skrot kolejnosci danych przebiegu referencyjnego z
    ZAPISANYCH zdjec wag, bez powtarzania treningu.

    Zdjecia SA wagami tego przebiegu, wiec ewaluacja z nich wczytana daje
    dokladnie te liczby, ktore dalby oryginalny przebieg -- to nie jest
    przyblizenie, tylko pominiecie ponownego liczenia tego samego. Kolejnosc
    danych jest funkcja ziarna i konfiguracji loadera, wiec odtwarza sie
    przejsciem po loaderze bez kroku optymalizacji.

    Powod istnienia: przebieg `nn.Bilinear` trwa 51 minut, a przerwana sesja
    zostawia komplet zdjec na dysku. Powtarzanie go bylo by placeniem godziny
    GPU za liczby, ktore juz sa.
    """
    d = _snap_dir()
    missing = [cp for cp in checkpoints if not (d / f"{tag}_step{cp}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"{tag}: brak zdjec dla krokow {missing} -- nie ma czego odtwarzac")

    (model, nets, _opt, _scaler, train_loader, train_ds,
     val_loader, val_ds) = build_everything(condition, seed, fast, device, amp,
                                            batch_size, num_workers, val_batches)
    t0 = time.perf_counter()
    evals: dict[int, dict] = {}
    for cp in checkpoints:
        sd = torch.load(d / f"{tag}_step{cp}.pt", map_location="cpu", weights_only=False)
        for k, net in nets.items():
            net.load_state_dict({kk: vv.to(device) for kk, vv in sd[k].items()})
        evals[cp] = eval_subset(model, val_loader, device, amp)
        print(f"    [{tag}] odtworzony krok {cp:5d}  val RMSE {evals[cp]['RMSE']:.6f}")
        del sd

    order_hash = hashlib.sha256()
    it = iter(train_loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        order_hash.update(batch["scene_idx"].numpy().tobytes())
        order_hash.update(batch["location_id"].numpy().tobytes())
        order_hash.update(batch["angle_deg"].numpy().tobytes())

    train_ds.close()
    val_ds.close()
    del model, nets, val_loader
    torch.cuda.empty_cache()
    return {
        "tag": tag, "fast_bilinear": fast, "steps": steps, "seed": seed,
        "compared_against": None, "replayed_from_snapshots": True,
        "data_order_sha256": order_hash.hexdigest(),
        "final_loss_mean_last100": None,
        "evals": {str(k): v for k, v in evals.items()},
        "weight_diffs": {},
        "seconds": round(time.perf_counter() - t0, 1),
        "s_per_step": None,
    }


def run_once(tag: str, *, fast: bool, steps: int, seed: int, condition: str,
             checkpoints: tuple[int, ...], val_batches: int, amp: bool,
             num_workers: int, batch_size: int, save_snapshots: bool,
             device: torch.device, compare_against: str | None = None) -> dict:
    """Jeden przebieg treningowy o dlugosci `steps`, ze zdjeciami wag.

    Zdjecie wag pelnego modelu to 317 M parametrow, czyli 1,27 GB. Trzymanie
    czterech przebiegow x szesc punktow pomiarowych w RAM to 30 GB -- wiec
    zdjecia lecą NA DYSK i tylko dla dwoch przebiegow REFERENCYJNYCH
    (`save_snapshots`), a pozostale dwa porownuja sie z referencja OD RAZU na
    punkcie pomiarowym (`compare_against`) i zapamietuja juz tylko liczby.
    """
    cond = CONDITIONS_BY_ID[condition]
    spec = RunSpec(condition=condition, seed=seed, total_steps=steps,
                   batch_size=batch_size, num_workers=num_workers, amp=amp)
    splits = load_splits(variant=cond.geometry)

    set_seed(seed)
    model, nets = build_model(cond, device, splits, fast_bilinear=fast)
    optimizer = build_optimizer(nets, spec)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    paths.add_parida_to_syspath()
    from models import criterion
    loss_fn = criterion.LogDepthLoss()

    # Generator DataLoadera JAWNIE zaziarniony: kolejnosc probek nie moze
    # zalezec od tego, ile razy cokolwiek innego losowalo z globalnego RNG.
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader, train_ds = build_dataloader(
        DatasetConfig(variant=cond.geometry, mode="train",
                      angle_subset=cond.angle_subset, angle_seed=cond.angle_seed),
        batch_size=batch_size, num_workers=num_workers, splits=splits, generator=gen,
    )
    from ..dataset.echo_h5_dataset import EchoH5Dataset
    val_ds = EchoH5Dataset(
        DatasetConfig(variant=cond.geometry, mode="val",
                      angle_subset=cond.angle_subset, angle_seed=cond.angle_seed,
                      augment=False),
        splits=splits)
    # Staly, rownomiernie rozlozony podzbior walidacji -- co k-ta probka, zeby
    # objac wszystkie sceny i wszystkie orientacje, a nie pierwsza scene z listy.
    n_val_want = val_batches * batch_size
    stride = max(1, len(val_ds) // n_val_want)
    sub_idx = list(range(0, len(val_ds), stride))[:n_val_want]
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, sub_idx), batch_size=batch_size,
        shuffle=False, num_workers=max(2, num_workers // 2), pin_memory=True,
        persistent_workers=True)

    order_hash = hashlib.sha256()
    evals: dict[int, dict] = {}
    diffs: dict[int, dict] = {}
    losses: list[float] = []

    def snapshot(step: int) -> None:
        """Zdjecie wag: zapis referencji i/lub natychmiastowe porownanie."""
        if not (save_snapshots or compare_against):
            return
        sd = {k: {n: p.detach().to("cpu", torch.float32).clone()
                  for n, p in v.state_dict().items()} for k, v in nets.items()}
        d = _snap_dir()
        d.mkdir(parents=True, exist_ok=True)
        if save_snapshots:
            torch.save(sd, d / f"{tag}_step{step}.pt")
        if compare_against:
            ref = torch.load(d / f"{compare_against}_step{step}.pt",
                             map_location="cpu", weights_only=False)
            diffs[step] = state_diff(ref, sd)
            del ref
        del sd

    if 0 in checkpoints:
        snapshot(0)
        evals[0] = eval_subset(model, val_loader, device, amp)

    model.train()
    it = iter(train_loader)
    step = 0
    t0 = time.perf_counter()
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        # Skrot kolejnosci danych -- dowod, ze wszystkie przebiegi widza
        # dokladnie ten sam ciag probek. Bez tego rozbieznosc wag mogla by
        # pochodzic z innej kolejnosci, a nie z niedeterminizmu jader.
        order_hash.update(batch["scene_idx"].numpy().tobytes())
        order_hash.update(batch["location_id"].numpy().tobytes())
        order_hash.update(batch["angle_deg"].numpy().tobytes())

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(batch)
            dp, dg = out["depth_predicted"], out["depth_gt"]
            m = dg != 0
            loss = loss_fn(dp[m], dg[m])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        step += 1

        if step in checkpoints:
            snapshot(step)
            evals[step] = eval_subset(model, val_loader, device, amp)
            print(f"    [{tag}] krok {step:5d}  loss {np.mean(losses[-100:]):.5f}  "
                  f"val RMSE {evals[step]['RMSE']:.6f}  "
                  f"{step / (time.perf_counter() - t0):.2f} kr/s")

    train_ds.close()
    val_ds.close()
    del model, nets, optimizer, scaler, val_loader
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "fast_bilinear": fast,
        "steps": steps,
        "seed": seed,
        "compared_against": compare_against,
        "data_order_sha256": order_hash.hexdigest(),
        "final_loss_mean_last100": float(np.mean(losses[-100:])),
        "evals": {str(k): v for k, v in evals.items()},
        "weight_diffs": {str(k): v for k, v in diffs.items()},
        "seconds": round(time.perf_counter() - t0, 1),
        "s_per_step": round((time.perf_counter() - t0) / max(step, 1), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="A", choices=sorted(CONDITIONS_BY_ID))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-batches", type=int, default=64,
                    help="wsadow walidacji na punkt pomiarowy (staly podzbior)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="krotkie kroki kontrolne, do sprawdzenia samego kodu")
    ap.add_argument("--reuse-reference", action="store_true",
                    help="jesli zdjecia wag przebiegu referencyjnego juz leza na dysku "
                         "(przerwana sesja), odtworz z nich jego metryki zamiast trenowac "
                         "go po raz drugi -- 51 min oszczednosci na przebieg nn.Bilinear")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cps = tuple(c for c in DEFAULT_CHECKPOINTS if c <= args.steps)
    if args.steps not in cps:
        cps = cps + (args.steps,)
    if args.smoke:
        cps = tuple(c for c in (0, args.steps) if c <= args.steps)

    device = torch.device("cuda")
    amp = not args.no_amp
    out_path = args.out or (_out_dir() / "determinism_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _snap_dir().exists() and not args.reuse_reference:
        shutil.rmtree(_snap_dir())

    common = dict(steps=args.steps, seed=args.seed, condition=args.condition,
                  checkpoints=cps, val_batches=args.val_batches, amp=amp,
                  num_workers=args.num_workers, batch_size=args.batch_size,
                  device=device)

    print("=" * 74)
    print(f"BLOK 1.1  kontrola niedeterminizmu   warunek {args.condition}, "
          f"ziarno {args.seed}, {args.steps} krokow")
    print(f"          punkty pomiarowe: {cps}")
    print("=" * 74)

    # Kolejnosc przebiegow wynika z tego, kto jest czyja referencja:
    #   slow_1 -- referencja obu porownan z udzialem wersji wolnej (zapisuje wagi)
    #   slow_2 -- porownanie z slow_1  -> para 1 (podloga)
    #   fast_1 -- porownanie z slow_1  -> para 3 (podstawienie); tez zapisuje wagi
    #   fast_2 -- porownanie z fast_1  -> para 2 (podloga wersji szybkiej)
    plan = (
        ("slow_1", False, True, None),
        ("slow_2", False, False, "slow_1"),
        ("fast_1", True, True, "slow_1"),
        ("fast_2", True, False, "fast_1"),
    )
    runs: dict[str, dict] = {}
    partial = out_path.with_suffix(".partial.json")
    for tag, fast, save, against in plan:
        snaps_ready = save and all((_snap_dir() / f"{tag}_step{c}.pt").exists() for c in cps)
        if args.reuse_reference and snaps_ready:
            print(f"\n  przebieg {tag}: zdjecia wag JUZ SA -- odtwarzam metryki, nie trenuje")
            runs[tag] = replay_reference(tag, fast=fast, **common)
            print(f"    odtworzony w {runs[tag]['seconds']} s")
        else:
            print(f"\n  przebieg {tag}  (fast_bilinear={fast}"
                  f"{', porownanie z ' + against if against else ''})")
            runs[tag] = run_once(tag, fast=fast, save_snapshots=save,
                                 compare_against=against, **common)
            print(f"    zakonczony w {runs[tag]['seconds']} s "
                  f"({runs[tag]['s_per_step']:.4f} s/krok)")
        # Zapis czastkowy po KAZDYM przebiegu: przerwana sesja nie moze kosztowac
        # wszystkiego, co juz policzone (raz juz kosztowala).
        partial.write_text(json.dumps({"ukonczone": list(runs), "runs": runs},
                                      indent=2, ensure_ascii=False, default=float),
                           encoding="utf-8")

    pairs = {
        "para1_slow_vs_slow": ("slow_1", "slow_2", "PODLOGA: wlasny niedeterminizm cuDNN/atomikow"),
        "para2_fast_vs_fast": ("fast_1", "fast_2", "podloga wariantu szybkiego"),
        "para3_slow_vs_fast": ("slow_1", "fast_1", "EFEKT PODSTAWIENIA"),
    }
    # Porownanie wag jest zapisane w przebiegu, ktory je liczyl (ten z
    # `compare_against`), wiec para czyta je stamtad, a nie z pary wag.
    diff_owner = {"para1_slow_vs_slow": "slow_2",
                  "para2_fast_vs_fast": "fast_2",
                  "para3_slow_vs_fast": "fast_1"}
    results: dict = {}
    for name, (a, b, desc) in pairs.items():
        per_step = {}
        owner = runs[diff_owner[name]]["weight_diffs"]
        for cp in cps:
            d = dict(owner[str(cp)])
            ea = runs[a]["evals"].get(str(cp), {})
            eb = runs[b]["evals"].get(str(cp), {})
            d["rmse_a"] = ea.get("RMSE")
            d["rmse_b"] = eb.get("RMSE")
            d["abs_rmse_diff"] = (abs(ea["RMSE"] - eb["RMSE"])
                                  if ea.get("RMSE") is not None and eb.get("RMSE") is not None
                                  else None)
            per_step[str(cp)] = d
        results[name] = {
            "runs": [a, b],
            "opis": desc,
            "data_order_identical": runs[a]["data_order_sha256"] == runs[b]["data_order_sha256"],
            "per_step": per_step,
        }

    final = str(args.steps)
    p1 = results["para1_slow_vs_slow"]["per_step"][final]
    p2 = results["para2_fast_vs_fast"]["per_step"][final]
    p3 = results["para3_slow_vs_fast"]["per_step"][final]
    ratio_w = p3["rel_l2"] / p1["rel_l2"] if p1["rel_l2"] > 0 else float("inf")
    ratio_r = ((p3["abs_rmse_diff"] / p1["abs_rmse_diff"])
               if p1["abs_rmse_diff"] else float("inf"))
    init_identical = results["para3_slow_vs_fast"]["per_step"]["0"]["bit_identical"] if "0" in results["para3_slow_vs_fast"]["per_step"] else None
    order_ok = all(r["data_order_identical"] for r in results.values())

    verdict = {
        "init_weights_bit_identical": init_identical,
        "data_order_identical_all_runs": order_ok,
        "rel_l2_floor_slow": p1["rel_l2"],
        "rel_l2_floor_fast": p2["rel_l2"],
        "rel_l2_substitution": p3["rel_l2"],
        "ratio_substitution_over_floor_weights": ratio_w,
        "abs_rmse_diff_floor_slow": p1["abs_rmse_diff"],
        "abs_rmse_diff_floor_fast": p2["abs_rmse_diff"],
        "abs_rmse_diff_substitution": p3["abs_rmse_diff"],
        "ratio_substitution_over_floor_rmse": ratio_r,
        # Kryterium: podstawienie nie moze rozjechac sie BARDZIEJ niz podloga.
        # Prog 1.0 z zapasem na skonczonosc pomiaru (jedna para na wielkosc).
        "criterion_met": bool(order_ok and init_identical and ratio_w <= 1.0 and ratio_r <= 1.0),
        "criterion_met_lenient_2x": bool(order_ok and init_identical
                                         and ratio_w <= 2.0 and ratio_r <= 2.0),
    }

    payload = {
        "script": "determinism_check.py",
        "version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "amp": amp,
        "checkpoints": list(cps),
        "val_subset_batches": args.val_batches,
        "runs": runs,
        "pairs": results,
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")
    # ~15 GB zdjec wag ma wartosc wylacznie w trakcie pomiaru -- wszystkie
    # liczby sa juz w JSON-ie, wiec katalog idzie do kosza od razu.
    shutil.rmtree(_snap_dir(), ignore_errors=True)
    partial.unlink(missing_ok=True)

    print("\n" + "=" * 74)
    print(f"WYNIK po {args.steps} krokach")
    print("=" * 74)
    print(f"  {'para':28s} {'rel L2 wag':>12s} {'|d RMSE|':>12s}")
    for name, (a, b, desc) in pairs.items():
        d = results[name]["per_step"][final]
        print(f"  {name:28s} {d['rel_l2']:12.6e} {d['abs_rmse_diff']:12.6e}   {desc}")
    print(f"\n  wagi startowe bit-identyczne (slow vs fast): {init_identical}")
    print(f"  kolejnosc danych identyczna we wszystkich przebiegach: {order_ok}")
    print(f"  iloraz podstawienie/podloga -- wagi: {ratio_w:.3f}, RMSE: {ratio_r:.3f}")
    print(f"  KRYTERIUM (<= 1.0): {'SPELNIONE' if verdict['criterion_met'] else 'NIESPELNIONE'}")
    print(f"\nzapisano: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
