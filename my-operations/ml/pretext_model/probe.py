#!/usr/bin/env python
"""SONDOWANIE ZAMROZONYCH REPREZENTACJI -- diagnoza negatywnego transferu.

    python my-operations/ml/pretext_model/probe.py depth --encoder pretext_K36 --seed 0
    python my-operations/ml/pretext_model/probe.py aux   --encoder pretext_K36 --seed 0
    python my-operations/ml/pretext_model/probe.py cka

PYTANIE. Zadanie pretekstowe orientacji jest rozwiazane bardzo dobrze (MAAE
25,65 +/- 0,74 stopnia wobec 90 losowego), a mimo to pretrenowany enkoder nie
pomaga w predykcji glebi (wszystkie p > 0,07, takze przy 10 % zbioru docelowego).
Diagnoza z 2026-08-13 §5.1 -- "zbior docelowy wystarcza, zeby nauczyc sie tego
samego od zera" -- zostala OBALONA 2026-08-15 §2. Zostaje pytanie: czy cechy,
ktorych uczy sie zadanie pretekstowe, w ogole NIOSA informacje o glebi.

Sonda odpowiada na to bez trenowania czegokolwiek od nowa: enkoder jest
ZAMROZONY, uczy sie wylacznie dekoder. Jesli glebia da sie z tych cech odczytac
tak samo slabo jak z cech LOSOWYCH, to znaczy, ze cechy orientacyjne po prostu
nie sa cechami glebi -- i to jest odpowiedz, a nie brak odpowiedzi.

KONTROLA Z LOSOWYM ZAMROZONYM ENKODEREM JEST OBOWIAZKOWA. Losowe cechy splotowe
sa w wizji zaskakujaco mocnym punktem odniesienia, a `RGBDepthNet` to U-Net
z polaczeniami skrotowymi: `rgbdepth_conv1feature` (64 kanaly w pelnej
rozdzielczosci) trafia wprost do ostatniej warstwy dekodera, wiec nawet losowy
enkoder podaje dekoderowi uzyteczne krawedzie. Bez tej kontroli kazda liczba
sondy jest nieinterpretowalna: nie wiadomo, czy mierzy jakosc pretreningu, czy
sama zdolnosc dekodera do pracy na dowolnych cechach.

DWA ZIARNA, DWIE ROLE (i to nie jest szczegol):
  * ENCODER_SEED  -- STALE dla wszystkich przebiegow. Enkoder `random` musi byc
    TEN SAM w kazdym ziarnie sondy, bo enkodery `pretext_*` i `depth_trained`
    tez sa pojedynczymi, ustalonymi checkpointami. Gdyby losowac go per ziarno,
    warunek `random` dostalby dodatkowe zrodlo wariancji, ktorego pozostale nie
    maja, i porownanie mierzyloby dwie rozne rzeczy.
  * --seed        -- ziarno SONDY: inicjalizacja dekodera/glowy i kolejnosc
    danych. To ono jest powtarzane 3 razy.
Dekoder jest reinicjalizowany po wczytaniu enkodera, wiec przy danym `--seed`
KAZDY warunek startuje z identycznego dekodera.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.pretext_model"

from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import (MAX_DEPTH_REPLICA, DatasetConfig,  # noqa: E402
                                       build_dataloader)
from ..depth_model.metrics import SampleStatsCollector, circular_distance_deg  # noqa: E402
from ..dataset.splits import load_splits  # noqa: E402
from .model import load_pretrained_encoder  # noqa: E402
from .train_pretext import set_seed  # noqa: E402

DEFAULT_STEPS = 40_000
DEFAULT_BATCH = 32

# Ziarno enkodera `random` -- STALE, patrz naglowek modulu.
ENCODER_SEED = 20260817

# Ulamek lokalizacji treningowych odlozony na ewaluacje sond pomocniczych.
AUX_EVAL_FRACTION = 0.20
AUX_SPLIT_SEED = 20260817

ENC_PREFIX = "rgbdepth_convlayer"
DEC_PREFIX = "rgbdepth_upconvlayer"


def encoder_sources() -> dict[str, Path | None]:
    """Skad bierze sie kazdy enkoder. ZADEN nie jest trenowany w tej sesji."""
    P = paths.ML_OUTPUTS
    return {
        "pretext_K36": P / "pretext" / "pretext_K36_seed0" / "best_encoder.pth",
        "pretext_K4": P / "pretext" / "pretext_K4_seed0" / "best_encoder.pth",
        # KONTROLA ROZDZIELAJACA (dodana 2026-08-17 po pierwszych wynikach):
        # gesta siatka 36 orientacji, ale budzet par ROWNY K=4 (16 par na
        # lokalizacje). Bez niej przewaga `pretext_K36` nad `pretext_K4`
        # w sondzie glebi ma dwa mozliwe zrodla -- jakosc pretreningu albo
        # 81x wieksza liczbe par -- i nie da sie ich rozdzielic.
        "pretext_K36_p16": P / "pretext" / "pretext_K36_p16_seed0" / "best_encoder.pth",
        # None = losowy, nietrenowany. Kontrola krytyczna.
        "random": None,
        # Gorna granica: enkoder, ktory PRZESZEDL pelne uczenie na glebi.
        "depth_trained": P / "pretext_transfer" / "transfer_scratch_seed0" / "best_rgbdepth.pth",
    }


def out_dir() -> Path:
    return paths.ML_OUTPUTS / "probing"


# --------------------------------------------------------------- zamrozenie


def encoder_checksum(net: nn.Module) -> str:
    """Suma kontrolna WSZYSTKICH tensorow enkodera, razem z buforami.

    Bufory (`running_mean`/`running_var` BatchNormu) sa tu celowo: `requires_grad
    = False` ich NIE zamraza -- zmieniaja sie przy kazdym przejsciu w przod
    w trybie `train()`. Suma liczona tylko po parametrach pokazalaby "wagi
    nietkniete" przy enkoderze, ktory po cichu dryfuje statystykami.
    """
    h = hashlib.sha256()
    for name, t in sorted(net.state_dict().items()):
        if name.startswith(ENC_PREFIX):
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def freeze_encoder(net: nn.Module) -> None:
    for name, mod in net.named_children():
        if name.startswith(ENC_PREFIX):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)


def build_net(encoder: str, probe_seed: int):
    """Buduje `RGBDepthNet` z zadanym enkoderem; dekoder swiezy wg `probe_seed`."""
    paths.add_parida_to_syspath()
    from models.models import ModelBuilder
    from models.networks import weights_init

    # 1. Enkoder: zawsze z tego samego ziarna, zeby `random` byl powtarzalny.
    set_seed(ENCODER_SEED)
    net = ModelBuilder().build_rgbdepth()

    src = encoder_sources()[encoder]
    if src is None:
        report = {"init": "random", "encoder_seed": ENCODER_SEED, "ok": True, "n_loaded": 0}
    else:
        if not src.exists():
            raise FileNotFoundError(f"brak checkpointu enkodera: {src}")
        report = load_pretrained_encoder(net, src)
        if not report["ok"]:
            raise RuntimeError(f"przeniesienie enkodera nie doszlo do skutku: {src}")

    # 2. Dekoder: reinicjalizowany ziarnem SONDY -> przy danym --seed kazdy
    #    warunek startuje z identycznego dekodera, wiec roznica wyniku pochodzi
    #    wylacznie z enkodera.
    set_seed(probe_seed)
    for name, mod in net.named_children():
        if name.startswith(DEC_PREFIX):
            mod.apply(weights_init)

    freeze_encoder(net)
    return net, report


class FrozenEncoderDepthNet(nn.Module):
    """U-Net Paridy z ZAMROZONYM enkoderem. Uczy sie wylacznie dekoder.

    Przejscie w przod odtwarza `RGBDepthNet.forward()` co do litery, razem
    z polaczeniami skrotowymi -- rozni sie wylacznie tym, ze czesc enkodujaca
    idzie pod `torch.no_grad()`. Skroty ZOSTAJA: sonda ma mierzyc, ile glebi da
    sie odczytac z cech tego enkodera, a nie z okrojonej architektury.
    """

    def __init__(self, net: nn.Module, max_depth: float):
        super().__init__()
        self.net = net
        self.max_depth = max_depth

    def train(self, mode: bool = True):
        """Enkoder ZAWSZE w `eval()`, niezaleznie od trybu modulu.

        Bez tego `model.train()` przestawilby BatchNormy enkodera z powrotem
        w tryb uczenia i ich statystyki biegnace zaczelyby sie zmieniac --
        enkoder przestalby byc zamrozony, mimo `requires_grad = False`.
        """
        super().train(mode)
        freeze_encoder(self.net)
        return self

    def forward(self, x):
        n = self.net
        with torch.no_grad():
            c1 = n.rgbdepth_convlayer1(x["img"])
            c2 = n.rgbdepth_convlayer2(c1)
            c3 = n.rgbdepth_convlayer3(c2)
            c4 = n.rgbdepth_convlayer4(c3)
            c5 = n.rgbdepth_convlayer5(c4)
        u1 = n.rgbdepth_upconvlayer1(c5)
        u2 = n.rgbdepth_upconvlayer2(torch.cat((u1, c4), dim=1))
        u3 = n.rgbdepth_upconvlayer3(torch.cat((u2, c3), dim=1))
        u4 = n.rgbdepth_upconvlayer4(torch.cat((u3, c2), dim=1))
        d = n.rgbdepth_upconvlayer5(torch.cat((u4, c1), dim=1))
        scaled = d * self.max_depth
        return {"img_depth": scaled, "depth_predicted": scaled,
                "audio_depth": None, "attention": None, "depth_gt": x["depth"]}


# ------------------------------------------------------------- §1.1 sonda glebi


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
    return res


def cmd_depth(args) -> int:
    rid = f"probe_depth_{args.encoder}_seed{args.seed}"
    d = out_dir() / rid
    splits = load_splits(variant="main")
    amp = not args.no_amp
    device = torch.device("cuda")

    print("=" * 78)
    print(f"SONDA GLEBI (enkoder ZAMROZONY)   {args.encoder}   ziarno {args.seed}")
    print("=" * 78)

    net, report = build_net(args.encoder, args.seed)
    sum_before = encoder_checksum(net)
    n_frozen = sum(p.numel() for n_, m in net.named_children()
                   if n_.startswith(ENC_PREFIX) for p in m.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  enkoder: {report.get('init', 'checkpoint')}  wczytanych kluczy "
          f"{report.get('n_loaded')}  suma kontrolna {sum_before}")
    print(f"  zamrozonych parametrow {n_frozen:,} · uczonych {n_train:,}")

    model = FrozenEncoderDepthNet(net, MAX_DEPTH_REPLICA).to(device)
    model.train()
    paths.add_parida_to_syspath()
    from models import criterion
    loss_fn = criterion.LogDepthLoss()
    # Optymalizator dostaje WYLACZNIE parametry z gradientem -- inaczej Adam
    # trzymalby momenty dla zamrozonych wag i cicho sugerowal, ze sa uczone.
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    train_loader, train_ds = build_dataloader(
        DatasetConfig(variant="main", mode="train", angle_subset="all"),
        batch_size=args.batch_size, num_workers=args.num_workers, splits=splits)
    val_loader, val_ds = build_dataloader(
        DatasetConfig(variant="main", mode="val", angle_subset="all", augment=False),
        batch_size=args.batch_size, num_workers=max(2, args.num_workers // 2),
        splits=splits, shuffle=False, drop_last=False)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    if args.dry_run:
        train_ds.close(); val_ds.close()
        print("\n--dry-run: nic nie uruchomiono.")
        return 0
    if (d / "status.json").exists() and not args.force:
        print(f"\nBLAD: {d} juz gotowe. Uzyj --force.")
        train_ds.close(); val_ds.close()
        return 2
    d.mkdir(parents=True, exist_ok=True)

    best = {"rmse": float("inf"), "step": 0}
    step, epoch = 0, 0
    it = iter(train_loader)
    running, t0 = [], time.perf_counter()
    metrics_fp = d / "metrics.jsonl"

    while step < args.steps:
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
                  f"{args.display_freq * args.batch_size / dt:.1f} probek/s")
            running, t0 = [], time.perf_counter()

        if step % args.validation_freq == 0 or step == args.steps:
            r = evaluate(model, val_loader, device, amp, loss_fn, val_ds.scenes,
                         args.edge_threshold)
            with metrics_fp.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **r}, ensure_ascii=False, default=str) + "\n")
            print(f"  [val {step}] RMSE {r['all']['RMSE']:.5f}")
            if r["all"]["RMSE"] < best["rmse"]:
                best = {"rmse": r["all"]["RMSE"], "step": step}
                (d / "best.json").write_text(
                    json.dumps({"step": step, **r}, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
            t0 = time.perf_counter()

    sum_after = encoder_checksum(net)
    frozen_ok = sum_before == sum_after
    (d / "status.json").write_text(json.dumps({
        "run_id": rid, "encoder": args.encoder, "seed": args.seed,
        "encoder_source": str(encoder_sources()[args.encoder]),
        "encoder_report": report,
        "checksum_before": sum_before, "checksum_after": sum_after,
        "encoder_frozen_verified": frozen_ok,
        "n_frozen_params": n_frozen, "n_trained_params": n_train,
        "best_val_rmse": best["rmse"], "best_step": best["step"],
        "steps": step, "finished": step >= args.steps,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    train_ds.close(); val_ds.close()

    print(f"\n  KONTROLA ZAMROZENIA: {sum_before} -> {sum_after}  "
          f"{'OK' if frozen_ok else 'NARUSZONE!'}")
    if not frozen_ok:
        print("  BLAD: enkoder sie zmienil -- wynik sondy jest niewazny.")
        return 3
    print(f"KONIEC: najlepszy val RMSE {best['rmse']:.5f} na kroku {best['step']}")
    return 0


# ------------------------------------------------- §1.2 sondy pomocnicze


@torch.no_grad()
def cache_features(net, device, splits, amp: bool, batch_size: int, workers: int):
    """Usrednione przestrzennie cechy conv5 (512-D) dla lokalizacji TRENINGOWYCH.

    Sonda liniowa nie potrzebuje przeplywu wstecznego przez enkoder, wiec cechy
    liczymy RAZ i trenujemy na nich glowy w sekundach zamiast godzin.
    Augmentacja WYLACZONA -- przy zapamietanych cechach i tak utrwalilaby jedno
    losowanie, a bez niej sonda jest deterministyczna.
    """
    loader, ds = build_dataloader(
        DatasetConfig(variant="main", mode="train", angle_subset="all", augment=False),
        batch_size=batch_size, num_workers=workers, splits=splits,
        shuffle=False, drop_last=False)
    feats, angles, scenes, locs = [], [], [], []
    for b in loader:
        img = b["img"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            x = net.rgbdepth_convlayer1(img)
            x = net.rgbdepth_convlayer2(x)
            x = net.rgbdepth_convlayer3(x)
            x = net.rgbdepth_convlayer4(x)
            x = net.rgbdepth_convlayer5(x)
        feats.append(x.float().mean(dim=(2, 3)).cpu())
        angles.append(b["angle_deg"].clone())
        scenes.append(b["scene_idx"].clone())
        locs.append(b["location_id"].clone())
    ds.close()
    return (torch.cat(feats), torch.cat(angles).numpy(),
            torch.cat(scenes).numpy(), torch.cat(locs).numpy())


def _aux_split(scene_idx: np.ndarray, loc_id: np.ndarray) -> np.ndarray:
    """Maska ewaluacyjna sond pomocniczych -- podzial PO LOKALIZACJI.

    DLACZEGO WLASNY PODZIAL, A NIE ZAMROZONY `replica_locations.json`. Tamten
    oddaje na `val`/`test` trzy sceny HELD-OUT w calosci, a klasami sond
    pomocniczych sa sceny TRENINGOWE (tozsamosc sceny) albo orientacja
    bezwzgledna, ktora w nieznanym pomieszczeniu nie ma punktu odniesienia.
    Ewaluacja na scenach held-out dalaby wiec poziom losowy dla KAZDEGO warunku,
    lacznie z `depth_trained` -- czyli pomiar bez mocy rozdzielczej.
    Dzielimy zamiast tego lokalizacje TRENINGOWE 80/20: sonda uczy sie na jednych
    pozycjach, a odpowiada na innych, w tych samych pomieszczeniach.
    Zamrozony podzial zbioru NIE jest tu ruszany -- to jest podzial wewnetrzny
    sondy, uzywany wylacznie tutaj.
    """
    key = scene_idx.astype(np.int64) * 100_000 + loc_id.astype(np.int64)
    uniq = np.unique(key)
    rng = np.random.default_rng(AUX_SPLIT_SEED)
    n_eval = int(round(uniq.size * AUX_EVAL_FRACTION))
    eval_locs = set(rng.choice(uniq, size=n_eval, replace=False).tolist())
    return np.array([k in eval_locs for k in key])


def _train_linear_head(X, y, is_eval, n_classes, seed, device, steps=3000, lr=1e-3):
    """Jedna warstwa liniowa na zapamietanych cechach. Zwraca predykcje na eval."""
    torch.manual_seed(seed)
    head = nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr = X[~is_eval].to(device)
    ytr = torch.from_numpy(y[~is_eval]).long().to(device)
    Xev = X[is_eval].to(device)
    # Standaryzacja liczona WYLACZNIE na czesci uczacej -- inaczej sonda
    # widzialaby statystyki zbioru ewaluacyjnego.
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xev = (Xtr - mu) / sd, (Xev - mu) / sd
    n = Xtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n, (256,), generator=g).to(device)
        opt.zero_grad(set_to_none=True)
        lossf(head(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    with torch.no_grad():
        return head(Xev).argmax(1).cpu().numpy()


def cmd_aux(args) -> int:
    device = torch.device("cuda")
    splits = load_splits(variant="main")
    amp = not args.no_amp
    net, report = build_net(args.encoder, args.seed)
    net = net.to(device).eval()
    sum_before = encoder_checksum(net)

    print("=" * 78)
    print(f"SONDY POMOCNICZE (enkoder ZAMROZONY)   {args.encoder}   ziarno {args.seed}")
    print("=" * 78)
    t0 = time.perf_counter()
    X, angles, scene_idx, loc_id = cache_features(
        net, device, splits, amp, args.batch_size, args.num_workers)
    print(f"  cechy: {tuple(X.shape)} w {time.perf_counter() - t0:.0f} s")

    is_eval = _aux_split(scene_idx, loc_id)
    print(f"  podzial sondy: uczace {int((~is_eval).sum())} · ewaluacyjne {int(is_eval.sum())}")

    res = {}

    # --- orientacja bezwzgledna: 36 klas co 10 stopni ---
    y_ang = (angles // 10).astype(np.int64)
    pred = _train_linear_head(X, y_ang, is_eval, 36, args.seed, device)
    true_deg = angles[is_eval].astype(float)
    pred_deg = (pred * 10).astype(float)
    maae = float(np.mean(circular_distance_deg(pred_deg, true_deg)))
    res["orientacja"] = {
        "n_classes": 36, "MAAE_deg": maae, "MAAE_chance_deg": 90.0,
        "top1": float(np.mean(pred == y_ang[is_eval])), "top1_chance": 1 / 36,
        "uwaga": "MAAE jest metryka porownywalna; poziom losowy 90 stopni niezaleznie od K",
    }
    print(f"  orientacja:      MAAE {maae:6.2f} st. (losowo 90)   "
          f"top-1 {100 * res['orientacja']['top1']:.1f} % (losowo 2,8 %)")

    # --- tozsamosc sceny: 15 scen treningowych ---
    if not args.orientation_only:
        n_sc = int(scene_idx.max()) + 1
        pred_s = _train_linear_head(X, scene_idx.astype(np.int64), is_eval, n_sc,
                                    args.seed, device)
        acc = float(np.mean(pred_s == scene_idx[is_eval]))
        res["scena"] = {"n_classes": n_sc, "top1": acc, "top1_chance": 1 / n_sc}
        print(f"  tozsamosc sceny: top-1 {100 * acc:.1f} % (losowo {100 / n_sc:.1f} %)")

    sum_after = encoder_checksum(net)
    d = out_dir() / f"probe_aux_{args.encoder}_seed{args.seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({
        "encoder": args.encoder, "seed": args.seed, "encoder_report": report,
        "checksum_before": sum_before, "checksum_after": sum_after,
        "encoder_frozen_verified": sum_before == sum_after,
        "n_eval": int(is_eval.sum()), "n_train": int((~is_eval).sum()),
        "aux_split_seed": AUX_SPLIT_SEED, "aux_eval_fraction": AUX_EVAL_FRACTION,
        "wyniki": res, "finished": True,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  KONTROLA ZAMROZENIA: {'OK' if sum_before == sum_after else 'NARUSZONE!'}")
    return 0 if sum_before == sum_after else 3


# ------------------------------------------------------------------ §1.3 CKA


def _linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Liniowe CKA miedzy dwiema macierzami aktywacji (n_probek x n_cech)."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    hsic = np.linalg.norm(B.T @ A, ord="fro") ** 2
    na = np.linalg.norm(A.T @ A, ord="fro")
    nb = np.linalg.norm(B.T @ B, ord="fro")
    return float(hsic / (na * nb)) if na > 0 and nb > 0 else float("nan")


@torch.no_grad()
def _layer_activations(net, device, splits, n_images: int, batch_size: int):
    loader, ds = build_dataloader(
        DatasetConfig(variant="main", mode="val", angle_subset="all", augment=False),
        batch_size=batch_size, num_workers=4, splits=splits, shuffle=False, drop_last=False)
    acc: dict[str, list] = {f"conv{i}": [] for i in range(1, 6)}
    seen = 0
    for b in loader:
        img = b["img"].to(device, non_blocking=True)
        x = img
        for i in range(1, 6):
            x = getattr(net, f"rgbdepth_convlayer{i}")(x)
            acc[f"conv{i}"].append(x.float().mean(dim=(2, 3)).cpu().numpy())
        seen += img.shape[0]
        if seen >= n_images:
            break
    ds.close()
    return {k: np.concatenate(v)[:n_images] for k, v in acc.items()}


def cmd_cka(args) -> int:
    device = torch.device("cuda")
    splits = load_splits(variant="main")
    acts = {}
    for enc in ("pretext_K36", "random", "depth_trained"):
        net, _ = build_net(enc, 0)
        net = net.to(device).eval()
        acts[enc] = _layer_activations(net, device, splits, args.n_images, args.batch_size)
        print(f"  aktywacje {enc}: gotowe")

    pary = (("pretext_K36", "depth_trained"), ("random", "depth_trained"),
            ("pretext_K36", "random"))
    res = {f"{a}__vs__{b}": {L: _linear_cka(acts[a][L], acts[b][L])
                             for L in acts[a]} for a, b in pary}
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "cka.json").write_text(json.dumps({
        "n_images": args.n_images, "metric": "linear CKA na cechach usrednionych przestrzennie",
        "uwaga": "para `random vs depth_trained` jest PODLOGA -- bez niej liczby CKA "
                 "nie maja skali odniesienia",
        "wyniki": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {'para':34s} " + " ".join(f"{f'conv{i}':>7s}" for i in range(1, 6)))
    for k, v in res.items():
        print(f"  {k:34s} " + " ".join(f"{v[f'conv{i}']:7.3f}" for i in range(1, 6)))
    print(f"\nzapisano: {d / 'cka.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--encoder", required=True, choices=sorted(encoder_sources()))
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
        p.add_argument("--num-workers", type=int, default=8)
        p.add_argument("--no-amp", action="store_true")

    pd = sub.add_parser("depth", help="sonda glebi -- zamrozony enkoder, uczony dekoder")
    common(pd)
    pd.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    pd.add_argument("--lr", type=float, default=1e-4)
    pd.add_argument("--weight-decay", type=float, default=5e-4)
    pd.add_argument("--validation-freq", type=int, default=1000)
    pd.add_argument("--display-freq", type=int, default=200)
    pd.add_argument("--edge-threshold", type=float, default=0.10)
    pd.add_argument("--force", action="store_true")
    pd.add_argument("--dry-run", action="store_true")
    pd.set_defaults(func=cmd_depth)

    pa = sub.add_parser("aux", help="sondy liniowe: orientacja i tozsamosc sceny")
    common(pa)
    pa.add_argument("--orientation-only", action="store_true")
    pa.set_defaults(func=cmd_aux)

    pc = sub.add_parser("cka", help="podobienstwo reprezentacji warstwa po warstwie")
    pc.add_argument("--n-images", type=int, default=1000)
    pc.add_argument("--batch-size", type=int, default=32)
    pc.set_defaults(func=cmd_cka)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
