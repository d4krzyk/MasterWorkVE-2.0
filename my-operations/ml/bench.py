"""Benchmark przepustowosci -- Blok 3.

Pytanie, na ktore odpowiada: czy GPU czeka na dysk. Kazda probka to ~295 KB
surowych danych w trzech chunkach gzip (echo 167 KB + RGB 64 KB + glebia 64 KB),
a chunking HDF5 jest 1:1 z probka, wiec kazdy `__getitem__` to trzy niezalezne
dekompresje. Przy losowej kolejnosci (a trening MUSI tasowac) nie ma zadnej
lokalnosci odczytu, wiec cache stron systemu pomaga tylko przy drugiej epoce.

Mierzone sa trzy rzeczy osobno, bo tylko rozdzielone daja odpowiedz:
  1. sam dataloader, bez modelu       -> ile probek/s dostarcza CPU
  2. sam model, na danych syntetycznych -> ile probek/s przerabia GPU
  3. porownanie (1) z (2)             -> kto jest waskim gardlem

Wariant bez kompresji jest mierzony, a nie zgadywany: kompresja gzip na tym
zbiorze daje wspolczynnik zaledwie ~1.23 (240 KB na dysku wobec 295 KB surowo),
wiec placi sie za nia pelnym kosztem CPU przy prawie zadnej oszczednosci
miejsca -- ale to trzeba pokazac pomiarem, a nie arytmetyka.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

from . import paths
from .echo_h5_dataset import DatasetConfig, EchoH5Dataset, build_dataloader
from .splits import load_splits


@dataclass
class LoaderResult:
    num_workers: int
    batch_size: int
    prefetch_factor: int | None
    augment: bool
    batches: int
    samples: int
    seconds: float
    samples_per_s: float
    batches_per_s: float
    full_epoch: bool
    source: str = "gzip"


@dataclass
class ModelResult:
    batch_size: int
    amp: bool
    iters: int
    seconds: float
    samples_per_s: float
    ms_per_iter: float
    peak_mem_gb: float


# ------------------------------------------------------------------ dataloader


def bench_loader(
    *,
    variant: str = "main",
    mode: str = "train",
    angle_subset: str = "all",
    num_workers: int = 4,
    batch_size: int = 32,
    prefetch_factor: int | None = None,
    max_batches: int | None = 300,
    warmup_batches: int = 5,
    augment: bool = True,
    dataset_override: EchoH5Dataset | None = None,
) -> LoaderResult:
    """Jedno przejscie po dataloaderze BEZ modelu.

    `warmup_batches` sa odrzucane: pierwsze wsady obejmuja start procesow
    workerow i zapelnienie kolejki prefetch, wiec wliczone zanizalyby wynik tym
    bardziej, im krotszy pomiar -- a porownujemy konfiguracje miedzy soba.
    """
    splits = load_splits(variant=variant)
    cfg = DatasetConfig(variant=variant, mode=mode, angle_subset=angle_subset, augment=augment)
    loader, ds = build_dataloader(
        cfg, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        splits=splits, prefetch_factor=prefetch_factor, pin_memory=True, drop_last=True,
    )

    total_batches = len(loader)
    limit = total_batches if max_batches is None else min(max_batches, total_batches)

    it = iter(loader)
    for _ in range(min(warmup_batches, limit)):
        next(it)

    n = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(it):
        # Dotykamy tensora, zeby zmusic do faktycznej materializacji -- bez tego
        # mierzylibysmy koszt kolejki, a nie odczytu.
        _ = int(batch["audio"].shape[0])
        n += 1
        if n >= limit - warmup_batches:
            break
    dt = time.perf_counter() - t0
    ds.close()
    del loader

    samples = n * batch_size
    return LoaderResult(
        num_workers=num_workers, batch_size=batch_size, prefetch_factor=prefetch_factor,
        augment=augment, batches=n, samples=samples, seconds=round(dt, 3),
        samples_per_s=round(samples / dt, 1), batches_per_s=round(n / dt, 2),
        full_epoch=(max_batches is None),
    )


# ----------------------------------------------------------------------- model


def _build_nets(audio_shape, device):
    """Buduje siec Paridy bez modyfikacji jej plikow.

    UWAGA: `train.py` Paridy wola `builder.build_audiodepth()` BEZ argumentu,
    czyli z domyslnym `audio_shape=[2,257,121]` -- ksztaltem dla mp3d. Dla
    Repliki wejscie ma 166 ramek, wiec splaszczona warstwa `conv1x1` wychodzi
    inna (8*28*17 = 3808 zamiast 8*28*11 = 2464) i forward konczy sie bledem
    ksztaltu. Tu przekazujemy `audio_shape` jawnie.
    """
    paths.add_parida_to_syspath()
    from models.models import ModelBuilder

    b = ModelBuilder()
    nets = (
        b.build_rgbdepth(),
        b.build_audiodepth(audio_shape=list(audio_shape)),
        b.build_attention(),
        b.build_material_property(),
    )
    return tuple(n.to(device) for n in nets)


def bench_model(
    *,
    batch_size: int = 32,
    amp: bool = False,
    iters: int = 30,
    warmup: int = 8,
    audio_shape=(2, 257, 166),
    device: str = "cuda",
) -> ModelResult:
    """Forward + backward na danych SYNTETYCZNYCH.

    Dane losowe celowo: chcemy czysty czas GPU, bez wtretu z dysku. Liczba
    krokow optymalizatora i architektura sa te same co w treningu, wiec wynik
    przenosi sie 1:1 na szacunek czasu przebiegu.
    """
    paths.add_parida_to_syspath()
    from models.audioVisual_model import AudioVisualModel
    from models import criterion

    dev = torch.device(device)
    nets = _build_nets(audio_shape, dev)

    class _Opt:
        max_depth = 14.104

    model = AudioVisualModel(nets, _Opt()).to(dev)
    model.train()
    loss_fn = criterion.LogDepthLoss()
    optimizer = torch.optim.Adam(
        [{"params": n.parameters(), "lr": 1e-4} for n in nets],
        betas=(0.9, 0.999), weight_decay=5e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    batch = {
        "img": torch.randn(batch_size, 3, 128, 128, device=dev),
        "audio": torch.rand(batch_size, *audio_shape, device=dev) * 5.0,
        # Glebia dodatnia i w zasiegu modelu -- strata maskuje zera, wiec same
        # zera daly by pusty tensor i bezsensowny pomiar.
        "depth": torch.rand(batch_size, 1, 128, 128, device=dev) * 8.0 + 0.2,
    }

    def step():
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(batch)
            dp, dg = out["depth_predicted"], out["depth_gt"]
            loss = loss_fn(dp[dg != 0], dg[dg != 0])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, nets, optimizer
    torch.cuda.empty_cache()

    return ModelResult(
        batch_size=batch_size, amp=amp, iters=iters, seconds=round(dt, 3),
        samples_per_s=round(iters * batch_size / dt, 1),
        ms_per_iter=round(1000 * dt / iters, 2),
        peak_mem_gb=round(peak, 2),
    )


# ------------------------------------------------------- wariant bez kompresji


def rewrite_uncompressed(scene: str, variant: str, out_root: Path | None = None) -> Path:
    """Kopiuje jedna scene do HDF5 BEZ kompresji, zachowujac uklad i atrybuty.

    Sluzy wylacznie do pomiaru -- gdyby wygral, przepisanie calego zbioru jest
    mechaniczne. Chunking zostaje ten sam (1 probka), zeby porownanie dotyczylo
    tylko kompresji, a nie ukladu danych.
    """
    out_root = out_root or (paths.BENCH_DIR / "uncompressed" / variant)
    out_root.mkdir(parents=True, exist_ok=True)
    dst_dir = out_root / scene
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{scene}.h5"
    if dst.exists():
        return dst

    src = paths.scene_h5(scene, variant)
    tmp = dst.with_suffix(".h5.tmp")
    with h5py.File(src, "r") as fs, h5py.File(tmp, "w") as fd:
        for k, v in fs.attrs.items():
            fd.attrs[k] = v
        fd.attrs["compression"] = "none (bench copy)"

        def copy(name, obj):
            if isinstance(obj, h5py.Dataset):
                d = fd.create_dataset(
                    name, shape=obj.shape, dtype=obj.dtype,
                    chunks=obj.chunks, compression=None,
                )
                # Kopiowanie blokami: caly `echo` jednej sceny to kilka GB.
                if obj.chunks and obj.shape[0] > 512:
                    for i in range(0, obj.shape[0], 256):
                        d[i:i + 256] = obj[i:i + 256]
                else:
                    d[...] = obj[...]
                for ak, av in obj.attrs.items():
                    d.attrs[ak] = av
            else:
                g = fd.require_group(name)
                for ak, av in obj.attrs.items():
                    g.attrs[ak] = av

        fs.visititems(copy)
    tmp.rename(dst)
    return dst


class _UncompressedDataset(EchoH5Dataset):
    """Ten sam dataset, ale czytajacy z kopii bez kompresji."""

    def __init__(self, cfg: DatasetConfig, splits, root: Path):
        self._bench_root = root
        super().__init__(cfg, splits=splits)

    def _build_index(self):
        super()._build_index()
        self.scene_paths = [str(self._bench_root / s / f"{s}.h5") for s in self.scenes]


def bench_uncompressed(
    scene: str, *, variant: str = "main", num_workers: int = 4,
    batch_size: int = 32, max_batches: int = 200, augment: bool = True,
) -> tuple[LoaderResult, LoaderResult, dict]:
    """Porownuje odczyt gzip vs bez kompresji na JEDNEJ scenie.

    Pojedyncza scena, a nie caly zbior: chodzi o stosunek przepustowosci, ktory
    jest wlasnoscia formatu, a nie rozmiaru. Przepisanie 26 GiB tylko po to,
    zeby zmierzyc stosunek, byloby marnotrawstwem doby pracy dysku.
    """
    from .splits import Splits

    dst = rewrite_uncompressed(scene, variant)
    root = dst.parent.parent

    # Sztuczny podzial: jedna scena, wszystkie jej lokalizacje jako "train".
    with h5py.File(paths.scene_h5(scene, variant), "r") as f:
        loc_ids = sorted(int(x) for x in f["locations/loc_id"][:])
    one = Splits(train={scene: loc_ids}, val={}, test={}, meta={"synthetic": True})

    cfg = DatasetConfig(variant=variant, mode="train", angle_subset="all", augment=augment)

    def run(ds_cls_kwargs, tag) -> LoaderResult:
        ds = (_UncompressedDataset(cfg, one, root) if tag == "uncompressed"
              else EchoH5Dataset(cfg, splits=one))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True,
            **({"persistent_workers": True} if num_workers > 0 else {}),
        )
        it = iter(loader)
        for _ in range(3):
            next(it)
        n = 0
        t0 = time.perf_counter()
        for batch in it:
            _ = int(batch["audio"].shape[0])
            n += 1
            if n >= max_batches:
                break
        dt = time.perf_counter() - t0
        ds.close()
        return LoaderResult(
            num_workers=num_workers, batch_size=batch_size, prefetch_factor=None,
            augment=augment, batches=n, samples=n * batch_size, seconds=round(dt, 3),
            samples_per_s=round(n * batch_size / dt, 1), batches_per_s=round(n / dt, 2),
            full_epoch=False, source=tag,
        )

    gz = run(None, "gzip")
    un = run(None, "uncompressed")
    sizes = {
        "gzip_MB": round(paths.scene_h5(scene, variant).stat().st_size / 1e6, 1),
        "uncompressed_MB": round(dst.stat().st_size / 1e6, 1),
    }
    sizes["wspolczynnik_kompresji"] = round(sizes["uncompressed_MB"] / sizes["gzip_MB"], 3)
    sizes["przyspieszenie"] = round(un.samples_per_s / gz.samples_per_s, 2)
    return gz, un, sizes


def cleanup_uncompressed(variant: str = "main") -> int:
    """Kasuje kopie benchmarkowe -- to sa gigabajty, ktore nie sa danymi."""
    root = paths.BENCH_DIR / "uncompressed" / variant
    if not root.exists():
        return 0
    n = sum(1 for _ in root.rglob("*.h5"))
    shutil.rmtree(root)
    return n


# ---------------------------------------------------------------------- raport


def estimate_run_hours(model_ms_per_iter: float, loader_samples_per_s: float,
                       batch_size: int, steps: int) -> dict:
    """Szacunek czasu jednego przebiegu przy STALEJ liczbie krokow gradientu.

    Krok trwa tyle, ile wolniejsza z dwoch sciezek -- one nakladaja sie w
    czasie (workery czytaja, gdy GPU liczy), wiec `max`, a nie suma.
    """
    gpu_s = model_ms_per_iter / 1000.0
    io_s = batch_size / loader_samples_per_s if loader_samples_per_s > 0 else float("inf")
    step_s = max(gpu_s, io_s)
    return {
        "krokow": steps,
        "batch_size": batch_size,
        "s_na_krok_gpu": round(gpu_s, 4),
        "s_na_krok_io": round(io_s, 4),
        "s_na_krok_efektywnie": round(step_s, 4),
        "waskie_gardlo": "dataloader (I/O)" if io_s > gpu_s else "GPU",
        "godzin_bez_walidacji": round(steps * step_s / 3600, 2),
    }


def save_report(payload: dict, name: str) -> Path:
    paths.BENCH_DIR.mkdir(parents=True, exist_ok=True)
    fp = paths.BENCH_DIR / f"{name}.json"
    payload = dict(payload)
    payload["utc"] = datetime.now(timezone.utc).isoformat()
    fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return fp
