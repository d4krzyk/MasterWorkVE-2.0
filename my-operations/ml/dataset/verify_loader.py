"""Weryfikacja dataloadera -- Blok 2.

Sprawdza rzeczy, ktorych zla wartosc NIE objawia sie wyjatkiem, tylko cicho
psuje wynik naukowy:

* ksztalt/dtype wsadu niezgodny z `audioVisual_model.py` -> siec i tak policzy
  cos, byle liczby sie zgadzaly wymiarowo;
* NaN/Inf w echu -> strata `log(|d|+1)` da NaN i gradient znika bez komunikatu;
* zla licznosc podzbioru katow -> caly wniosek o gestosci katowej idzie do kosza;
* niepowtarzalny `random_K` -> warunek D nie da sie odtworzyc;
* wspolna lokalizacja w val i test -> wybor checkpointu na zbiorze testowym;
* glebia > `max_depth` -> model mnozy sigmoid przez 14.104, wiec te piksele sa
  NIEOSIAGALNE z definicji i wchodza do RMSE jako staly, nieusuwalny blad.

Kazdy test zwraca (nazwa, PASS/FAIL, liczby). Skrypt konczy sie kodem != 0,
jesli cokolwiek nie przeszlo -- zeby dalo sie go wpiac przed start treningu.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

from . import angles as angles_mod
from .. import paths
from .echo_h5_dataset import (
    EXPECTED_IMAGE_SHAPE,
    EXPECTED_SPEC_SHAPE,
    MAX_DEPTH_REPLICA,
    DatasetConfig,
    EchoH5Dataset,
    build_dataloader,
    expected_n_samples,
)
from .splits import ALL_SCENES, HELDOUT_SCENES, Splits, load_splits


@dataclass
class Check:
    name: str
    passed: bool
    detail: dict

    def line(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}"


# --------------------------------------------------------------------- testy


def check_batch_contract(splits: Splits, variant: str, batch_size: int = 8) -> Check:
    """Ksztalty i dtype dokladnie takie, jakich dotyka `AudioVisualModel.forward`.

    Wartosci referencyjne nie sa zgadywane: `opt.audio_shape = [2,257,166]` i
    `RGBDepthNet(input_nc=3)` z repo Paridy, `depth` z jednym kanalem, bo
    `depth_prediction` wychodzi z `unet_upconv(..., output_nc=1)`.
    """
    detail: dict = {}
    ok = True
    for mode in ("train", "val", "test"):
        cfg = DatasetConfig(variant=variant, mode=mode, angle_subset="all")
        loader, ds = build_dataloader(
            cfg, batch_size=batch_size, num_workers=0, shuffle=False,
            splits=splits, pin_memory=False, drop_last=False,
        )
        batch = next(iter(loader))
        got = {
            "img": (tuple(batch["img"].shape), str(batch["img"].dtype)),
            "depth": (tuple(batch["depth"].shape), str(batch["depth"].dtype)),
            "audio": (tuple(batch["audio"].shape), str(batch["audio"].dtype)),
        }
        want = {
            "img": ((batch_size, 3, *EXPECTED_IMAGE_SHAPE), "torch.float32"),
            "depth": ((batch_size, 1, *EXPECTED_IMAGE_SHAPE), "torch.float32"),
            "audio": ((batch_size, *EXPECTED_SPEC_SHAPE), "torch.float32"),
        }
        mode_ok = got == want
        ok &= mode_ok
        detail[mode] = {"got": {k: [list(v[0]), v[1]] for k, v in got.items()},
                        "want": {k: [list(v[0]), v[1]] for k, v in want.items()},
                        "ok": mode_ok}
        ds.close()
    return Check("ksztalt i dtype wsadu zgodne z audioVisual_model.py", ok, detail)


def check_finite(splits: Splits, variant: str, n_batches: int = 40, batch_size: int = 32) -> Check:
    """NaN/Inf na losowej probce wsadow z kazdego podzbioru."""
    detail: dict = {}
    ok = True
    for mode in ("train", "val", "test"):
        # augment=False: sprawdzamy DANE, nie generator liczb losowych PIL-a.
        cfg = DatasetConfig(variant=variant, mode=mode, angle_subset="all", augment=False)
        loader, ds = build_dataloader(
            cfg, batch_size=batch_size, num_workers=4, shuffle=True,
            splits=splits, pin_memory=False, drop_last=False,
        )
        bad = {"img": 0, "depth": 0, "audio": 0}
        seen = 0
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            seen += batch["img"].shape[0]
            for k in bad:
                bad[k] += int((~torch.isfinite(batch[k])).sum())
        mode_ok = all(v == 0 for v in bad.values())
        ok &= mode_ok
        detail[mode] = {"samples_checked": seen, "nonfinite": bad, "ok": mode_ok}
        ds.close()
    return Check("brak NaN/Inf", ok, detail)


def check_counts(splits: Splits, variant: str) -> Check:
    """Licznosc kazdego podzbioru katow -- osobno per split i lacznie.

    Liczba oczekiwana bierze sie z podzialu (liczba lokalizacji x katy na
    lokalizacje), a nie z drugiego przebiegu tego samego kodu, wiec test naprawde
    cos sprawdza, a nie porownuje funkcji z sama soba.
    """
    subsets = ["all", "every_2", "every_3", "every_4", "every_6", "every_9",
               "cardinal", "random_4", "random_6", "random_9", "random_12", "random_18"]
    detail: dict = {}
    ok = True
    for sub in subsets:
        per_loc = angles_mod.angles_per_location(sub)
        row = {"angles_per_location": per_loc}
        total_got = 0
        total_want = 0
        for mode in ("train", "val", "test"):
            ds = EchoH5Dataset(DatasetConfig(variant=variant, mode=mode, angle_subset=sub), splits=splits)
            got = len(ds)
            want = expected_n_samples(splits, mode, sub)
            row[mode] = {"got": got, "want": want}
            total_got += got
            total_want += want
            ok &= got == want
        row["total"] = {"got": total_got, "want": total_want}
        ok &= total_got == total_want
        detail[sub] = row
    return Check("licznosc probek dla kazdego --angle-subset", ok, detail)


def check_random_reproducible(splits: Splits, variant: str) -> Check:
    """random_K z tym samym ziarnem -> identyczny podzbior; z innym -> inny.

    Drugi warunek jest tak samo wazny jak pierwszy: gdyby ziarno bylo ignorowane,
    test odtwarzalnosci przechodzilby trywialnie, a warunek D nie mialby zadnego
    losowania.
    """
    detail: dict = {}
    ok = True

    def sig(seed: int, sub: str = "random_4") -> tuple:
        ds = EchoH5Dataset(
            DatasetConfig(variant=variant, mode="train", angle_subset=sub, angle_seed=seed),
            splits=splits,
        )
        return (ds.index_scene.tobytes(), ds.index_row.tobytes(), ds.index_angle.tobytes())

    a1, a2, b = sig(0), sig(0), sig(1)
    same_seed = a1 == a2
    diff_seed = a1 != b
    ok &= same_seed and diff_seed
    detail["random_4"] = {"ten_sam_seed_daje_to_samo": same_seed,
                          "inny_seed_daje_cos_innego": diff_seed}

    # Rozklad katow: przy losowaniu per lokalizacja kazdy z 36 katow powinien
    # wystapic w calym zbiorze mniej wiecej rownie czesto. Silna nierownowaga
    # oznaczalaby blad w hashowaniu ziarna (np. staly RNG dla kazdej lokalizacji).
    ds = EchoH5Dataset(
        DatasetConfig(variant=variant, mode="train", angle_subset="random_4", angle_seed=0),
        splits=splits,
    )
    counts = np.bincount(ds.index_angle.astype(np.int64) // angles_mod.ANGLE_STEP_DEG,
                         minlength=angles_mod.N_ANGLES)
    expected = len(ds) / angles_mod.N_ANGLES
    # Losowanie bez zwracania K z 36 na lokalizacje -> licznik kata to suma
    # zmiennych Bernoulliego(K/36); 5 sigma to bardzo luzny prog na 36 komorek.
    sd = math.sqrt(len(ds) / angles_mod.N_ANGLES * (1 - 4 / 36))
    max_dev = float(np.max(np.abs(counts - expected)))
    balanced = max_dev < 5 * sd
    ok &= balanced
    detail["rozklad_katow_random_4"] = {
        "oczekiwany_licznik": round(expected, 1),
        "min": int(counts.min()), "max": int(counts.max()),
        "max_odchylka": round(max_dev, 1), "prog_5sigma": round(5 * sd, 1),
        "zrownowazony": balanced,
    }
    return Check("random_K odtwarzalny i zrownowazony", ok, detail)


def check_splits_disjoint(splits: Splits) -> Check:
    """Rozlacznosc PO LOKALIZACJACH oraz zgodnosc ze skladem scen Paridy."""
    detail: dict = {}
    ok = True

    def keys(mode: str) -> set[tuple[str, int]]:
        return {(s, int(i)) for s, ids in splits.locations(mode).items() for i in ids}

    tr, va, te = keys("train"), keys("val"), keys("test")
    for a, b, nm in ((tr, va, "train-val"), (tr, te, "train-test"), (va, te, "val-test")):
        inter = a & b
        ok &= not inter
        detail[nm] = {"wspolnych_lokalizacji": len(inter),
                      "przyklady": sorted(inter)[:5]}

    # Rozlacznosc scen train vs held-out. Bez tego rozlacznosc lokalizacji nie
    # wystarcza: ta sama scena w train i test to inny, slabszy protokol niz Gao.
    train_scenes = set(splits.scenes("train"))
    heldout = set(splits.scenes("val")) | set(splits.scenes("test"))
    scene_overlap = train_scenes & heldout
    ok &= not scene_overlap
    detail["sceny"] = {
        "train": sorted(train_scenes),
        "heldout": sorted(heldout),
        "wspolne": sorted(scene_overlap),
        "zgodne_z_base_options": sorted(heldout) == sorted(HELDOUT_SCENES),
        "pokrycie_wszystkich_18": sorted(train_scenes | heldout) == sorted(ALL_SCENES),
    }
    ok &= detail["sceny"]["zgodne_z_base_options"] and detail["sceny"]["pokrycie_wszystkich_18"]

    # Kazda lokalizacja held-outu trafia dokladnie raz -- ani nie ginie, ani nie
    # jest liczona dwa razy.
    for scene in HELDOUT_SCENES:
        n_val = len(splits.val.get(scene, []))
        n_test = len(splits.test.get(scene, []))
        overlap = set(splits.val.get(scene, [])) & set(splits.test.get(scene, []))
        detail[f"podzial_{scene}"] = {"val": n_val, "test": n_test, "suma": n_val + n_test,
                                      "wspolne": len(overlap)}
        ok &= not overlap
    return Check("train/val/test rozlaczne po lokalizacjach", ok, detail)


def check_depth_range(variant: str, splits: Splits, *, chunk: int = 512, full: bool = True) -> Check:
    """Pelny skan glebi wzgledem `max_depth = 14.104`.

    DLACZEGO PELNY, a nie na probce. Model liczy `sigmoid(x) * max_depth`, wiec
    jego wyjscie nigdy nie przekroczy 14.104 m. Kazdy piksel prawdy o wiekszej
    glebi wnosi do RMSE staly blad, ktorego zadne uczenie nie usunie, a jego
    udzial trzeba podac w pracy jako ograniczenie -- to liczba do zacytowania,
    wiec nie moze pochodzic z przypadkowej probki.

    Czytamy blokami po `chunk` wierszy: chunking HDF5 to 1 probka, ale odczyt
    blokiem amortyzuje narzut wywolan i dekompresji.
    """
    detail: dict = {}
    ok = True
    total_px = 0
    over_px = 0
    zero_px = 0
    gmax = -np.inf
    gmin = np.inf

    scenes = sorted(set(splits.scenes("train")) | set(splits.scenes("val")) | set(splits.scenes("test")))
    for scene in scenes:
        p = paths.scene_h5(scene, variant)
        with h5py.File(p, "r") as f:
            dset = f["depth"]
            n = dset.shape[0] if full else min(dset.shape[0], 1024)
            s_over = 0
            s_zero = 0
            s_max = -np.inf
            s_min = np.inf
            for i in range(0, n, chunk):
                d = np.asarray(dset[i:i + chunk], dtype=np.float32)
                s_over += int((d > MAX_DEPTH_REPLICA).sum())
                s_zero += int((d == 0).sum())
                s_max = max(s_max, float(d.max()))
                s_min = min(s_min, float(d.min()))
            npx = n * EXPECTED_IMAGE_SHAPE[0] * EXPECTED_IMAGE_SHAPE[1]
        total_px += npx
        over_px += s_over
        zero_px += s_zero
        gmax = max(gmax, s_max)
        gmin = min(gmin, s_min)
        detail[scene] = {
            "probek": n, "max_m": round(s_max, 4), "min_m": round(s_min, 4),
            "pikseli_powyzej_max_depth": s_over,
            "procent_powyzej": round(100.0 * s_over / npx, 6),
            "procent_zerowych": round(100.0 * s_zero / npx, 6),
        }

    detail["_RAZEM"] = {
        "wariant": variant,
        "max_depth_modelu_m": MAX_DEPTH_REPLICA,
        "pikseli": total_px,
        "global_max_m": round(float(gmax), 4),
        "global_min_m": round(float(gmin), 4),
        "pikseli_powyzej_max_depth": over_px,
        "procent_powyzej": round(100.0 * over_px / total_px, 6),
        "procent_zerowych_(maskowanych_w_stracie)": round(100.0 * zero_px / total_px, 6),
        "pelny_skan": full,
    }
    # Sam fakt istnienia pikseli > max_depth nie jest bledem kodu, tylko faktem
    # o danych -- ale musi byc zaraportowany liczbowo, wiec FAIL zapada dopiero
    # przy udziale, ktory realnie zaburza RMSE.
    ok = (100.0 * over_px / total_px) < 0.5
    return Check("zakres glebi wzgledem max_depth", ok, detail)


def check_echo_stats(splits: Splits, variant: str, n_batches: int = 30, batch_size: int = 32) -> Check:
    """Statystyki echa -- kontrola, ze float16 na dysku nie zjadl dynamiki."""
    cfg = DatasetConfig(variant=variant, mode="train", angle_subset="all", augment=False)
    loader, ds = build_dataloader(
        cfg, batch_size=batch_size, num_workers=4, shuffle=True,
        splits=splits, pin_memory=False, drop_last=False,
    )
    mx, mn, tot, cnt, zeros = -np.inf, np.inf, 0.0, 0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        a = batch["audio"]
        mx = max(mx, float(a.max()))
        mn = min(mn, float(a.min()))
        tot += float(a.sum())
        cnt += a.numel()
        zeros += int((a == 0).sum())
    ds.close()
    detail = {
        "max": round(mx, 5), "min": round(mn, 5),
        "srednia": round(tot / cnt, 6),
        "procent_zer": round(100.0 * zeros / cnt, 4),
        "probek": n_batches * batch_size,
    }
    ok = mn >= 0.0 and mx > 0.0  # magnituda STFT jest nieujemna z definicji
    return Check("statystyki echa (magnituda STFT, nieujemna)", ok, detail)


def dump_samples(splits: Splits, variant: str, out_dir: Path, n: int = 6) -> Check:
    """Zrzuca kilka probek jako PNG: spektrogram (2 kanaly) + RGB + glebia."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(variant=variant, mode="test", angle_subset="all",
                        augment=False, image_transform=False)
    ds = EchoH5Dataset(cfg, splits=splits)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    written = []
    for k, idx in enumerate(idxs):
        s = ds[int(idx)]
        scene = ds.scenes[int(s["scene_idx"])]
        loc = int(s["location_id"])
        ang = int(s["angle_deg"])
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
        # log1p tylko do OGLADANIA -- siec dostaje surowa magnitude.
        for ch in (0, 1):
            im = ax[ch].imshow(np.log1p(s["audio"][ch].numpy()), aspect="auto",
                               origin="lower", cmap="magma")
            ax[ch].set_title(f"echo ch{ch} log1p (2,257,166)")
            fig.colorbar(im, ax=ax[ch], fraction=0.046)
        ax[2].imshow(s["img"].permute(1, 2, 0).numpy().astype(np.uint8))
        ax[2].set_title("RGB 128x128")
        ax[2].axis("off")
        d = s["depth"][0].numpy()
        im = ax[3].imshow(d, cmap="viridis", vmin=0, vmax=MAX_DEPTH_REPLICA)
        ax[3].set_title(f"depth [m] max={d.max():.2f}")
        ax[3].axis("off")
        fig.colorbar(im, ax=ax[3], fraction=0.046)
        fig.suptitle(f"{variant} / {scene} / loc {loc} / {ang}°", fontsize=11)
        fig.tight_layout()
        fp = out_dir / f"sample_{k:02d}_{scene}_loc{loc}_{ang:03d}deg.png"
        fig.savefig(fp, dpi=90)
        plt.close(fig)
        written.append(fp.name)
    ds.close()
    return Check("zrzut probek PNG", True, {"katalog": str(out_dir), "pliki": written})


# ----------------------------------------------------------------------- run


def run_all(variant: str = "main", *, full_depth_scan: bool = True,
            out_dir: Path | None = None, quick: bool = False) -> tuple[list[Check], Path]:
    out_dir = out_dir or (paths.VERIFY_DIR / variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(variant=variant)

    checks = [
        check_splits_disjoint(splits),
        check_counts(splits, variant),
        check_random_reproducible(splits, variant),
        check_batch_contract(splits, variant),
        check_finite(splits, variant, n_batches=10 if quick else 40),
        check_echo_stats(splits, variant, n_batches=10 if quick else 30),
        check_depth_range(variant, splits, full=full_depth_scan and not quick),
        dump_samples(splits, variant, out_dir / "samples"),
    ]

    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "split_fingerprint": splits.meta.get("location_fingerprint"),
        "all_passed": all(c.passed for c in checks),
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
    }
    fp = out_dir / "verify_loader.json"
    fp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return checks, fp
