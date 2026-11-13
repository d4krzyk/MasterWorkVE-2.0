#!/usr/bin/env python
"""Diagnostyka galezi pelnego modelu `AudioVisualModel` (warunki A, D, B x ziarna 0, 1, 2).

    # kontrola odtworzenia (A_seed0, tylko przebieg bazowy) -> <out>/check/
    python my-operations/ml/analysis/branch_diagnostics.py --check

    # szybki test sciezki (1-2 wsady jednego modelu) -> <out>/dryrun/
    python my-operations/ml/analysis/branch_diagnostics.py --dry-run

    # pelne uruchomienie (9 modeli) -> <out>/branch_diag_*.json + summary .md/.csv
    python my-operations/ml/analysis/branch_diagnostics.py --conditions A D B --seeds 0 1 2

PYTANIE. Model pelny sklada glebie jako `alpha*audio_depth + (1-alpha)*img_depth`, a
`alpha` liczy siec uwagi z cech obrazu, echa i materialu. Ile kazda galaz waży w wyniku
i czy uwaga (a w niej cecha materialu) robi cokolwiek ponad stala mieszanke?

DLACZEGO FORWARD JEST TU PRZEPISANY. Kod Paridy (`beyond-image-to-depth/`) musi zostac
bajt w bajt nietkniety, a jego `AudioVisualModel.forward` nie pozwala podmienic `alpha`
ani `material_feat`. `_BranchForward` powtarza ten forward linia po linii na TYCH SAMYCH
podsieciach (`model.net_audio`, `model.net_rgbdepth`, `model.net_material`,
`model.net_attention`), z tymi samymi operacjami w tej samej kolejnosci, i dodaje tylko
punkty podmiany. Zgodnosc z oryginalem sprawdza `--dry-run` (roznica wzgledem
`model(dev)` na tych samych wsadach) i obowiazkowa kontrola odtworzenia `eval.json`.

DLACZEGO `evaluate.collect_table`, A NIE WLASNA PETLA. Zbior testowy, maska waznych
pikseli (gt > 0), AMP i agregacja RMSE po pikselach (`SampleStatsCollector` ->
`SampleTable.aggregate`, warstwa `all`) maja byc DOSLOWNIE te same, co w liczbach pracy.
Najpewniejszy sposob to wywolac te sama funkcje: `collect_table` dostaje zamiast modelu
obiekt `_BranchForward`, ktory zwraca `depth_predicted` wybranego wariantu. Kazdy wariant
to osobny przelot po test@36 (6 588 probek). Srednie alpha zbierane sa po drodze, jako
efekt uboczny, na masce tej samej co warstwa `all` (gt > 0, ew. `valid_ref`).

WARIANTY (kazdy = jeden przelot, RMSE `depth_predicted` w warstwie `all`):
  baseline            -- forward bez zmian (+ srednie alpha)
  branch_audio_depth  -- jako predykcja: audio_depth * max_depth (te same piksele wazne)
  branch_img_depth    -- jako predykcja: img_depth * max_depth
  alpha_const_mean    -- alpha := srednia alpha z przebiegu `baseline` TEGO modelu,
                         liczona po pikselach waznych ZBIORU TESTOWEGO test@36 (dwuprzebiegowo)
  alpha_const_0.5     -- alpha := 0.5
  material_zeros      -- material_feat := 0 (+ srednie alpha)
  material_shuffle    -- material_feat permutowane w obrebie wsadu (+ srednie alpha)

Roznice zapisywane jako `X − Y` doslownie (docs/KONWENCJE.md §2): `wariant − baseline`,
ujemne = wariant ma mniejszy blad.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import MAX_DEPTH_REPLICA, DatasetConfig, EchoH5Dataset  # noqa: E402
from ..dataset.splits import load_splits  # noqa: E402
from ..depth_model.evaluate import TEST_SETS, collect_table, load_model  # noqa: E402
from ..depth_model.metrics import EDGE_GRAD_THRESHOLD_M, SampleStatsCollector  # noqa: E402

# Wersja ustawien -- zmienia sie przy KAZDEJ zmianie definicji wariantow, maski,
# ziarna albo agregacji. Trafia do kazdego pliku wynikowego.
SETTINGS_VERSION = "branch_diag_v1 (2026-11-13)"

DEFAULT_OUT = paths.ML_OUTPUTS / "overnight_2026-11"
TEST_LABEL = "test@36"
ANGLE_SUBSET = TEST_SETS[TEST_LABEL]          # "all" -- ten sam slownik, co evaluate.py
N_TEST_EXPECTED = 6588
SHUFFLE_SEED = 12345                          # ziarno permutacji material_feat
CHECK_TOL = 1e-4                              # tolerancja kontroli odtworzenia eval.json
DRYRUN_BATCHES = 2

VARIANTS = ("baseline", "branch_audio_depth", "branch_img_depth", "alpha_const_mean",
            "alpha_const_0.5", "material_zeros", "material_shuffle")
# Warianty, w ktorych alpha pochodzi z sieci uwagi -- tylko dla nich srednie alpha maja sens.
ALPHA_STATS_VARIANTS = ("baseline", "material_zeros", "material_shuffle")


def run_id(cond: str, seed: int) -> str:
    return f"{cond}_seed{seed}"


def eval_json_path(rid: str) -> Path:
    return paths.ML_OUTPUTS / "eval" / rid / "eval.json"


# ------------------------------------------------------------------ forward z podmiana


class _BranchForward:
    """Forward `AudioVisualModel` z punktami podmiany. Wolany przez `collect_table` jak model.

    Kolejnosc i postac operacji jak w `beyond-image-to-depth/models/audioVisual_model.py`;
    caly forward wykonuje sie wewnatrz `torch.amp.autocast`, ktory otwiera `collect_table`,
    wiec typy tensorow sa te same, co w ewaluacji oryginalnej.

    output    -- co zwrocic jako `depth_predicted`: "depth_predicted" | "audio_depth" | "img_depth"
    alpha_const -- None (alpha z sieci) albo liczba rzeczywista (python float, skalar)
    material  -- None | "zeros" | "shuffle"
    """

    def __init__(self, model, *, output: str = "depth_predicted", alpha_const: float | None = None,
                 material: str | None = None, shuffle_seed: int = SHUFFLE_SEED,
                 collect_alpha: bool = False):
        assert output in ("depth_predicted", "audio_depth", "img_depth")
        assert material in (None, "zeros", "shuffle")
        self.m = model
        self.max_depth = float(model.opt.max_depth)
        self.output = output
        self.alpha_const = None if alpha_const is None else float(alpha_const)
        self.material = material
        self.collect_alpha = collect_alpha
        # Generator tworzony na nowo w KAZDYM przelocie, z tym samym ziarnem; loader ma
        # shuffle=False, wiec wsady (i ich permutacje) sa identyczne dla wszystkich modeli.
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(shuffle_seed))
        # akumulatory (float64)
        self.alpha_sum_valid = 0.0
        self.n_valid = 0
        self.alpha_sum_all = 0.0
        self.n_all = 0
        self.n_batches = 0
        self.n_samples = 0
        self.perm_fixed_points = 0
        self.batch_sizes: list[int] = []

    def eval(self):  # zgodnosc z interfejsem modelu (collect_table go nie wola, ale na wszelki wypadek)
        return self

    def __call__(self, inp: dict) -> dict:
        m = self.m
        rgb_input = inp["img"]
        audio_input = inp["audio"]
        depth_gt = inp["depth"]

        audio_depth, audio_feat = m.net_audio(audio_input)
        img_depth, img_feat = m.net_rgbdepth(rgb_input)
        material_class, material_feat = m.net_material(rgb_input)
        audio_feat = audio_feat.repeat(1, 1, img_feat.shape[-2], img_feat.shape[-1])  # tile jak u Paridy

        b = int(material_feat.shape[0])
        if self.material == "zeros":
            material_feat = torch.zeros_like(material_feat)
        elif self.material == "shuffle":
            # Permutacja w obrebie wsadu; MOZE miec punkty stale (probka dostaje wlasna cecha
            # materialu) -- ich liczba jest zliczana i raportowana. Wsad 1-elementowy = identycznosc.
            perm = torch.randperm(b, generator=self.gen)
            self.perm_fixed_points += int((perm == torch.arange(b)).sum())
            material_feat = material_feat[perm.to(material_feat.device)]

        alpha, _ = m.net_attention(img_feat, audio_feat, material_feat)
        if alpha.shape != depth_gt.shape:
            raise RuntimeError(f"ksztalt alpha {tuple(alpha.shape)} != ksztalt glebi {tuple(depth_gt.shape)}")

        if self.collect_alpha:
            # Maska jak warstwa `all` w SampleStatsCollector: gt > 0 (i valid_ref, jesli jest).
            valid = depth_gt > 0
            if inp.get("valid_ref") is not None:
                valid = valid & inp["valid_ref"].to(valid.device).bool()
            a = alpha.detach().double()
            self.alpha_sum_valid += float(a[valid].sum())
            self.n_valid += int(valid.sum())
            self.alpha_sum_all += float(a.sum())
            self.n_all += int(a.numel())

        a_used = alpha if self.alpha_const is None else self.alpha_const
        depth_prediction = ((a_used * audio_depth) + ((1 - a_used) * img_depth))

        self.n_batches += 1
        self.n_samples += b
        self.batch_sizes.append(b)

        chosen = {"depth_predicted": depth_prediction, "audio_depth": audio_depth,
                  "img_depth": img_depth}[self.output]
        return {"depth_predicted": chosen * self.max_depth, "depth_gt": depth_gt,
                "attention": alpha, "material_class": material_class}

    def alpha_stats(self) -> dict:
        return {
            "alpha_mean_valid_px": self.alpha_sum_valid / self.n_valid if self.n_valid else float("nan"),
            "alpha_mean_all_px": self.alpha_sum_all / self.n_all if self.n_all else float("nan"),
            "n_valid_px": self.n_valid, "n_all_px": self.n_all,
        }


# ------------------------------------------------------------------ jeden model


def _variant_kwargs(name: str, alpha_mean: float | None) -> dict:
    return {
        "baseline": dict(collect_alpha=True),
        "branch_audio_depth": dict(output="audio_depth"),
        "branch_img_depth": dict(output="img_depth"),
        "alpha_const_mean": dict(alpha_const=alpha_mean),
        "alpha_const_0.5": dict(alpha_const=0.5),
        "material_zeros": dict(material="zeros", collect_alpha=True),
        "material_shuffle": dict(material="shuffle", collect_alpha=True),
    }[name]


def _run_pass(fwd: _BranchForward, cond, splits, args, device) -> tuple[dict, float]:
    """Jeden przelot po test@36 przez `evaluate.collect_table` -- ta sama sciezka, co eval.json."""
    t0 = time.perf_counter()
    tab, info = collect_table(
        fwd, cond, splits, angle_subset=ANGLE_SUBSET, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=not args.no_amp,
        edge_threshold=EDGE_GRAD_THRESHOLD_M, mask_variant=None, shuffle_echo_seed=None)
    secs = time.perf_counter() - t0
    if info["n_samples"] != N_TEST_EXPECTED or len(tab) != N_TEST_EXPECTED:
        raise RuntimeError(f"test@36 ma {info['n_samples']}/{len(tab)} probek, oczekiwano {N_TEST_EXPECTED}")
    agg = tab.aggregate()
    return {"RMSE_all": agg["all"]["RMSE"], "overall": agg, "collect_info": info}, secs


def _load_reference(rid: str) -> dict:
    p = eval_json_path(rid)
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("intersection_mask_variant") is not None or d.get("shuffled_echo"):
        raise ValueError(f"{p}: to nie jest zwykla ewaluacja (maska/permutacja echa)")
    return {"path": str(p), "RMSE_all": float(d["test_sets"][TEST_LABEL]["overall"]["all"]["RMSE"]),
            "created_utc": d.get("created_utc"), "condition": d.get("condition"), "seed": d.get("seed")}


def _load(cond_id: str, seed: int, device, splits):
    rid = run_id(cond_id, seed)
    run_dir = paths.RUNS_DIR / rid                       # jawna nazwa, bez globow
    model, cond, cfg, loaded = load_model(run_dir, device, splits)
    if cond.id != cond_id or int(cfg["spec"]["seed"]) != seed:
        raise ValueError(f"{run_dir}: config.json mowi {cond.id}/seed{cfg['spec']['seed']}")
    for k in ("net_audio", "net_rgbdepth", "net_material", "net_attention"):
        if not hasattr(model, k):
            raise TypeError(f"{rid}: model {type(model).__name__} nie jest AudioVisualModel (brak {k})")
    if abs(float(model.opt.max_depth) - MAX_DEPTH_REPLICA) > 1e-12:
        raise ValueError(f"{rid}: max_depth {model.opt.max_depth} != {MAX_DEPTH_REPLICA}")
    if model.training:
        raise RuntimeError(f"{rid}: model nie jest w trybie eval()")
    weights = {k: str((run_dir / v).resolve()) for k, v in loaded.items()}
    return rid, model, cond, cfg, weights


def _settings(args) -> dict:
    return {
        "settings_version": SETTINGS_VERSION,
        "test_set": TEST_LABEL, "angle_subset": ANGLE_SUBSET, "n_samples_expected": N_TEST_EXPECTED,
        "amp": not args.no_amp, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "edge_threshold_m_per_px": EDGE_GRAD_THRESHOLD_M, "max_depth": MAX_DEPTH_REPLICA,
        "metric": "RMSE [m] zagregowane po pikselach (SampleTable.aggregate), warstwa `all`, maska gt > 0",
        "material_shuffle": {
            "seed": SHUFFLE_SEED,
            "opis": "torch.Generator(cpu) z ziarnem 12345 tworzony od nowa w kazdym przelocie; "
                    "dla kazdego wsadu po kolei torch.randperm(rozmiar_wsadu); loader shuffle=False, "
                    "wiec permutacje sa identyczne dla wszystkich modeli przy tym samym --batch-size. "
                    "Permutacja MOZE miec punkty stale (probka zachowuje wlasna cecha materialu); "
                    "ostatni niepelny wsad tez jest permutowany (wsad 1-elementowy = identycznosc).",
        },
        "alpha_const_mean": "alpha := srednia alpha (po pikselach waznych, gt > 0) z przebiegu "
                            "`baseline` TEGO SAMEGO modelu na ZBIORZE TESTOWYM test@36 (dwuprzebiegowo) "
                            "-- stala wyznaczona na danych, na ktorych jest oceniana.",
        "difference_convention": "X − Y doslownie (docs/KONWENCJE.md §2); wariant − baseline; ujemne = lepiej",
        "forward": "reimplementacja AudioVisualModel.forward na podsieciach modelu (kod Paridy nietkniety)",
    }


def diagnose(cond_id: str, seed: int, args, device, splits, *, check_only: bool = False,
             require_match: bool = False) -> tuple[dict, bool]:
    """Wszystkie warianty dla jednego modelu. Zwraca (wynik, czy_kontrola_ok)."""
    t_model = time.perf_counter()
    rid, model, cond, cfg, weights = _load(cond_id, seed, device, splits)
    ref = _load_reference(rid)
    out: dict = {
        "run_id": rid, "condition": cond_id, "seed": seed, "model": cond.model,
        "geometry": cond.geometry, "angle_subset_train": cond.angle_subset,
        "fast_bilinear": bool(cfg.get("spec", {}).get("extra", {}).get("fast_bilinear", False)),
        "weights": weights, "split_fingerprint": splits.meta.get("location_fingerprint"),
        "settings": _settings(args), "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_eval_json": ref, "variants": {}, "seconds_per_pass": {},
    }
    print(f"\n== {rid}  (warunek {cond.id}, {cond.model}, katy treningowe {cond.angle_subset})")

    alpha_mean = None
    ok = True
    for name in VARIANTS:
        fwd = _BranchForward(model, **_variant_kwargs(name, alpha_mean))
        res, secs = _run_pass(fwd, cond, splits, args, device)
        if fwd.collect_alpha:
            res["alpha"] = fwd.alpha_stats()
        if name == "material_shuffle":
            res["perm_fixed_points"] = fwd.perm_fixed_points
            res["perm_n_samples"] = fwd.n_samples
            res["perm_n_batches"] = fwd.n_batches
            res["perm_last_batch_size"] = fwd.batch_sizes[-1] if fwd.batch_sizes else 0
        if name == "alpha_const_mean":
            res["alpha_const"] = alpha_mean
        out["variants"][name] = res
        out["seconds_per_pass"][name] = round(secs, 2)
        extra = ""
        if "alpha" in res:
            extra = (f"  alpha(wazne) {res['alpha']['alpha_mean_valid_px']:.5f}"
                     f"  alpha(wszystkie) {res['alpha']['alpha_mean_all_px']:.5f}")
        print(f"   {name:20s} RMSE {res['RMSE_all']:.6f}{extra}   ({secs:.1f} s)")

        if name == "baseline":
            alpha_mean = res["alpha"]["alpha_mean_valid_px"]
            diff = res["RMSE_all"] - ref["RMSE_all"]
            ok = abs(diff) <= CHECK_TOL
            out["check"] = {"baseline_RMSE": res["RMSE_all"], "eval_json_RMSE": ref["RMSE_all"],
                            "baseline − eval_json": diff, "tolerance": CHECK_TOL, "passed": ok}
            print(f"   kontrola: baseline − eval.json = {res['RMSE_all']:.8f} − {ref['RMSE_all']:.8f}"
                  f" = {diff:+.2e}  ({'OK' if ok else 'NIEZGODNOSC'}, tol {CHECK_TOL:g})")
            if check_only or (require_match and not ok):
                break

    base = out["variants"]["baseline"]["RMSE_all"]
    out["differences"] = {f"RMSE({v}) − RMSE(baseline)": out["variants"][v]["RMSE_all"] - base
                          for v in VARIANTS[1:] if v in out["variants"]}
    out["seconds_total"] = round(time.perf_counter() - t_model, 1)
    del model
    torch.cuda.empty_cache()
    return out, ok


# ------------------------------------------------------------------ dry-run


@torch.no_grad()
def dry_run(args, device, splits) -> dict:
    """1-2 wsady jednego modelu: ksztalty, brak wyjatkow, zgodnosc z `model(dev)`.

    Petla jest kopia `collect_table` z limitem wsadow (tamta funkcja nie ma limitu) --
    to wylacznie test sciezki; liczby do pracy idą zawsze przez `collect_table`.
    """
    cid, seed = args.conditions[0], args.seeds[0]
    rid, model, cond, cfg, weights = _load(cid, seed, device, splits)
    ds = EchoH5Dataset(DatasetConfig(variant=cond.geometry, mode="test", angle_subset=ANGLE_SUBSET,
                                     angle_seed=cond.angle_seed, augment=False, mask_variant=None,
                                     mask_mode="intersection", shuffle_echo_seed=None), splits=splits)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    batches = []
    for i, batch in enumerate(loader):
        if i >= DRYRUN_BATCHES:
            break
        batches.append({k: v.to(device, non_blocking=True) for k, v in batch.items()})
    ds.close()
    amp = not args.no_amp

    # 1) zgodnosc reimplementacji z oryginalnym forwardem, wsad po wsadzie
    fwd = _BranchForward(model)
    max_abs = {}
    for k in ("depth_predicted",):
        diffs = []
        for dev in batches:
            with torch.amp.autocast("cuda", enabled=amp):
                o_ref = model(dev)
                o_new = fwd(dev)
            diffs.append(float((o_ref[k].float() - o_new[k].float()).abs().max()))
            for kk in ("audio_depth", "img_depth"):
                f2 = _BranchForward(model, output=kk)
                with torch.amp.autocast("cuda", enabled=amp):
                    o2 = f2(dev)
                max_abs[kk] = max(max_abs.get(kk, 0.0),
                                  float((o_ref[kk].float() - o2["depth_predicted"].float()).abs().max()))
        max_abs[k] = max(diffs)
    print(f"  max |model(dev) − reimplementacja|: {max_abs}")

    # 2) wszystkie warianty na tych wsadach, ta sama agregacja
    res = {}
    alpha_mean = None
    for name in VARIANTS:
        f = _BranchForward(model, **_variant_kwargs(name, alpha_mean))
        col = SampleStatsCollector(ds.scenes, EDGE_GRAD_THRESHOLD_M)
        shapes = None
        for dev in batches:
            with torch.amp.autocast("cuda", enabled=amp):
                o = f(dev)
            shapes = {k: list(v.shape) for k, v in o.items()}
            col.update(o["depth_predicted"].float(), o["depth_gt"].float(),
                       scene_idx=dev["scene_idx"], location_id=dev["location_id"],
                       angle_deg=dev["angle_deg"], valid_ref=dev.get("valid_ref"))
        r = {"RMSE_all": col.table().aggregate()["all"]["RMSE"], "shapes": shapes}
        if f.collect_alpha:
            r["alpha"] = f.alpha_stats()
        if name == "material_shuffle":
            r["perm_fixed_points"] = f.perm_fixed_points
        if name == "baseline":
            alpha_mean = r["alpha"]["alpha_mean_valid_px"]
        if name == "alpha_const_mean":
            r["alpha_const"] = alpha_mean
        res[name] = r
        print(f"   {name:20s} RMSE {r['RMSE_all']:.6f}"
              + (f"  alpha(wazne) {r['alpha']['alpha_mean_valid_px']:.5f}" if "alpha" in r else ""))
    return {"run_id": rid, "weights": weights, "n_batches": len(batches),
            "n_samples": sum(int(d["img"].shape[0]) for d in batches),
            "max_abs_diff_vs_model_forward": max_abs, "variants": res,
            "settings": _settings(args), "created_utc": datetime.now(timezone.utc).isoformat()}


# ------------------------------------------------------------------ zestawienie


def _msd(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else float("nan"))


def _quantities(r: dict) -> dict[str, float]:
    v = r["variants"]
    q = {
        "RMSE(baseline)": v["baseline"]["RMSE_all"],
        "RMSE(branch_audio_depth)": v["branch_audio_depth"]["RMSE_all"],
        "RMSE(branch_img_depth)": v["branch_img_depth"]["RMSE_all"],
        "alpha_mean_valid_px(baseline)": v["baseline"]["alpha"]["alpha_mean_valid_px"],
        "alpha_mean_all_px(baseline)": v["baseline"]["alpha"]["alpha_mean_all_px"],
        "RMSE(alpha_const_mean)": v["alpha_const_mean"]["RMSE_all"],
        "RMSE(alpha_const_0.5)": v["alpha_const_0.5"]["RMSE_all"],
        "RMSE(material_zeros)": v["material_zeros"]["RMSE_all"],
        "RMSE(material_shuffle)": v["material_shuffle"]["RMSE_all"],
        "alpha_mean_valid_px(material_zeros)": v["material_zeros"]["alpha"]["alpha_mean_valid_px"],
        "alpha_mean_all_px(material_zeros)": v["material_zeros"]["alpha"]["alpha_mean_all_px"],
        "alpha_mean_valid_px(material_shuffle)": v["material_shuffle"]["alpha"]["alpha_mean_valid_px"],
        "alpha_mean_all_px(material_shuffle)": v["material_shuffle"]["alpha"]["alpha_mean_all_px"],
    }
    q.update(r["differences"])
    q["alpha_mean_valid_px(material_zeros) − alpha_mean_valid_px(baseline)"] = (
        q["alpha_mean_valid_px(material_zeros)"] - q["alpha_mean_valid_px(baseline)"])
    q["alpha_mean_valid_px(material_shuffle) − alpha_mean_valid_px(baseline)"] = (
        q["alpha_mean_valid_px(material_shuffle)"] - q["alpha_mean_valid_px(baseline)"])
    q["RMSE(baseline) − RMSE(eval.json)"] = r["check"]["baseline − eval_json"]
    return q


def write_summary(results: list[dict], out_dir: Path, conditions, seeds, settings: dict) -> None:
    rows = []
    for c in conditions:
        rs = [r for r in results if r["condition"] == c]
        rs.sort(key=lambda r: r["seed"])
        qs = [_quantities(r) for r in rs]
        for name in qs[0]:
            vals = [q[name] for q in qs]
            m, sd = _msd(vals)
            rows.append({"condition": c, "quantity": name, "mean": m, "sd": sd, "n_seeds": len(vals),
                         **{f"seed{r['seed']}": q[name] for r, q in zip(rs, qs)}})

    seed_cols = [f"seed{s}" for s in seeds]
    with open(out_dir / "branch_diagnostics_summary.csv", "x", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "quantity", "mean", "sd", "n_seeds", *seed_cols])
        w.writeheader()
        for row in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})

    lines = [
        "# Diagnostyka galezi modelu pelnego (AudioVisualModel)",
        "",
        f"Wygenerowano: {datetime.now(timezone.utc).isoformat()} -- `my-operations/ml/analysis/branch_diagnostics.py`",
        f"Wersja ustawien: `{settings['settings_version']}`. Zbior: {TEST_LABEL} ({N_TEST_EXPECTED} probek), "
        f"AMP={'tak' if settings['amp'] else 'nie'}, batch {settings['batch_size']}.",
        "Metryka: RMSE [m] zagregowane po pikselach, warstwa `all` (gt > 0), jak w `evaluate.py`. "
        f"Srednia ± sd (ddof=1) po ziarnach {list(seeds)}.",
        "Roznice: `X − Y` doslownie (docs/KONWENCJE.md §2); ujemne = mniejszy blad.",
        "`alpha_const_mean`: alpha := srednia alpha z przebiegu bazowego TEGO modelu na zbiorze testowym "
        "test@36 (stala wyznaczona na danych oceny).",
        f"`material_shuffle`: permutacja w obrebie wsadu, ziarno {SHUFFLE_SEED}; moze miec punkty stale "
        "(liczba w plikach JSON per model).",
        "",
    ]
    for c in conditions:
        lines += [f"## Warunek {c}", "", "| wielkosc | srednia ± sd | " + " | ".join(seed_cols) + " |",
                  "|---|---|" + "---|" * len(seed_cols)]
        for row in rows:
            if row["condition"] != c:
                continue
            per = " | ".join(f"{row.get(s, float('nan')):.6f}" for s in seed_cols)
            lines.append(f"| `{row['quantity']}` | {row['mean']:.6f} ± {row['sd']:.6f} | {per} |")
        lines.append("")
    (out_dir / "branch_diagnostics_summary.md").open("x", encoding="utf-8").write("\n".join(lines))


# ------------------------------------------------------------------ sterowanie


def _write_new(path: Path, obj: dict) -> None:
    """Zapis fail-if-exists: nigdy nie nadpisuje istniejacego pliku."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def _gpu_busy_guard() -> None:
    """Nie startujemy, jesli na GPU juz cos liczy (ten skrypt ma byc jedynym procesem GPU)."""
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        busy = [ln for ln in r.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        busy = []
    if busy:
        raise SystemExit("BLAD: na GPU dzialaja juz procesy obliczeniowe:\n  " + "\n  ".join(busy))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=["A", "D", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="tylko kontrola odtworzenia eval.json dla A_seed0 (zapis do <out>/check/)")
    ap.add_argument("--dry-run", action="store_true", help="1-2 wsady jednego modelu (zapis do <out>/dryrun/)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args(argv)

    out_dir: Path = args.out.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _gpu_busy_guard()
    device = torch.device("cuda")
    splits = load_splits()

    if args.dry_run:
        t0 = time.perf_counter()
        res = dry_run(args, device, splits)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        p = out_dir / "dryrun" / f"dryrun_{res['run_id']}_{stamp}.json"
        _write_new(p, res)
        print(f"\nzapisano: {p}  ({res['seconds']} s)")
        return 0

    if args.check:
        t0 = time.perf_counter()
        res, ok = diagnose("A", 0, args, device, splits, check_only=True)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        p = out_dir / "check" / f"check_A_seed0_{stamp}.json"
        _write_new(p, res)
        print(f"\nKONTROLA {'ZALICZONA' if ok else 'NIEZALICZONA'}; zapisano: {p}  ({res['seconds']} s)")
        return 0 if ok else 3

    # --- pelne uruchomienie: najpierw sprawdz, ze nic nie zostanie nadpisane
    models = [(c, s) for c in args.conditions for s in args.seeds]
    targets = [out_dir / f"branch_diag_{run_id(c, s)}.json" for c, s in models]
    targets += [out_dir / "branch_diagnostics_summary.md", out_dir / "branch_diagnostics_summary.csv",
                out_dir / "branch_diag_CHECK_FAILED.json"]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise SystemExit("BLAD: pliki wynikowe juz istnieja (fail-if-exists):\n  " + "\n  ".join(existing))
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = []
    for i, (c, s) in enumerate(models):
        res, ok = diagnose(c, s, args, device, splits, require_match=(i == 0))
        if i == 0 and not ok:
            p = out_dir / "branch_diag_CHECK_FAILED.json"
            _write_new(p, res)
            print(f"\nKONTROLA ODTWORZENIA NIEZALICZONA dla {run_id(c, s)} -- przerywam. Zapisano: {p}")
            return 3
        _write_new(out_dir / f"branch_diag_{run_id(c, s)}.json", res)
        results.append(res)

    print("\nbaseline − eval.json (informacyjnie):")
    for r in results:
        ch = r["check"]
        print(f"   {r['run_id']:10s} {ch['baseline_RMSE']:.8f} − {ch['eval_json_RMSE']:.8f} = "
              f"{ch['baseline − eval_json']:+.2e}")
    write_summary(results, out_dir, args.conditions, args.seeds, _settings(args))
    print(f"\nzapisano: {out_dir}/branch_diag_*.json, branch_diagnostics_summary.{{md,csv}}  "
          f"({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
