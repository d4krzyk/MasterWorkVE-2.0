#!/usr/bin/env python
"""Galeria sieci TYLKO-ECHO (EA, ED, EB, ESE) do rozdzialu 5.3 -- bez treningu.

    python my-operations/ml/analysis/depth_gallery_echo_53.py [--skip-baseline]

Wyjscie (outputs/ml/figures/gallery/, nic nie jest nadpisywane poza wlasnymi plikami):
    depth_gallery_echo_53_glowna.png     3 wiersze (mediana RMSE EB), 15,5 x ~9 cm
    depth_gallery_echo_53_dodatkowa.png  6 wierszy (mediana + 90. percentyl), 15,5 x ~15 cm
    depth_gallery_echo_53_wybor.md       tabela wyboru, regula, kontrola 1e-3
    depth_gallery_echo_53_podpis.md      fakty do podpisu
    depth_gallery_echo_53_baseline.md    (opcjonalnie, CPU) stale mapy i zroznicowanie predykcji

PRZEBIEGI (jawnie, ziarno 0): EA_seed0, ED_seed0, EB_seed0, ESE_seed0. ESE jest oceniane z
PRAWDZIWYM echem (jak w eval.json: shuffled_echo = False); permutowane echo dotyczy tylko treningu.

REGULA WYBORU: z tabeli per probka eval/EB_seed0/samples_test@36.npz, RMSE probki =
sqrt(sum_sq / n_px). W kazdej scenie testowej: probka o RMSE najblizszym MEDIANIE sceny
(wersja glowna) i 90. PERCENTYLOWI (dodatkowa); remis -> pierwsza w tabeli. Brak tabeli ->
losowanie z ziarnem 20261113. KONTROLA: RMSE z modeli = RMSE z ich tabel per probka (1e-3).

Rysowanie, preprocessing i tryb eval wspolne z `depth_gallery_v2.py` (import, bez zmian).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

import json  # noqa: E402

from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import MAX_DEPTH_REPLICA, DatasetConfig, EchoH5Dataset  # noqa: E402
from ..dataset.splits import load_splits  # noqa: E402
from ..depth_model.evaluate import load_model  # noqa: E402
from .depth_gallery_v2 import DEPTH_VMAX, ERR_VMAX, OUT, SEED, TOL, pl, predict, render  # noqa: E402

SELECTION_SEED = 20261113
RUNS = {"EA": "EA_seed0", "ED": "ED_seed0", "EB": "EB_seed0", "ESE": "ESE_seed0"}
TABLES = {k: paths.ML_OUTPUTS / "eval" / v / "samples_test@36.npz" for k, v in RUNS.items()}
EVAL_JSON = {k: [paths.ML_OUTPUTS / "eval" / f"{k}_seed{s}" / "eval.json" for s in (0, 1, 2)]
             for k in RUNS}
CARDINAL = (0, 90, 180, 270)
BASELINE_BUDGET_S = 15 * 60


def dist_cardinal(a: int) -> int:
    return int(min(min(abs(a - c), 360 - abs(a - c)) for c in CARDINAL))


def table_rmse(path):
    z = np.load(path, allow_pickle=True)
    n_px = z["all.n_px"].astype(float)
    return z, np.sqrt(z["all.sq"] / np.maximum(n_px, 1)), n_px


def select_rows(scenes):
    z, rmse, n_px = table_rmse(TABLES["EB"])
    names = [str(s) for s in z["scene_names"]]
    assert names == list(scenes), (names, scenes)
    rows = []
    for si, name in enumerate(names):
        m = np.flatnonzero((z["scene_idx"] == si) & (n_px > 0))
        for etyk, q in (("typowy", 50), ("trudny", 90)):
            target = float(np.percentile(rmse[m], q))
            j = m[int(np.argmin(np.abs(rmse[m] - target)))]
            rows.append(dict(scena=name, loc=int(z["location_id"][j]), kat=int(z["angle_deg"][j]),
                             etykieta=etyk, q=q, cel=target, rmse_tab_EB=float(rmse[j]),
                             si=si, n_scene=int(m.size)))
    return rows


def fallback_rows(ds):
    rng = np.random.default_rng(SELECTION_SEED)
    rows = []
    for si, name in enumerate(ds.scenes):
        pool = np.flatnonzero(ds.index_scene == si)
        for etyk in ("typowy", "trudny"):
            j = int(rng.choice(pool))
            rows.append(dict(scena=name, loc=int(ds.index_loc[j]), kat=int(ds.index_angle[j]),
                             etykieta=etyk, q=None, cel=float("nan"), rmse_tab_EB=float("nan"),
                             si=si, n_scene=int(pool.size)))
    return rows


def eval_mean(k):
    v = [float(json.loads(p.read_text(encoding="utf-8"))["test_sets"]["test@36"]["overall"]["all"]["RMSE"])
         for p in EVAL_JSON[k]]
    return float(np.mean(v)), float(np.std(v, ddof=1)), v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args(argv)
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"urzadzenie: {device}, AMP: {amp}, ziarno: {SEED}")
    splits = load_splits()
    ds = EchoH5Dataset(DatasetConfig(mode="test", angle_subset="all", augment=False), splits=splits)

    have_tables = all(p.exists() for p in TABLES.values())
    if have_tables:
        rows = select_rows(ds.scenes); regula = "tabela per próbka eval/EB_seed0/samples_test@36.npz"
    else:
        print("brak tabel per probka -> losowanie z ziarnem", SELECTION_SEED)
        rows = fallback_rows(ds); regula = f"losowanie z ziarnem {SELECTION_SEED} (brak tabel)"

    key = (ds.index_scene.astype(np.int64) * 10**9 + ds.index_loc.astype(np.int64) * 10**4
           + ds.index_angle.astype(np.int64))
    idx = []
    for r in rows:
        hit = np.flatnonzero(key == ds.scenes.index(r["scena"]) * 10**9 + r["loc"] * 10**4 + r["kat"])
        assert hit.size == 1, (r, hit)
        idx.append(int(hit[0]))
    items = [ds[i] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]) for k in ("img", "depth", "audio")}
    gt = batch["depth"].numpy()[:, 0]
    valid = gt > 0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    rgb = [np.clip(it["img"].numpy().transpose(1, 2, 0) * std + mean, 0, 1) for it in items]

    preds, srm = {}, {}
    for m, run in RUNS.items():
        model, cond, _, loaded = load_model(paths.RUNS_DIR / run, device, splits)
        assert cond.geometry == "main"
        preds[m] = predict(model, batch, device, amp)       # prawdziwe echo (ds bez shuffle_echo_seed)
        print(f"  {run}: wagi {sorted(loaded.values())}")
        del model
        srm[m] = np.array([np.sqrt(((preds[m][i][valid[i]] - gt[i][valid[i]]) ** 2).mean())
                           for i in range(len(rows))])
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Kontrola 1e-3 wzgledem tabel per probka KAZDEGO modelu.
    bad = []
    for r_i, r in enumerate(rows):
        for m in RUNS:
            r[f"mod_{m}"] = float(srm[m][r_i])
    for m in RUNS:
        if not have_tables:
            break
        z, rm, _ = table_rmse(TABLES[m])
        for i, r in enumerate(rows):
            j = np.flatnonzero((z["scene_idx"] == r["si"]) & (z["location_id"] == r["loc"])
                               & (z["angle_deg"] == r["kat"]))
            assert j.size == 1
            r[f"tab_{m}"] = float(rm[j[0]]); r[f"mod_{m}"] = float(srm[m][i])
            r[f"d_{m}"] = abs(r[f"tab_{m}"] - r[f"mod_{m}"])
            if r[f"d_{m}"] > TOL:
                bad.append((m, r["scena"], r["loc"], r["kat"], r[f"d_{m}"]))
    for r in rows:
        r["dist_kard"] = dist_cardinal(r["kat"])
    print("kontrola 1e-3:", "OK" if not bad else f"NIEZGODNE {bad}",
          f"(max |d| = {max((r.get(f'd_{m}', 0) for r in rows for m in RUNS)):.2e})")

    ev = {k: eval_mean(k) for k in RUNS}
    hd = lambda k: f"RMSE\ntest@36\n{pl(ev[k][0], 3)} m"  # noqa: E731
    C = {
        "rgb": dict(kind="rgb", head="RGB\n(wejście)"),
        "gt": dict(kind="gt", head="prawda"),
        "EA": dict(kind="pred", m="EA", head="EA\n4 orient.\nkardynalne\n" + hd("EA")),
        "ED": dict(kind="pred", m="ED", head="ED\n4 orient.\nlosowane\n" + hd("ED")),
        "EB": dict(kind="pred", m="EB", head="EB\n36 orient.\n\n" + hd("EB")),
        "ESE": dict(kind="pred", m="ESE", head="ESE\n36 orient.\ninne echo\n" + hd("ESE")),
        "errA": dict(kind="err", m="EA", head="|błąd|\nEA"),
        "errB": dict(kind="err", m="EB", head="|błąd|\nEB"),
    }
    order = [C[k] for k in ("rgb", "gt", "EA", "ED", "EB", "ESE", "errA", "errB")]
    OUT.mkdir(parents=True, exist_ok=True)
    main_i = [i for i, r in enumerate(rows) if r["etykieta"] == "typowy"]

    def sub(ii):
        return ([rows[i] for i in ii], [rgb[i] for i in ii], gt[ii],
                {m: p[ii] for m, p in preds.items()}, {m: s[ii] for m, s in srm.items()})
    r_, g_, gt_, p_, s_ = sub(main_i)
    render(order, r_, g_, gt_, p_, s_, None, OUT / "depth_gallery_echo_53_glowna.png",
           height_gap_in=0.17, label_w_in=0.78)
    render(order, rows, rgb, gt, preds, srm, None, OUT / "depth_gallery_echo_53_dodatkowa.png",
           height_gap_in=0.15, label_w_in=0.78)

    base = None
    if not args.skip_baseline:
        base = baseline(ds, splits, ev)
    ds.close()
    write_md(rows, regula, ev, have_tables, bad, base)
    print("zapisano:", *[p.name for p in sorted(OUT.glob("depth_gallery_echo_53_*"))])
    return 1 if bad else 0


# ------------------------------------------------------------------ baseline (opcjonalnie, CPU)


def baseline(ds_test, splits, ev):
    t0 = time.time()
    dev = torch.device("cpu")
    ds_tr = EchoH5Dataset(DatasetConfig(variant="main", mode="train", angle_subset="all", augment=False),
                          splits=splits)
    n_tr, n_te = len(ds_tr), len(ds_test)
    print(f"baseline: train {n_tr} probek, test {n_te}")
    nw = 8
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=False, num_workers=nw)
    s = c = None
    for bi, b in enumerate(dl):
        d = b["depth"][:, 0].double()
        v = (d > 0).double()
        s = (d * v).sum(0) if s is None else s + (d * v).sum(0)
        c = v.sum(0) if c is None else c + v.sum(0)
        if bi == 20:
            est = (time.time() - t0) / 21 * (len(dl) + n_te / 64 * 1.5)
            print(f"  szacowany czas calosci: {est / 60:.1f} min")
            if est > BASELINE_BUDGET_S:
                ds_tr.close()
                print("  > 15 min -- pomijam baseline")
                return {"skipped": True, "est_min": est / 60}
    ds_tr.close()
    glob_mean = float(s.sum() / c.sum())
    pix_mean = (s / c.clamp(min=1)).float()
    pix_mean = torch.where(c > 0, pix_mean, torch.tensor(glob_mean, dtype=torch.float32))
    print(f"  srednia globalna (train, wazne piksele): {glob_mean:.4f} m, t = {time.time() - t0:.0f} s")

    models = {}
    for m in ("EB", "ESE"):
        mod, *_ = load_model(paths.RUNS_DIR / RUNS[m], dev, splits)
        models[m] = mod
    acc = {m: [0.0, 0.0] for m in models}      # sumy po probkach: p, p^2 (tensory per piksel)
    psum = {m: None for m in models}; psq = {m: None for m in models}
    gsum = gsq = None
    n_valid = n_gt10 = 0
    sq_pix = sq_glob = 0.0
    dl = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=nw)
    for b in dl:
        d = b["depth"][:, 0]
        v = d > 0
        n_valid += int(v.sum()); n_gt10 += int((d > DEPTH_VMAX).sum())
        sq_pix += float((((pix_mean[None] - d) ** 2) * v).double().sum())
        sq_glob += float((((glob_mean - d) ** 2) * v).double().sum())
        dd = d.double()
        gsum = dd.sum(0) if gsum is None else gsum + dd.sum(0)
        gsq = (dd ** 2).sum(0) if gsq is None else gsq + (dd ** 2).sum(0)
        for m, mod in models.items():
            with torch.no_grad():
                p = mod({k: b[k] for k in ("img", "depth", "audio")})["depth_predicted"][:, 0].double()
            psum[m] = p.sum(0) if psum[m] is None else psum[m] + p.sum(0)
            psq[m] = (p ** 2).sum(0) if psq[m] is None else psq[m] + (p ** 2).sum(0)
        if time.time() - t0 > BASELINE_BUDGET_S:
            print("  > 15 min -- pomijam baseline")
            return {"skipped": True, "est_min": (time.time() - t0) / 60}

    def pix_std(sm, sq):
        mu = sm / n_te
        return float(((sq / n_te - mu ** 2).clamp(min=0).sqrt()).mean())
    out = {"skipped": False, "glob_mean": glob_mean, "n_tr": n_tr, "n_te": n_te,
           "rmse_pix_mean": float(np.sqrt(sq_pix / n_valid)), "rmse_glob": float(np.sqrt(sq_glob / n_valid)),
           "std_gt": pix_std(gsum, gsq), "std_EB": pix_std(psum["EB"], psq["EB"]),
           "std_ESE": pix_std(psum["ESE"], psq["ESE"]), "frac10": 100 * n_gt10 / n_valid,
           "n_valid": n_valid, "n_gt10": n_gt10, "seconds": time.time() - t0}
    print(f"  baseline gotowy w {out['seconds']:.0f} s")
    return out


# ------------------------------------------------------------------ opisy


def write_md(rows, regula, ev, have_tables, bad, base):
    L = ["# Galeria echo 5.3: wybrane próbki", "",
         f"Źródło wyboru: {regula}. RMSE próbki = sqrt(Σ błędu² / liczba pikseli z prawdą > 0), test@36, ziarno 0.", "",
         "**Reguła (nie ręczna):** w każdej z trzech scen testowych wybrano próbkę o RMSE EB najbliższym medianie "
         "sceny („typowy”, wersja główna) oraz najbliższym 90. percentylowi („trudny”, wersja dodatkowa razem z "
         "typowymi). Percentyl: interpolacja liniowa; remis rozstrzyga kolejność w tabeli.", "",
         "**Kontrola:** RMSE policzone z modeli (best_*.pth, AMP, prawdziwe echo także dla ESE) vs tabela per "
         f"próbka danego modelu, tolerancja {TOL:g}.", "",
         "Odległość = odległość kąta od najbliższej orientacji kardynalnej (0/90/180/270°), czyli odległość dla EA.", "",
         "| scena | lok. | kąt [°] | rodzaj | odl. od kard. [°] | RMSE EA | RMSE ED | RMSE EB | RMSE ESE | max \\|tab − model\\| |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dm = max(r.get(f"d_{m}", float("nan")) for m in RUNS)
        L.append(f"| {r['scena']} | {r['loc']} | {r['kat']} | {r['etykieta']} | {r['dist_kard']} | "
                 + " | ".join(f"{r['mod_' + m]:.4f}" if f"mod_{m}" in r else "n/d" for m in RUNS)
                 + f" | {dm:.1e} |")
    L += ["", f"Wynik kontroli 1e-3: {'wszystkie zgodne' if not bad and have_tables else ('NIEZGODNE: ' + str(bad) if bad else 'brak tabel')}.", ""]
    (OUT / "depth_gallery_echo_53_wybor.md").write_text("\n".join(L), encoding="utf-8")

    P = ["# Galeria echo 5.3: fakty do podpisu", "",
         "- **Wybór próbek:** reguła, nie ręcznie. W każdej scenie testowej (apartment_2, frl_apartment_5, "
         "office_4) próbka o medianie RMSE modelu EB (wersja główna, 3 wiersze) i z 90. percentyla (wersja "
         "dodatkowa, 6 wierszy). Tabela i kontrola: `depth_gallery_echo_53_wybor.md`.",
         "- **Modele:** sieci tylko-echo, ziarno 0, najlepszy checkpoint wg walidacji na 36 orientacjach; tryb eval, "
         "AMP, maska głębia > 0. Wszystkie oceniane z **prawdziwym echem**, także ESE (jak w `eval.json`: "
         "`shuffled_echo = False`).",
         "- **Trening:** EA na 4 orientacjach kardynalnych (0/90/180/270°); ED na 4 orientacjach losowanych z 36; "
         "EB na 36 orientacjach; ESE na 36 orientacjach z echem z innej lokalizacji (permutowanym).",
         f"- **Skala głębi:** wspólna 0–{DEPTH_VMAX:.0f} m, wartości powyżej obcięte; skala błędu 0–{ERR_VMAX:.0f} m. "
         + (f"Pikseli > 10 m w test@36: {pl(base['frac10'], 3)} % ({base['n_gt10']}/{base['n_valid']}); "
            if base and not base.get("skipped") else "")
         + "dla progu 10,29 m 0,014 % (`depth_range_stats.md`). Szary = piksele bez prawdy.",
         "- **RMSE test@36 z `eval.json`** (średnia z 3 ziaren ± odch. std. z próby; pula pikseli całego zbioru):"]
    for k in RUNS:
        m, s, v = ev[k]
        P.append(f"  - {k}: {pl(m, 4)} ± {pl(s, 4)} m (ziarna: {', '.join(pl(x, 4) for x in v)})")
    P += ["- Liczba pod mapą = RMSE tej jednej próbki [m], nie wynik na zbiorze."]
    (OUT / "depth_gallery_echo_53_podpis.md").write_text("\n".join(P) + "\n", encoding="utf-8")

    if base is None:
        return
    B = ["# Baseline dla sieci tylko-echo (opcjonalnie, CPU)", ""]
    if base.get("skipped"):
        B.append(f"Pominięto: szacowany czas {base['est_min']:.0f} min > 15 min.")
    else:
        e_ese, e_eb = ev["ESE"][0], ev["EB"][0]
        B += [f"Zbiór treningowy B: {base['n_tr']} próbek (36 orientacji), test@36: {base['n_te']} próbek, "
              f"{base['n_valid']} ważnych pikseli; czas {base['seconds']:.0f} s.", "",
              "## Fakty (zmierzone)", "",
              f"- Średnia globalna głębi (ważne piksele treningu B): {pl(base['glob_mean'], 4)} m.",
              f"- RMSE test@36 stałej mapy = średnia głębi **per piksel** z treningu B: {pl(base['rmse_pix_mean'], 4)} m.",
              f"- RMSE test@36 **stałej liczby** (średnia globalna): {pl(base['rmse_glob'], 4)} m.",
              f"- RMSE test@36 (eval.json, średnia 3 ziaren): EB {pl(e_eb, 4)} m, ESE {pl(e_ese, 4)} m "
              f"(seed 0: ESE {pl(json.loads(EVAL_JSON['ESE'][0].read_text(encoding='utf-8'))['test_sets']['test@36']['overall']['all']['RMSE'], 4)} m).",
              f"- ESE względem mapy per piksel: {pl(e_ese - base['rmse_pix_mean'], 4)} m "
              f"(stosunek {pl(e_ese / base['rmse_pix_mean'], 3)}); względem stałej liczby: "
              f"{pl(e_ese - base['rmse_glob'], 4)} m (stosunek {pl(e_ese / base['rmse_glob'], 3)}).",
              f"- EB względem mapy per piksel: {pl(e_eb - base['rmse_pix_mean'], 4)} m "
              f"(stosunek {pl(e_eb / base['rmse_pix_mean'], 3)}).",
              f"- Średnie (po pikselach) odchylenie standardowe wartości między próbkami testowymi, per piksel "
              f"(predykcje seed 0, float32 na CPU, prawdziwe echo): EB {pl(base['std_EB'], 4)} m, "
              f"ESE {pl(base['std_ESE'], 4)} m; prawda: {pl(base['std_gt'], 4)} m.", "",
              "## Hipotezy (niezweryfikowane tym pomiarem)", "",
              "- Jeśli RMSE ESE jest bliskie lub większe niż stałej mapy, ESE nie wyciąga z echa informacji o "
              "scenie i jej wynik odpowiada predykcji niezależnej od próbki (bramka zadziałała).",
              "- Małe zróżnicowanie predykcji ESE między próbkami (std) względem EB świadczyłoby, że ESE "
              "zwija się do mapy średniej; sama liczba tego nie dowodzi (mapa średnia vs. predykcja "
              "zależna od próbki, ale błędna, dają różny RMSE przy podobnym std).",
              "- Uwaga: predykcje liczone w float32 na CPU, a RMSE w eval.json z AMP na GPU; różnica numeryczna "
              "nie była osobno mierzona."]
    (OUT / "depth_gallery_echo_53_baseline.md").write_text("\n".join(B) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
