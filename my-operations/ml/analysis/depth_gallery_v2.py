#!/usr/bin/env python
"""Galeria predykcji glebi do pracy (v2): jawne przebiegi, regula wyboru probek, druk 15,5 cm.

    python my-operations/ml/analysis/depth_gallery_v2.py

NIE trenuje niczego i niczego nie nadpisuje (stara galeria: `depth_gallery.py`).
Wejscie: najlepsze checkpointy (`best_*.pth`, wybrane po walidacji) PRZEBIEGOW O ZIARNIE 0,
wskazanych jawnie ponizej -- zadnych globow. Wyjscie (obok siebie, w `figures/gallery/`):
    depth_gallery_v2_pelna.png   8 kolumn, 15,5 x ~17,5 cm
    depth_gallery_v2_waska.png   7 kolumn, 15,5 x ~15 cm
    depth_gallery_v2_wybor.md    wybrane probki + regula + kontrola 1e-3
    depth_gallery_v2_podpis.md   fakty do podpisu rysunku

REGULA WYBORU PROBEK (nie reczna). Z tabeli per probka `eval/B_seed0/samples_test@36.npz`
liczymy RMSE probki = sqrt(sum_sq / n_px). W KAZDEJ scenie testowej bierzemy probke
o RMSE najblizszym medianie sceny ("typowy") i najblizszym 90. percentylowi ("trudny").
Remisy rozstrzyga kolejnosc w tabeli. Brak tabeli => losowanie z ziarnem 20261113 (komunikat).
KONTROLA: RMSE wybranych probek policzone na zywo z modelu B musi zgadzac sie z tabela do 1e-3.

Preprocessing, maska (depth > 0), tryb eval i AMP identyczne jak w `depth_model/evaluate.py`
(AMP tylko na CUDA; na CPU liczy w float32 -- wyniki moga sie rozjechac o szum numeryczny).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402
from ..dataset.echo_h5_dataset import MAX_DEPTH_REPLICA, DatasetConfig, EchoH5Dataset  # noqa: E402
from ..dataset.splits import load_splits  # noqa: E402
from ..depth_model.evaluate import load_model  # noqa: E402

SEED = 0
SELECTION_SEED = 20261113          # uzywane TYLKO gdy brak tabel per probka
TOL = 1e-3
DEPTH_VMAX = 10.0                  # wspolna skala glebi [m]; wartosci powyzej obciete
ERR_VMAX = 3.0                     # wspolna skala bledu [m]
SZARY = "#d9d9d6"                  # piksele bez prawdy (glebia 0)
INK, INK_2 = "#0b0b0b", "#52514e"

OUT = paths.ML_OUTPUTS / "figures" / "gallery"
RGBONLY_ROOT = paths.ML_OUTPUTS / "overnight_2026-11" / "rgbonly"

# JAWNE przebiegi (ziarno 0) -- bez globow.
FULL_RUNS = {"B": "B_seed0", "A": "A_seed0", "EB": "EB_seed0", "EA": "EA_seed0"}
RGBONLY_RUN = "RGBONLY_B_seed0"
TABLE_B = paths.ML_OUTPUTS / "eval" / "B_seed0" / "samples_test@36.npz"
# eval.json do naglowkow: (katalog, plik) dla kazdego ziarna 0,1,2 -- jawnie.
EVAL_JSON = {
    "B": [paths.ML_OUTPUTS / "eval" / f"B_seed{s}" / "eval.json" for s in (0, 1, 2)],
    "A": [paths.ML_OUTPUTS / "eval" / f"A_seed{s}" / "eval.json" for s in (0, 1, 2)],
    "EB": [paths.ML_OUTPUTS / "eval" / f"EB_seed{s}" / "eval.json" for s in (0, 1, 2)],
    "EA": [paths.ML_OUTPUTS / "eval" / f"EA_seed{s}" / "eval.json" for s in (0, 1, 2)],
    "RGB": [RGBONLY_ROOT / f"RGBONLY_B_seed{s}" / "eval.json" for s in (0, 1, 2)],
}


def pl(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


# ------------------------------------------------------------------ wybor probek


def select_samples(ds_scenes, seed_fallback: int = SELECTION_SEED):
    """Zwraca (lista wierszy, opis reguly). Wiersz: scena, loc, kat, rmse_tab, rodzaj, cel."""
    if not TABLE_B.exists():
        return None, f"brak tabeli {TABLE_B.name}: losowanie z ziarnem {seed_fallback}"
    z = np.load(TABLE_B, allow_pickle=True)
    names = [str(s) for s in z["scene_names"]]
    assert names == list(ds_scenes), (names, ds_scenes)
    n_px = z["all.n_px"].astype(float)
    rmse = np.sqrt(z["all.sq"] / np.maximum(n_px, 1))
    rows = []
    for si, name in enumerate(names):
        m = np.flatnonzero((z["scene_idx"] == si) & (n_px > 0))
        r = rmse[m]
        for rodzaj, q in (("typowy (mediana)", 50), ("trudny (90. percentyl)", 90)):
            target = float(np.percentile(r, q))
            j = m[int(np.argmin(np.abs(r - target)))]     # argmin: remis -> pierwszy w tabeli
            rows.append(dict(scena=name, loc=int(z["location_id"][j]), kat=int(z["angle_deg"][j]),
                             rmse_tab=float(rmse[j]), rodzaj=rodzaj, q=q, cel=target,
                             n_scene=int(m.size)))
    return rows, "tabela per probka eval/B_seed0/samples_test@36.npz"


# ------------------------------------------------------------------ modele


def load_rgbonly(device):
    from ..pretext_model.transfer import RGBOnlyModel
    paths.add_parida_to_syspath()
    from models.models import ModelBuilder
    net = ModelBuilder().build_rgbdepth()
    net.load_state_dict(torch.load(RGBONLY_ROOT / RGBONLY_RUN / "best_rgbdepth.pth",
                                   map_location="cpu", weights_only=True))
    return RGBOnlyModel(net, MAX_DEPTH_REPLICA).to(device).eval()


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def eval_rmse_test36(key):
    v = []
    for p in EVAL_JSON[key]:
        d = json.loads(p.read_text(encoding="utf-8"))
        v.append(float(d["test_sets"]["test@36"]["overall"]["all"]["RMSE"]))
    return float(np.mean(v)), float(np.std(v, ddof=1)), v


def predict(model, batch, device, amp):
    with torch.no_grad():
        dev = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(dev)
    return out["depth_predicted"].float().cpu().numpy()[:, 0]


# ------------------------------------------------------------------ rysunek


def render(cols, rows, rgb, gt, preds, srm, headers, path, *, height_gap_in, label_w_in):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK_2})
    cm_d = plt.get_cmap("turbo").copy(); cm_d.set_bad(SZARY)
    cm_e = plt.get_cmap("inferno").copy(); cm_e.set_bad(SZARY)

    W = 15.5 / 2.54
    n, nr = len(cols), len(rows)
    right = 0.04
    pw = (W - label_w_in - right) / n            # szerokosc panelu [in]; panele kwadratowe
    ph = pw * gt.shape[1] / gt.shape[2]
    head_h, cb_h = 0.60, 0.56
    H = head_h + nr * ph + (nr - 1) * height_gap_in + cb_h + 0.05
    fig = plt.figure(figsize=(W, H))
    valid = gt > 0

    def ax_at(c, r):
        x = label_w_in + c * pw
        ytop = H - head_h - r * (ph + height_gap_in)
        return fig.add_axes([x / W, (ytop - ph) / H, (pw - 0.025) / W, ph / H])

    im_d = im_e = None
    for r in range(nr):
        for c, col in enumerate(cols):
            ax = ax_at(c, r)
            kind = col["kind"]
            if kind == "rgb":
                ax.imshow(rgb[r])
            elif kind == "gt":
                im_d = ax.imshow(np.ma.masked_where(~valid[r], gt[r]), cmap=cm_d, vmin=0, vmax=DEPTH_VMAX)
            elif kind == "pred":
                im_d = ax.imshow(np.ma.masked_where(~valid[r], preds[col["m"]][r]), cmap=cm_d,
                                 vmin=0, vmax=DEPTH_VMAX)
                ax.text(0.5, -0.05, f"{pl(srm[col['m']][r], 3)} m", transform=ax.transAxes,
                        ha="center", va="top", fontsize=6.5, color=INK)
            else:
                e = np.abs(preds[col["m"]][r] - gt[r])
                im_e = ax.imshow(np.ma.masked_where(~valid[r], e), cmap=cm_e, vmin=0, vmax=ERR_VMAX)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color(INK_2)
            if r == 0:
                ax.text(0.5, 1.04, col["head"], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.2, linespacing=1.1)
        row = rows[r]
        ytop = H - head_h - r * (ph + height_gap_in)
        fig.text((label_w_in - 0.05) / W, (ytop - ph / 2) / H,
                 f"{row['scena']}\nlok. {row['loc']}, {row['kat']}°\n{row['etykieta']}",
                 ha="right", va="center", fontsize=6.2, linespacing=1.2)

    # Dwa paski skali w jednej linii na dole.
    fmt = FuncFormatter(lambda v, _: pl(v, 0))
    ybar = 0.30 / H
    hbar = 0.11 / H
    dcols = [i for i, c in enumerate(cols) if c["kind"] in ("gt", "pred")]
    ecols = [i for i, c in enumerate(cols) if c["kind"] == "err"]
    for idx, im, lab, ext, ticks in (
            (dcols, im_d, f"głębia [m] (obcięta do {DEPTH_VMAX:.0f} m)", "max", [0, 2, 4, 6, 8, 10]),
            (ecols, im_e, "|błąd| [m]", "max", [0, 1, 2, 3])):
        x0 = (label_w_in + idx[0] * pw + 0.06) / W
        x1 = (label_w_in + (idx[-1] + 1) * pw - 0.085) / W
        cax = fig.add_axes([x0, ybar, x1 - x0, hbar])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=ext)
        cb.set_ticks(ticks)
        cb.set_label(lab, fontsize=7, labelpad=1.5)
        cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
        cb.ax.xaxis.set_major_formatter(fmt)
        cb.outline.set_linewidth(0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> int:
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"urzadzenie: {device}, AMP: {amp}, ziarno: {SEED}")
    splits = load_splits()
    ds = EchoH5Dataset(DatasetConfig(mode="test", angle_subset="all", augment=False), splits=splits)

    rows, regula = select_samples(ds.scenes)
    if rows is None:       # awaryjnie: losowanie z ustalonym ziarnem
        print(regula)
        rng = np.random.default_rng(SELECTION_SEED)
        rows = []
        for si, name in enumerate(ds.scenes):
            pool = np.flatnonzero(ds.index_scene == si)
            for rodzaj in ("typowy (losowo)", "trudny (losowo)"):
                j = int(rng.choice(pool))
                rows.append(dict(scena=name, loc=int(ds.index_loc[j]), kat=int(ds.index_angle[j]),
                                 rmse_tab=float("nan"), rodzaj=rodzaj, q=None, cel=float("nan"),
                                 n_scene=int(pool.size)))
    for r in rows:
        r["etykieta"] = "typowy" if r["rodzaj"].startswith("typowy") else "trudny"

    key = (ds.index_scene.astype(np.int64) * 10**9 + ds.index_loc.astype(np.int64) * 10**4
           + ds.index_angle.astype(np.int64))
    idx = []
    for r in rows:
        si = ds.scenes.index(r["scena"])
        k = si * 10**9 + r["loc"] * 10**4 + r["kat"]
        hit = np.flatnonzero(key == k)
        assert hit.size == 1, (r, hit)
        idx.append(int(hit[0]))
    items = [ds[i] for i in idx]
    batch = {k: torch.stack([it[k] for it in items]) for k in ("img", "depth", "audio")}
    gt = batch["depth"].numpy()[:, 0]
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    rgb = [np.clip(it["img"].numpy().transpose(1, 2, 0) * std + mean, 0, 1) for it in items]
    valid = gt > 0

    # Modele (ziarno 0) -- kolejnosc jawna.
    preds, params = {}, {}
    for m, run in FULL_RUNS.items():
        model, cond, _, loaded = load_model(paths.RUNS_DIR / run, device, splits)
        assert cond.geometry == "main", (run, cond.geometry)
        preds[m] = predict(model, batch, device, amp)
        params[m] = n_params(model)
        print(f"  {run}: wagi {sorted(loaded.values())}")
        del model
    model = load_rgbonly(device)
    preds["RGB"] = predict(model, batch, device, amp)
    params["RGB"] = n_params(model)
    print(f"  {RGBONLY_RUN}: best_rgbdepth.pth")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    srm = {m: np.array([np.sqrt(((p[i][valid[i]] - gt[i][valid[i]]) ** 2).mean())
                        for i in range(len(rows))]) for m, p in preds.items()}

    # Kontrola 1e-3 wzgledem tabeli per probka.
    for i, r in enumerate(rows):
        r["rmse_B"] = float(srm["B"][i])
        r["diff"] = abs(r["rmse_B"] - r["rmse_tab"]) if np.isfinite(r["rmse_tab"]) else float("nan")
        r["ok"] = bool(r["diff"] <= TOL) if np.isfinite(r["diff"]) else None
    bad = [r for r in rows if r["ok"] is False]
    for r in rows:
        print(f"  {r['scena']:16s} lok {r['loc']:3d} {r['kat']:3d}°  tab {pl(r['rmse_tab'], 5)}  "
              f"model {pl(r['rmse_B'], 5)}  |d| {r['diff']:.2e}  {'OK' if r['ok'] else 'NIEZGODNE'}")
    if bad:
        print("KONTROLA 1e-3 NIEUDANA -- zapisuje wynik mimo to, z oznaczeniem")

    # RMSE test@36 z eval.json (srednia z 3 ziaren).
    ev = {k: eval_rmse_test36(k) for k in EVAL_JSON}
    hd = lambda k: f"RMSE\ntest@36\n{pl(ev[k][0], 3)} m"  # noqa: E731
    C = {
        "rgb": dict(kind="rgb", head="RGB\n(wejście)"),
        "gt": dict(kind="gt", head="prawda"),
        "B": dict(kind="pred", m="B", head="obraz+echo\n36 orient.\n" + hd("B")),
        "A": dict(kind="pred", m="A", head="obraz+echo\n4 orient.\n" + hd("A")),
        "EB": dict(kind="pred", m="EB", head="samo echo\n36 orient.\n" + hd("EB")),
        "EA": dict(kind="pred", m="EA", head="samo echo\n4 orient.\n" + hd("EA")),
        "RGB": dict(kind="pred", m="RGB", head="sam obraz 36\n(RGB-only)\n" + hd("RGB")),
        "errB": dict(kind="err", m="B", head="|błąd|\nobraz+echo\n36 orient."),
        "errR": dict(kind="err", m="RGB", head="|błąd|\nsam obraz 36\n(RGB-only)"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    render([C[k] for k in ("rgb", "gt", "B", "A", "EB", "EA", "RGB", "errB")], rows, rgb, gt, preds, srm,
           None, OUT / "depth_gallery_v2_pelna.png", height_gap_in=0.34, label_w_in=0.78)
    render([C[k] for k in ("rgb", "gt", "B", "RGB", "EB", "errB", "errR")], rows, rgb, gt, preds, srm,
           None, OUT / "depth_gallery_v2_waska.png", height_gap_in=0.12, label_w_in=0.78)

    # Odsetek pikseli > 10 m w zbiorze test@36 (zeby podpis mial liczbe, nie "okolo").
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    n_valid = n_gt10 = 0
    for b in dl:
        d = b["depth"]
        n_valid += int((d > 0).sum()); n_gt10 += int((d > DEPTH_VMAX).sum())
    frac10 = 100 * n_gt10 / n_valid
    ds.close()

    write_md(rows, regula, ev, params, frac10, n_valid, n_gt10, bad)
    print("zapisano:", *[p.name for p in sorted(OUT.glob("depth_gallery_v2_*"))])
    return 1 if bad else 0


def write_md(rows, regula, ev, params, frac10, n_valid, n_gt10, bad):
    L = ["# Galeria v2: wybrane próbki", "",
         f"Źródło reguły: {regula}. RMSE próbki = sqrt(Σ błędu² / liczba pikseli z prawdą > 0), "
         "model B, ziarno 0, test@36.", "",
         "**Reguła (nie ręczna):** w każdej z trzech scen testowych wybrano próbkę, której RMSE jest "
         "najbliższe medianie RMSE sceny („typowy”) oraz 90. percentylowi („trudny”); remis rozstrzyga "
         "kolejność w tabeli. Percentyl: interpolacja liniowa (`numpy.percentile`).", "",
         f"**Kontrola:** RMSE wybranych próbek policzone z modelu B (`best_*.pth`, AMP) musi zgadzać się "
         f"z tabelą per próbka z tolerancją {TOL:g}.", "",
         "| scena | lokalizacja | kąt [°] | rodzaj | cel (percentyl sceny) [m] | RMSE B, tabela [m] | "
         "RMSE B, model [m] | \\|różnica\\| | kontrola |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['scena']} | {r['loc']} | {r['kat']} | {r['rodzaj']} | {r['cel']:.5f} | "
                 f"{r['rmse_tab']:.5f} | {r['rmse_B']:.5f} | {r['diff']:.1e} | "
                 f"{'OK' if r['ok'] else 'NIEZGODNE'} |")
    L += ["", f"Wynik kontroli: {'wszystkie 6 zgodne' if not bad else f'NIEZGODNE: {len(bad)}'} "
          f"(tolerancja {TOL:g}).", ""]
    (OUT / "depth_gallery_v2_wybor.md").write_text("\n".join(L), encoding="utf-8")

    names = {"B": "obraz+echo, 36 orientacji (B_seed0)", "A": "obraz+echo, 4 orientacje (A_seed0)",
             "EB": "samo echo, 36 orientacji (EB_seed0)", "EA": "samo echo, 4 orientacje (EA_seed0)",
             "RGB": "sam obraz, 36 orientacji, RGB-only (RGBONLY_B_seed0)"}
    P = ["# Galeria v2: fakty do podpisu", "",
         "- **Wybór próbek:** reguła, nie ręcznie. W każdej z trzech scen testowych (apartment_2, "
         "frl_apartment_5, office_4) próbka o medianie RMSE („typowy”) i z 90. percentyla („trudny”), "
         "wg RMSE modelu B na tabeli per próbka (test@36). Wybrane próbki i kontrola: `depth_gallery_v2_wybor.md`.",
         "- **Modele:** wszystkie ziarno 0, najlepszy checkpoint wg walidacji na 36 orientacjach; tryb eval, "
         "AMP, ta sama maska (głębia > 0) i preprocessing co w ewaluacji.",
         f"- **Skala głębi:** wspólna 0–{DEPTH_VMAX:.0f} m dla wszystkich map głębi, wartości powyżej "
         f"{DEPTH_VMAX:.0f} m obcięte do koloru maksimum. Pikseli z głębią > {DEPTH_VMAX:.0f} m w test@36: "
         f"{pl(frac10, 3)} % ważnych pikseli ({n_gt10}/{n_valid}); dla progu 10,29 m "
         "(`depth_range_stats.md`) 0,014 %. Skala błędu: 0–3 m (też obcięta). Szary = piksele bez prawdy (głębia 0).",
         "- **RMSE test@36 z `eval.json`** (średnia z 3 ziaren, ± odch. std. z próby; RMSE puli pikseli całego zbioru, "
         "nie tych 6 próbek):"]
    for k in ("B", "A", "EB", "EA", "RGB"):
        m, s, v = ev[k]
        P.append(f"  - {names[k]}: {pl(m, 4)} ± {pl(s, 4)} m (ziarna: {', '.join(pl(x, 4) for x in v)})")
    P += ["- **Liczby parametrów:** "
          f"pełny model (obraz+echo) {params['B']:,} ; RGB-only {params['RGB']:,} ; "
          f"tylko-echo {params['EB']:,}".replace(",", " ") + ".",
          "- **RGB-only vs pełny:** RGB-only (sama `RGBDepthNet`, krok 7) uczono na tych samych parach "
          "(lokalizacja, kąt) co B, tym samym budżetem kroków; różni się od pełnego modelu brakiem echa, "
          "materiału i mechanizmu uwagi (a więc nie jest czystym pomiarem wkładu echa).",
          "- Liczba pod mapą predykcji = RMSE tej jednej próbki [m]; nie jest to wynik na zbiorze."]
    (OUT / "depth_gallery_v2_podpis.md").write_text("\n".join(P) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
