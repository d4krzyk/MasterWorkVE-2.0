#!/usr/bin/env python
"""Galeria jakosciowa: obraz, prawda i predykcje wszystkich wersji obok siebie.

    python my-operations/ml/analysis/depth_gallery.py
    python my-operations/ml/analysis/depth_gallery.py --n 8 --runs B_seed0 EB_seed0 A_seed0

Po co: wszystkie liczby w pracy sa skalarne (RMSE, delty, przedzialy ufnosci).
Rysunek jakosciowy pokazuje, CO te liczby znacza -- w szczegolnosci gdzie model
z samego echa gubi geometrie, a gdzie ratuje ja obraz.

UKLAD KOLUMN. Kazdy wiersz to jedna probka testowa:
    RGB | prawda | <predykcja kazdego wskazanego przebiegu> | blad bezwzgledny
Wszystkie mapy glebi rysowane sa we WSPOLNEJ skali (0 .. max_depth), inaczej
porownanie miedzy kolumnami bylo by zludzeniem -- kazdy panel dostawalby wlasna
normalizacje i nawet zly model wygladalby dobrze.

Piksele NIEWAZNE (glebia 0 w prawdzie) sa wyszarzane, bo model nie jest na nich
punktowany; bez tego czytelnik widzi "blad" tam, gdzie zadnej prawdy nie ma.
Kolor "nieważnych" jest USTAWIANY JAWNIE (szary) -- domyslny kolor maski to
przezroczystosc, ktora na bialym tle czyta sie jak obiekt tuz przed kamera,
a w scenach z dziura w suficie zajmuje cwierc kadru.

DWA PASKI SKALI, oba na dole: glebia (wspolna dla kolumn z mapami glebi) i blad
bezwzgledny. Rysunek deklaruje wspolna skale 0..max_depth, wiec musi ja tez
pokazac -- bez paska zaden panel glebi nie daje sie odczytac ilosciowo.

CZEGO TEN RYSUNEK NIE POKAZUJE. Kolumna "SAM OBRAZ bez echa" NIE jest pomiarem
wkladu echa: ten model rozni sie od sasiednich nie tylko brakiem echa, ale tez
brakiem mechanizmu uwagi, brakiem sieci materialu i 19x mniejsza liczba
parametrow (STAN_WYNIKOW.md §8). Czystym pomiarem wkladu echa jest warunek `SE`
(echo przetasowane). Ostrzezenie jest drukowane POD rysunkiem, bo bez niego
kolumna czyta sie dokladnie odwrotnie do intencji.
"""

from __future__ import annotations

import argparse
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

# Domyslny zestaw: pelny model, sama galaz echa, oraz warunek rzadki (4 katy).
DEFAULT_RUNS = ("B_seed0", "A_seed0", "EB_seed0", "EA_seed0", "transfer:scratch_seed0")

# Etykiety MUSZA podawac liczbe katow treningowych, inaczej kolumna nie mowi,
# co porownuje. "echo losowe" bez tej informacji bylo mylace: warunek `SE` jest
# trenowany na PELNYCH 36 katach, permutowane jest tylko przypisanie echa do
# obrazu -- czyli to kontrola "ile wnosi echo", a nie wariant gestosci.
LABELS = {
    "B_seed0": "obraz + echo\n36 kątów",
    "A_seed0": "obraz + echo\n4 kąty",
    "D_seed0": "obraz + echo\n4 losowane",
    "EB_seed0": "SAMO ECHO\n36 kątów",
    "EA_seed0": "SAMO ECHO\n4 kąty",
    "ED_seed0": "SAMO ECHO\n4 losowane",
    "SE_seed0": "obraz + echo\n36 kątów, echo\nz innej lokalizacji",
    "ESE_seed0": "SAMO ECHO\n36 kątów, echo\nz innej lokalizacji",
    "transfer:scratch_seed0": "SAM OBRAZ\nbez echa",
}


def load_any(spec: str, device, splits):
    """Wczytuje przebieg `depth_model` ALBO `pretext_transfer`.

    Prefiks `transfer:` wskazuje na zadanie docelowe Modelu 2 -- siec `RGBOnlyModel`,
    czyli sam `RGBDepthNet` bez galezi audio. Ma inny uklad katalogu i inna klase
    modelu niz warunki macierzy, wiec potrzebuje osobnej sciezki wczytania.
    """
    if not spec.startswith("transfer:"):
        return load_model(paths.RUNS_DIR / spec, device, splits)[0]

    from ..pretext_model.transfer import RGBOnlyModel
    name = spec.split(":", 1)[1]
    d = paths.ML_OUTPUTS / "pretext_transfer" / f"transfer_{name}"
    paths.add_parida_to_syspath()
    from models.models import ModelBuilder
    net = ModelBuilder().build_rgbdepth()
    net.load_state_dict(torch.load(d / "best_rgbdepth.pth", map_location="cpu",
                                   weights_only=True))
    return RGBOnlyModel(net, MAX_DEPTH_REPLICA).to(device).eval()

# Kolor pikseli niewaznych. Szary, nie bialy: bialy na bialym tle rysunku czyta
# sie jak obiekt tuz przed kamera.
SZARY_NIEWAZNE = "#d9d9d6"


def _pl(x: float, nd: int = 3) -> str:
    """Przecinek dziesietny -- praca jest po polsku (docs/RYSUNKI.md)."""
    return f"{x:.{nd}f}".replace(".", ",")


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def unnormalize(img: torch.Tensor) -> np.ndarray:
    a = img.cpu().numpy().transpose(1, 2, 0)
    return np.clip(a * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--n", type=int, default=6, help="ile probek testowych")
    ap.add_argument("--seed", type=int, default=0, help="ziarno wyboru probek")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Kolor pikseli niewaznych ustawiany JAWNIE -- patrz naglowek modulu.
    cmap_glebia = plt.get_cmap("turbo").copy()
    cmap_blad = plt.get_cmap("inferno").copy()
    for cm in (cmap_glebia, cmap_blad):
        cm.set_bad(SZARY_NIEWAZNE)

    device = torch.device("cuda")
    splits = load_splits()

    # Probki wybierane RAZ, wspolne dla wszystkich modeli -- inaczej kolumny
    # pokazywalyby rozne sceny i nic by z tego nie wynikalo.
    ds = EchoH5Dataset(DatasetConfig(mode="test", angle_subset="all", augment=False),
                       splits=splits)
    rng = np.random.default_rng(args.seed)
    # Preferujemy probki z duza zmiennoscia glebi -- plaska sciana nie pokazuje
    # niczego ciekawego ani o echu, ani o obrazie.
    #
    # ALE stratyfikowanie po scenach jest konieczne: sam wybor po zmiennosci
    # daje komplet probek z `frl_apartment_5` (najbardziej zagracona scena
    # held-out), a rysunek ma pokazywac zachowanie modelu, nie jedna scene.
    per_scene = max(1, args.n // len(ds.scenes))
    idx: list[int] = []
    for si, _ in enumerate(ds.scenes):
        pool = np.flatnonzero(ds.index_scene == si)
        if pool.size == 0:
            continue
        cand = rng.choice(pool, size=min(120, pool.size), replace=False)
        score = []
        for i in cand:
            d = ds[int(i)]["depth"].numpy()
            v = d[d > 0]
            if v.size < 100:
                score.append(0.0)
                continue
            # Kryterium: widok W GLAB, nie na sciane.
            #
            # Samo odchylenie standardowe nie wystarcza -- sciana ogladana pod
            # katem tez ma spory rozrzut, a nie pokazuje nic ciekawego ani o echu,
            # ani o obrazie. Mnozymy przez 90. percentyl glebi, ktory jest wysoki
            # tylko wtedy, gdy w kadrze jest przestrzen (korytarz, otwarty pokoj),
            # a niski przy scianie tuz przed agentem.
            score.append(float(np.percentile(v, 90) * v.std()))
        idx += [int(cand[j]) for j in np.argsort(score)[-per_scene:][::-1]]
    idx = idx[:args.n]
    args.n = len(idx)

    batch = {k: torch.stack([ds[i][k] for i in idx]) for k in ("img", "depth", "audio")}
    gt = batch["depth"].numpy()[:, 0]
    rgb = [unnormalize(ds[i]["img"]) for i in idx]
    meta = [(ds.scenes[int(ds.index_scene[i])], int(ds.index_loc[i]), int(ds.index_angle[i]))
            for i in idx]

    preds, labels = [], []
    for run in args.runs:
        if not run.startswith("transfer:") and not (paths.RUNS_DIR / run / "config.json").exists():
            print(f"  pomijam {run}: brak przebiegu")
            continue
        model = load_any(run, device, splits)
        with torch.no_grad():
            dev = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=True):
                out = model(dev)
            preds.append(out["depth_predicted"].float().cpu().numpy()[:, 0])
        labels.append(LABELS.get(run, run))
        del model
        torch.cuda.empty_cache()
        print(f"  policzono {run}")

    ncol = 2 + len(preds) + 1
    fig, axes = plt.subplots(args.n, ncol, figsize=(2.1 * ncol, 2.25 * args.n))
    if args.n == 1:
        axes = axes[None, :]
    valid = gt > 0

    for r in range(args.n):
        axes[r, 0].imshow(rgb[r]); axes[r, 0].set_ylabel(
            f"{meta[r][0]}\nlok {meta[r][1]}, {meta[r][2]}°", fontsize=6)
        g = np.ma.masked_where(~valid[r], gt[r])
        im_glebia = axes[r, 1].imshow(g, cmap=cmap_glebia, vmin=0, vmax=MAX_DEPTH_REPLICA)
        for c, p in enumerate(preds):
            axes[r, 2 + c].imshow(np.ma.masked_where(~valid[r], p[r]), cmap=cmap_glebia,
                                  vmin=0, vmax=MAX_DEPTH_REPLICA)
        # Blad liczony wzgledem PIERWSZEGO wskazanego przebiegu -- domyslnie
        # pelnego modelu, czyli tego, ktory ma byc odniesieniem.
        err = np.ma.masked_where(~valid[r], np.abs(preds[0][r] - gt[r]))
        im_blad = axes[r, -1].imshow(err, cmap=cmap_blad, vmin=0, vmax=3.0)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    # Naglowek kolumny bledu MUSI niesc pelna etykiete modelu odniesienia
    # (lacznie z liczba katow), inaczej nie wiadomo, czyj to blad.
    heads = (["RGB (wejście)", "prawda"] + labels
             + ["|błąd|\n" + labels[0].replace("\n", ", ")])
    for c, h in enumerate(heads):
        axes[0, c].set_title(h, fontsize=7)

    fig.suptitle(f"Predykcja głębi — porównanie wersji modelu "
                 f"(wspólna skala 0–{_pl(MAX_DEPTH_REPLICA)} m)", fontsize=10)

    # Paski skali na DOLE: pierwszy pod kolumnami glebi, drugi pod kolumna bledu.
    # Wczesniej byl tylko pasek bledu, przyklejony z boku w polowie wysokosci --
    # nie bylo wiadomo, do ktorej kolumny nalezy, a szesc kolumn glebi nie mialo
    # zadnej skali mimo deklaracji we wspolnym tytule.
    cb_g = fig.colorbar(im_glebia, ax=axes[:, 1:-1].ravel().tolist(),
                        orientation="horizontal", location="bottom",
                        fraction=0.015, pad=0.015, aspect=55)
    cb_g.set_label("głębia [m]", fontsize=8)
    cb_b = fig.colorbar(im_blad, ax=axes[:, -1].tolist(),
                        orientation="horizontal", location="bottom",
                        fraction=0.015, pad=0.015, aspect=12)
    cb_b.set_label("|błąd| [m]", fontsize=8)
    for cb in (cb_g, cb_b):
        cb.ax.tick_params(labelsize=7)
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _pl(v, 0)))

    stopka = ("szary = piksele nieważne (brak prawdy: dziury w skanie, niebo) — "
              "model nie jest na nich punktowany")
    if any(lab.startswith("SAM OBRAZ") for lab in labels):
        stopka += ("\nkolumna „SAM OBRAZ bez echa” NIE mierzy wkładu echa: ten model różni się "
                   "też brakiem uwagi, brakiem sieci materiału i 19× liczbą parametrów "
                   "(STAN_WYNIKOW.md §8)")
    # Stopka PONIZEJ obszaru figury (va="top" przy y=0): wewnatrz kolidowalaby
    # z paskami skali, a `bbox_inches="tight"` i tak ja obejmie przy zapisie.
    fig.text(0.5, -0.004, stopka, ha="center", va="top", fontsize=7.5, color="#52514e")
    out = args.out or (paths.ML_OUTPUTS / "gallery" / "depth_gallery.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    # Liczby pod rysunkiem -- zeby dalo sie podpisac konkretnymi wartosciami.
    stats = {}
    for lab, p in zip(labels, preds):
        e = [float(np.sqrt(((p[r][valid[r]] - gt[r][valid[r]]) ** 2).mean()))
             for r in range(args.n)]
        stats[lab.replace("\n", " ")] = {"RMSE_per_sample": e, "mean": float(np.mean(e))}
    (out.parent / "depth_gallery.json").write_text(
        json.dumps({"probki": [{"scena": m[0], "lokalizacja": m[1], "kat": m[2]} for m in meta],
                    "RMSE_na_tych_probkach": stats,
                    "uwaga": "to sa TE PROBKI, nie caly zbior testowy -- do podpisu rysunku, "
                             "nie do tabeli wynikow"}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    ds.close()
    print(f"\nzapisano: {out}")
    for k, v in stats.items():
        print(f"  {k:28s} RMSE na tych {args.n} probkach: {_pl(v['mean'], 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
