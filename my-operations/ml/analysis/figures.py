#!/usr/bin/env python
"""Rysunki do pracy. ZERO GPU -- wszystko z zapisanych plikow dowodowych.

    python my-operations/ml/analysis/figures.py

Produkuje do `outputs/ml/figures/`:

    rys_1_krzywa_nasycenia.png     GLOWNY RYSUNEK -- RMSE w funkcji siatki K
    rys_2_generalizacja_katowa.png RMSE w funkcji odleglosci od siatki treningowej
    rys_3_rozklad_efektu.png       gestosc katowa vs ilosc danych, oba modele

ZASADY, KTORE TU OBOWIAZUJA (i dlaczego, zeby nikt ich potem nie "poprawil"):

1. KAZDY punkt ma slupek bledu z RZECZYWISTEGO sd po ziarnach. Gladka linia bez
   slupkow sugerowalaby precyzje, ktorej te pomiary nie maja -- przy 3 ziarnach
   sd siega 0,0136 RMSE (K=18), czyli wiecej niz cala roznica K=18 -> K=36.
2. JEDNA os Y na panel. Efekt gestosci w `echo2depth` (0,147) i w pelnym modelu
   (0,020) roznia sie 7x, wiec na wspolnej osi slupki pelnego modelu bylyby
   niewidoczne. Stad DWA panele z wlasnymi osiami, a nie jeden wykres z dwiema
   skalami.
3. Serie sa opisane BEZPOSREDNIO przy krzywej, nie tylko w legendzie -- rysunek
   ma byc czytelny takze po wydruku w skali szarosci i dla osob z zaburzeniami
   rozroznialnosci barw.
4. Os X rysunku 1 jest LOGARYTMICZNA. K to liczebnosc siatki (skala ilorazowa),
   a wartosci 4..36 na osi liniowej sciskaja caly interesujacy zakres 4-12 przy
   lewej krawedzi -- czyli dokladnie tam, gdzie jest wniosek pracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402

OUT = paths.ML_OUTPUTS / "figures"

# Paleta kategoryczna zwalidowana (light mode, powierzchnia #fcfcfb):
# najgorsza para CVD dE 9,2 / normal 27,6 -- powyzej progow 8 i 15.
C_MAIN = "#2a78d6"    # slot 1, blue   -- seria glowna
C_ALT = "#eb6834"     # slot 2, orange -- seria druga / linia odniesienia
C_THIRD = "#1baf7a"   # slot 3, aqua   -- linia odniesienia
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": INK_2, "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": INK, "figure.dpi": 200,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def _clean(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ rysunek 1


def rysunek_1_krzywa_nasycenia() -> Path:
    d = _load(paths.ML_OUTPUTS / "echo_ablation" / "final_results_2026-08-13.json")
    pts = d["krzywa_stalego_budzetu"]["punkty"]
    ks = sorted(pts, key=int)
    x = np.array([int(k) for k in ks], dtype=float)
    y = np.array([pts[k]["mean"] for k in ks])
    e = np.array([pts[k]["sd"] or 0.0 for k in ks])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _clean(ax)

    # Obszar nasycenia -- to jest WNIOSEK rysunku, wiec musi byc widoczny
    # zanim czytelnik zacznie odczytywac punkty.
    ax.axvspan(9, 12, color=C_MAIN, alpha=0.07, zorder=1)
    ax.annotate("nasycenie\nK = 9–12", xy=(10.4, y.max() - 0.012),
                ha="center", va="top", fontsize=9, color=INK_2)

    ax.errorbar(x, y, yerr=e, color=C_MAIN, linewidth=2, marker="o",
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                capsize=4, elinewidth=1.4, zorder=3)

    # Etykiety tylko przy koncach i przy punkcie odciecia -- nie przy kazdym
    # punkcie (patrz zasada "nigdy liczba nad kazdym markerem").
    for k, xi, yi in ((4, x[0], y[0]), (9, x[2], y[2]), (36, x[-1], y[-1])):
        ax.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(k)}" for k in x])
    ax.minorticks_off()
    ax.set_xlabel("liczba orientacji w siatce K, z której losowane są kąty")
    ax.set_ylabel("RMSE na zbiorze testowym (36 orientacji)")
    ax.set_title("Nasycenie efektu gęstości kątowej przy stałym budżecie próbek",
                 loc="left", fontweight="bold", pad=14)
    ax.text(0.0, 1.02, "sieć głębi (tylko echo) · 4 próbki na lokalizację (5 496) w każdym "
                       "punkcie · 3 ziarna, wąsy = sd",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    # Strzalki zysku: cala wartosc rysunku to porownanie tych dwoch odcinkow.
    ax.annotate("", xy=(9, y[2]), xytext=(4, y[0]),
                arrowprops=dict(arrowstyle="-", color=C_ALT, lw=1.2, ls=(0, (4, 3))))
    ax.annotate(f"4 → 9:  −{y[0] - y[2]:.3f}", xy=(6.0, (y[0] + y[2]) / 2),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9.5, color=C_ALT, fontweight="bold")
    # Adnotacja idzie NAD plaska czescia krzywej, nie pod nia: pod spodem
    # wchodzila na os X i na wasy punktu K=18.
    ax.annotate(f"9 → 36:  −{y[2] - y[-1]:.3f}   (6,7× mniej)",
                xy=(20, y[3] + 0.011), ha="center", va="bottom",
                fontsize=9.5, color=INK_2)
    ax.set_ylim(y.min() - e[-1] - 0.012, y.max() + e[0] + 0.016)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "rys_1_krzywa_nasycenia.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ rysunek 2


def rysunek_2_generalizacja_katowa() -> Path:
    d = _load(paths.ML_OUTPUTS / "echo_ablation" / "echo_3seeds.json")
    kr = d["krzywa_katowa_EA"]
    x = np.array(sorted(float(k) for k in kr))
    y = np.array([kr[str(int(k))]["mean"] for k in x])
    e = np.array([kr[str(int(k))]["sd"] for k in x])
    r = d["RMSE_test36"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _clean(ax)

    # Warunki `ED` i `EB` NIE MAJA tej krzywej i to nie jest brak danych:
    # ucza sie na kątach z calej siatki, wiec KAZDY kat testowy lezy w
    # odleglosci 0 od ich siatki treningowej. Dlatego sa liniami odniesienia,
    # a nie seriami -- narysowanie ich jako krzywych sugerowaloby, ze zmierzono
    # cos, czego zmierzyc sie nie da.
    for val, col, name in ((r["ED"]["mean"], C_ALT, "ED — 4 kąty losowe z 36"),
                           (r["EB"]["mean"], C_THIRD, "EB — wszystkie 36")):
        ax.axhline(val, color=col, linewidth=2, ls=(0, (5, 3)), zorder=2)
        ax.annotate(f"{name}   {val:.3f}", xy=(x[-1], val), xytext=(-2, 6),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=col, fontweight="bold")

    ax.errorbar(x, y, yerr=e, color=C_MAIN, linewidth=2, marker="o",
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                capsize=4, elinewidth=1.4, zorder=3)
    # Etykieta serii siada NAD punktem K=30, nie nad ostatnim: nad ostatnim
    # nachodzila na gorna czesc wasa i na prawa krawedz osi.
    ax.annotate("EA — 4 kierunki kardynalne", xy=(x[-2], y[-2] + e[-2]), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=9,
                color=C_MAIN, fontweight="bold")

    ax.annotate(f"luka {y[-1] - y[0]:+.3f}\n({100 * (y[-1] - y[0]) / y[0]:.0f} %)",
                xy=(x[-1], y[-1]), xytext=(-16, -36), textcoords="offset points",
                ha="center", fontsize=9, color=INK_2)
    ax.set_ylim(min(r["EB"]["mean"], y.min()) - 0.022, y.max() + e[-1] + 0.030)

    ax.set_xticks(x)
    ax.set_xlabel("odległość testowanego kąta od najbliższego kąta treningowego [°]")
    ax.set_ylabel("RMSE")
    ax.set_title("Baseline 4-kierunkowy nie pokrywa przestrzeni orientacji",
                 loc="left", fontweight="bold", pad=14)
    ax.text(0.0, 1.02, "sieć głębi (tylko echo) · 3 ziarna, wąsy = sd · warunki ED i EB nie mają "
                       "tej osi: pokrywają całą siatkę",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "rys_2_generalizacja_katowa.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ rysunek 3


def rysunek_3_rozklad_efektu() -> Path:
    e3 = _load(paths.ML_OUTPUTS / "echo_ablation" / "echo_3seeds.json")["kontrasty"]
    gl = _load(paths.ML_OUTPUTS / "echo_ablation" / "final_results_2026-08-15.json")
    sk = gl["glowne_3ziarna"]["skladowe"]

    panele = (
        ("sieć głębi (tylko echo)",
         [("gęstość kątowa\n(D − A)", e3["gestosc katowa D-A"]["mean"],
           e3["gestosc katowa D-A"]["sd_seeds"]),
          ("ilość danych\n(B − D)", e3["ilosc danych B-D"]["mean"],
           e3["ilosc danych B-D"]["sd_seeds"])],
         70.2),
        ("sieć głębi (pełna: obraz + echo)",
         [("gęstość kątowa\n(D − A)", abs(sk["gestosc_D_minus_A"]["mean"]),
           sk["gestosc_D_minus_A"]["sd"]),
          ("ilość danych\n(B − D)", abs(sk["ilosc_danych_B_minus_D"]["mean"]),
           sk["ilosc_danych_B_minus_D"]["sd"])],
         42.0),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    for ax, (tytul, slupki, udzial) in zip(axes, panele):
        _clean(ax)
        xs = np.arange(len(slupki))
        vals = [s[1] for s in slupki]
        errs = [s[2] for s in slupki]
        ax.bar(xs, vals, yerr=errs, width=0.55,
               color=[C_MAIN, C_ALT], capsize=5,
               error_kw=dict(elinewidth=1.4, ecolor=INK_2), zorder=3)
        for xi, v, er in zip(xs, vals, errs):
            ax.annotate(f"{v:.3f}", xy=(xi, v + er), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=10, fontweight="bold", color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in slupki], fontsize=9)
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.32)
        ax.set_title(tytul, loc="left", fontsize=10, fontweight="bold", pad=10)
        ax.text(0.0, 1.005, f"udział gęstości: {udzial:.1f} %",
                transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    axes[0].set_ylabel("poprawa RMSE (wartość dodatnia = lepiej)")

    fig.suptitle("Prior wizualny przykrywa strukturę kątową echa",
                 x=0.005, y=0.985, ha="left", va="top", fontweight="bold", fontsize=11.5)
    fig.text(0.005, 0.925,
             "3 ziarna, wąsy = sd · UWAGA: osie Y mają różne skale — efekt w pełnym modelu "
             "jest 7,2× mniejszy",
             fontsize=8.5, color=INK_2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.915))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "rys_3_rozklad_efektu.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    _style()
    for fn in (rysunek_1_krzywa_nasycenia, rysunek_2_generalizacja_katowa,
               rysunek_3_rozklad_efektu):
        print(f"zapisano: {fn()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
