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
5. SEPARATOR DZIESIETNY TO PRZECINEK -- wszedzie: podzialki osi, etykiety
   punktow, adnotacje, procenty. Praca jest po polsku. Sluzy do tego `_pl()`
   i `_przecinki()`; nie formatowac liczb goly m f-stringiem.
6. NAZWA METRYKI JEST TA SAMA NA WSZYSTKICH RYSUNKACH: "RMSE glebi [m]".
   W projekcie sa DWIE rozne wielkosci nazywane RMSE (metryka glebi w metrach
   i odleglosc RMS spektrogramow) -- patrz docs/KONWENCJE.md §3. Os bez
   jednostki pozwalalaby pomylic je na rysunku.
7. ZNAK. Rysunki 3 i 4 pokazuja WIELKOSC poprawy (spadek RMSE, slupki dodatnie),
   rysunek 1 pokazuje ZMIANE RMSE wzdluz krzywej (wartosci ujemne). Jedno i
   drugie jest zgodne z konwencja z docs/KONWENCJE.md §2, ale MUSI byc nazwane
   na osi i w podpisie, bo inaczej ta sama wielkosc raz ma minus, a raz nie.
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


# Nazwa metryki uzywana na KAZDYM rysunku (zasada 6 w naglowku modulu).
OS_RMSE = "RMSE głębi [m]"


def _pl(x: float, nd: int = 3, znak: bool = False) -> str:
    """Liczba po polsku: przecinek dziesietny, opcjonalnie jawny znak."""
    return f"{x:{'+' if znak else ''}.{nd}f}".replace(".", ",")


def _przecinki(ax, osie: str = "y", nd: int = 3) -> None:
    """Przecinek dziesietny na podzialkach osi."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: _pl(v, nd))
    if "y" in osie:
        ax.yaxis.set_major_formatter(f)
    if "x" in osie:
        ax.xaxis.set_major_formatter(f)


def _naglowek(ax, tytul: str, *linie: str) -> None:
    """Tytul + podtytul w stalym rytmie -- ten sam na wszystkich rysunkach.

    Podtytul jest lamany RECZNIE (kolejne argumenty = kolejne wiersze), bo
    automatyczne lamanie zalezy od szerokosci figury i przy eksporcie do pracy
    potrafi wjechac na tytul albo poza obszar rysunku.
    """
    n = len(linie)
    ax.set_title(tytul, loc="left", fontweight="bold", pad=14 + 13 * n)
    for i, wiersz in enumerate(linie):
        ax.text(0.0, 1.02 + 0.052 * (n - 1 - i), wiersz, transform=ax.transAxes,
                fontsize=8.5, color=INK_2, va="bottom")


def _clean(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ rysunek 1


def rysunek_1_krzywa_nasycenia(v2: bool = False) -> Path:
    """v2=True: bez tytulu i podtytulu (tresc idzie do podpisu), 15,5 cm, 300 dpi, plik *_v2.png.
    Dane, wasy, skale i kolory takie same jak w wersji podstawowej."""
    d = _load(paths.ML_OUTPUTS / "echo_ablation" / "final_results_2026-08-13.json")
    pts = d["krzywa_stalego_budzetu"]["punkty"]
    ks = sorted(pts, key=int)
    x = np.array([int(k) for k in ks], dtype=float)
    y = np.array([pts[k]["mean"] for k in ks])
    e = np.array([pts[k]["sd"] or 0.0 for k in ks])
    if v2:
        print("rys_1 dane (K: srednia ± sd):", {int(k): f"{m:.5f} ± {sd:.5f}" for k, m, sd in zip(x, y, e)})

    fig, ax = plt.subplots(figsize=(15.5 / 2.54, 3.9) if v2 else (6.4, 4.0))
    _clean(ax)

    # Obszar nasycenia -- to jest WNIOSEK rysunku, wiec musi byc widoczny
    # zanim czytelnik zacznie odczytywac punkty.
    ax.axvspan(9, 12, color=C_MAIN, alpha=0.07, zorder=1)
    ax.annotate("nasycenie\nK = 9–12", xy=(10.4, y.max() - 0.004),
                ha="center", va="top", fontsize=9, color=INK_2)

    ax.errorbar(x, y, yerr=e, color=C_MAIN, linewidth=2, marker="o",
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                capsize=4, elinewidth=1.4, zorder=3)

    # Etykiety tylko przy koncach i przy punkcie odciecia -- nie przy kazdym
    # punkcie (patrz zasada "nigdy liczba nad kazdym markerem").
    # Konce krzywej w lewo/prawo od punktu, srodkowy w gore -- inaczej etykieta
    # K=4 siada na przerywanej linii zysku, a K=36 wychodzi poza obszar osi.
    for xi, yi, dx, dy, ha in ((x[0], y[0], 10, 9, "left"),
                               ((x[2], y[2], -9, -17, "right") if v2 else (x[2], y[2], 0, 14, "center")),
                               (x[-1], y[-1], 6, 5, "left")):
        ax.annotate(_pl(yi), xy=(xi, yi), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=9,
                    color=INK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(k)}" for k in x])
    ax.minorticks_off()
    ax.set_xlabel("liczba orientacji w siatce K, z której losowane są kąty")
    ax.set_ylabel(OS_RMSE)
    _przecinki(ax, "y", 3)
    if not v2:
        _naglowek(ax, "Nasycenie efektu gęstości kątowej przy stałym budżecie próbek",
                  "sieć głębi (tylko echo) · test na 36 orientacjach · 4 próbki na lokalizację "
                  "(5 496) w każdym punkcie",
                  "3 ziarna, wąsy = sd · liczby przy strzałkach to zmiana RMSE, ujemna = poprawa")

    # Strzalki zysku: cala wartosc rysunku to porownanie tych dwoch odcinkow.
    ax.annotate("", xy=(9, y[2]), xytext=(4, y[0]),
                arrowprops=dict(arrowstyle="-", color=C_ALT, lw=1.2, ls=(0, (4, 3))))
    ax.annotate(f"4 → 9:   −{_pl(y[0] - y[2])}", xy=(6.0, (y[0] + y[2]) / 2),
                xytext=(8, 7), textcoords="offset points",
                fontsize=9.5, color=C_ALT, fontweight="bold")
    # Adnotacja idzie NAD plaska czescia krzywej, nie pod nia: pod spodem
    # wchodzila na os X i na wasy punktu K=18.
    krotnosc = (y[0] - y[2]) / (y[2] - y[-1])
    ax.annotate(f"9 → 36:   −{_pl(y[2] - y[-1])}   ({_pl(krotnosc, 1)}× mniej)",
                xy=(20, y[3] + 0.013), ha="center", va="bottom",
                fontsize=9.5, color=INK_2)
    ax.set_ylim(y.min() - e[-1] - 0.014, y.max() + e[0] + 0.018)

    OUT.mkdir(parents=True, exist_ok=True)
    if v2:
        p = OUT / "rys_1_krzywa_nasycenia_v2.png"
        fig.tight_layout()
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            fig.savefig(p, dpi=300)
    else:
        p = OUT / "rys_1_krzywa_nasycenia.png"
        fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ rysunek 2


def rysunek_2_generalizacja_katowa(v2: bool = False) -> Path:
    """v2=True: bez tytulu i podtytulu, etykiety ED i osi X po nowemu, 15,5 cm, 300 dpi, plik *_v2.png.
    Dane, wasy, linie ED/EB, adnotacja luki i skale bez zmian."""
    d = _load(paths.ML_OUTPUTS / "echo_ablation" / "echo_3seeds.json")
    kr = d["krzywa_katowa_EA"]
    x = np.array(sorted(float(k) for k in kr))
    y = np.array([kr[str(int(k))]["mean"] for k in x])
    e = np.array([kr[str(int(k))]["sd"] for k in x])
    r = d["RMSE_test36"]

    if v2:
        print("rys_2 dane (odleglosc: srednia ± sd):", {int(k): f"{m:.5f} ± {e_:.5f}" for k, m, e_ in zip(x, y, e)},
              "| ED", f"{r['ED']['mean']:.5f}", "| EB", f"{r['EB']['mean']:.5f}")
    fig, ax = plt.subplots(figsize=(15.5 / 2.54, 3.9) if v2 else (6.4, 4.0))
    _clean(ax)

    # Warunki `ED` i `EB` NIE MAJA tej krzywej i to nie jest brak danych:
    # ucza sie na kątach z calej siatki, wiec KAZDY kat testowy lezy w
    # odleglosci 0 od ich siatki treningowej. Dlatego sa liniami odniesienia,
    # a nie seriami -- narysowanie ich jako krzywych sugerowaloby, ze zmierzono
    # cos, czego zmierzyc sie nie da.
    for val, col, name in ((r["ED"]["mean"], C_ALT,
                            "ED — 4 orientacje losowane z 36" if v2 else "ED — 4 kąty losowe z 36"),
                           (r["EB"]["mean"], C_THIRD, "EB — wszystkie 36")):
        ax.axhline(val, color=col, linewidth=2, ls=(0, (5, 3)), zorder=2)
        ax.annotate(f"{name}   {_pl(val)}", xy=(x[-1], val), xytext=(-2, 6),
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

    # Luka jest WNIOSKIEM rysunku, wiec stoi przy samej strzalce miedzy 0 a 40,
    # a nie jako luzny napis przy prawej krawedzi.
    ax.annotate("", xy=(x[0] + 1.2, y[-1]), xytext=(x[0] + 1.2, y[0]),
                arrowprops=dict(arrowstyle="<->", color=INK_2, lw=1.4))
    ax.annotate(f"luka {_pl(y[-1] - y[0], 3, znak=True)}\n"
                f"({_pl(100 * (y[-1] - y[0]) / y[0], 0)} %)",
                xy=(x[0] + 1.2, (y[0] + y[-1]) / 2), xytext=(8, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=9, color=INK_2, fontweight="bold")
    ax.set_ylim(min(r["EB"]["mean"], y.min()) - 0.020, y.max() + e[-1] + 0.026)

    ax.set_xticks(x)
    ax.set_xlabel("odległość orientacji testowej od najbliższej orientacji treningowej [°]" if v2
                  else "odległość testowanego kąta od najbliższego kąta treningowego [°]")
    ax.set_ylabel(OS_RMSE)
    _przecinki(ax, "y", 2)
    if not v2:
        _naglowek(ax, "Baseline 4-kierunkowy nie pokrywa przestrzeni orientacji",
                  "sieć głębi (tylko echo) · 3 ziarna, wąsy = sd",
                  "ED i EB są liniami odniesienia: pokrywają całą siatkę, więc nie mają tej osi")

    OUT.mkdir(parents=True, exist_ok=True)
    if v2:
        p = OUT / "rys_2_generalizacja_katowa_v2.png"
        fig.tight_layout()
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            fig.savefig(p, dpi=300)
    else:
        p = OUT / "rys_2_generalizacja_katowa.png"
        fig.savefig(p)
    plt.close(fig)
    return p


def rysunek_2b_krzywe_K() -> Path:
    """RMSE wg odleglosci od siatki treningowej dla EA (K=4) i EK6/9/12/18, test@36, 3 ziarna.

    Dane prosto z `eval/<warunek>_seed{0,1,2}/eval.json` (pole by_angular_distance), wasy = sd po
    ziarnach. ED i EB sa pominiete (nie maja osi odleglosci). Odleglosc = od najblizszego kata
    SIATKI K, nie od faktycznie wylosowanych katow lokalizacji. Punkty roznych serii sa
    przesuniete poziomo o <= 0,6 stopnia WYLACZNIE dla czytelnosci (przy 10 stopniach cztery serie
    maja prawie ta sama wartosc); podzialka osi X nie jest zmieniana. Bez tytulu (podpis w pracy).
    """
    serie = (("EA", "K = 4", C_MAIN, "o"), ("EK6", "K = 6", C_THIRD, "s"), ("EK9", "K = 9", C_ALT, "^"),
             ("EK12", "K = 12", "#7b52c9", "D"), ("EK18", "K = 18", INK_2, "v"))
    fig, ax = plt.subplots(figsize=(15.5 / 2.54, 3.9))
    _clean(ax)
    dodge = np.linspace(-0.6, 0.6, len(serie))
    for (c, lab, col, mk), dx in zip(serie, dodge):
        per = []
        for sd_ in (0, 1, 2):
            ev = _load(paths.ML_OUTPUTS / "eval" / f"{c}_seed{sd_}" / "eval.json")
            per.append(ev["test_sets"]["test@36"]["by_angular_distance"])
        dist = sorted(int(k) for k in per[0])
        m = np.array([np.mean([b[str(d)]["all"]["RMSE"] for b in per]) for d in dist])
        sd = np.array([np.std([b[str(d)]["all"]["RMSE"] for b in per], ddof=1) for d in dist])
        print(f"rys_2b {c}:", {d: f"{a:.5f} ± {b:.5f}" for d, a, b in zip(dist, m, sd)})
        ax.errorbar(np.array(dist) + dx, m, yerr=sd, color=col, linewidth=1.8, marker=mk, markersize=5.5,
                    markerfacecolor="white", markeredgewidth=1.6, capsize=3, elinewidth=1.2, zorder=3)
        off = {"EK12": (0, 17), "EK18": (13, -8)}.get(c, (9, 0))   # rozsuniecie etykiet przy 10 stopniach
        ax.annotate(f"{lab} ({c})" if c == "EA" else lab, xy=(dist[-1] + dx, m[-1]), xytext=off,
                    textcoords="offset points", ha="center" if c == "EK12" else "left", va="center",
                    fontsize=9, color=col,
                    fontweight="bold")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlim(-2.5, 49)
    ax.set_xlabel("odległość orientacji testowej od najbliższej orientacji siatki treningowej [°]")
    ax.set_ylabel(OS_RMSE)
    _przecinki(ax, "y", 2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "rys_2b_krzywe_K.png"
    fig.tight_layout()
    with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
        fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ rysunek 3


def rysunek_3_rozklad_efektu(v2: bool = False) -> Path:
    """v2=True: bez tytulu/podtytulu, etykieta \"liczba probek\" zamiast \"ilosc danych\", notka o skali
    jako maly opis przy prawym panelu, 15,5 cm, 300 dpi, plik *_v2.png. Dane i skale bez zmian."""
    dane = "liczba próbek" if v2 else "ilość danych"
    e3 = _load(paths.ML_OUTPUTS / "echo_ablation" / "echo_3seeds.json")["kontrasty"]
    gl = _load(paths.ML_OUTPUTS / "echo_ablation" / "final_results_2026-08-15.json")
    sk = gl["glowne_3ziarna"]["skladowe"]

    # Słupki pokazują WIELKOŚĆ poprawy (spadek RMSE), więc są dodatnie. Etykiety
    # opisują przejście (A → D), a nie różnicę `D − A`, bo ta w konwencji
    # z docs/KONWENCJE.md §2 jest ujemna. `echo_3seeds.json` trzyma te kontrasty
    # jako A − D / D − B (dodatnie, pod etykietami "D-A" / "B-D"); pełny model
    # trzyma poprawne D − A / B − D (ujemne) -- stąd abs() w obu panelach.
    panele = (
        ("sieć głębi (tylko echo)",
         [("gęstość kątowa\n(A → D)", abs(e3["gestosc katowa D-A"]["mean"]),
           e3["gestosc katowa D-A"]["sd_seeds"]),
          (f"{dane}\n(D → B)", abs(e3["ilosc danych B-D"]["mean"]),
           e3["ilosc danych B-D"]["sd_seeds"])],
         70.2),
        ("sieć głębi (pełna: obraz + echo)",
         [("gęstość kątowa\n(A → D)", abs(sk["gestosc_D_minus_A"]["mean"]),
           sk["gestosc_D_minus_A"]["sd"]),
          (f"{dane}\n(D → B)", abs(sk["ilosc_danych_B_minus_D"]["mean"]),
           sk["ilosc_danych_B_minus_D"]["sd"])],
         42.0),
    )

    if v2:
        for tyt, slupki, _ in panele:
            print("rys_3 dane", tyt, {s_[0].replace("\n", " "): f"{s_[1]:.5f} ± {s_[2]:.5f}" for s_ in slupki})
        krotnosc = panele[0][1][0][1] / panele[1][1][0][1]
    fig, axes = plt.subplots(1, 2, figsize=(15.5 / 2.54, 3.7) if v2 else (8.4, 4.0))
    for ax, (tytul, slupki, udzial) in zip(axes, panele):
        _clean(ax)
        xs = np.arange(len(slupki))
        vals = [s[1] for s in slupki]
        errs = [s[2] for s in slupki]
        ax.bar(xs, vals, yerr=errs, width=0.52,
               color=[C_MAIN, C_ALT], capsize=5,
               error_kw=dict(elinewidth=1.4, ecolor=INK_2), zorder=3)
        for xi, v, er in zip(xs, vals, errs):
            ax.annotate(_pl(v), xy=(xi, v + er), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=10, fontweight="bold", color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in slupki], fontsize=9)
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.30)
        # Obie osie podpisane TA SAMA nazwa, mimo ze to powtorzenie: panele maja
        # rozne skale, a wspolny podpis tylko po lewej sugerowalby wspolna os.
        ax.set_ylabel(f"spadek: {OS_RMSE}", fontsize=9)
        _przecinki(ax, "y", 3)
        if v2:   # dwuwierszowy tytul panelu: w jednym wierszu prawy tytul wychodzi poza figure
            tytul = tytul.replace(" (", "\n(", 1)
        ax.set_title(tytul, loc="left", fontsize=10, fontweight="bold", pad=26 if v2 else 22)
        ax.text(0.0, 1.02, f"udział gęstości: {_pl(udzial, 1)} %",
                transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    if v2:
        axes[1].text(0.03, 0.98, f"inna skala osi Y:\nefekt {_pl(krotnosc, 1)}× mniejszy",
                     transform=axes[1].transAxes, fontsize=9, color=INK_2, ha="left", va="top")

    if v2:
        fig.tight_layout(w_pad=1.5)
        p = OUT / "rys_3_rozklad_efektu_v2.png"
        OUT.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            fig.savefig(p, dpi=300)
        plt.close(fig)
        return p

    fig.suptitle("Prior wizualny przykrywa strukturę kątową echa",
                 x=0.008, y=0.995, ha="left", va="top", fontweight="bold", fontsize=11.5)
    fig.text(0.008, 0.955,
             "3 ziarna, wąsy = sd · UWAGA: panele mają różne skale osi Y — efekt "
             "w pełnym modelu jest 7,2× mniejszy",
             fontsize=8.5, color=INK_2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=2.5)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "rys_3_rozklad_efektu.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ rysunek 4


def rysunek_4_sonda_glebi() -> Path:
    d = _load(paths.ML_OUTPUTS / "echo_ablation" / "final_results_2026-08-15.json")["sonda_glebi"]
    P = d["punkty"]
    # p-wartosci czytane z pliku dowodowego, nie wpisane recznie -- przy
    # przeliczeniu macierzy rysunek ma sie zmienic razem z liczbami.
    rozklad = d["rozklad_przewagi_K36_nad_K4"]
    rnd = np.array(P["random"]["values"], dtype=float)
    span = P["random"]["mean"] - P["depth_trained"]["mean"]

    # Os X to POPRAWA WOBEC LOSOWEGO, a nie surowe RMSE. Powod: przy RMSE slupki
    # rosna w strone GORSZEGO wyniku, co przy rysunku "ile informacji o glebi
    # niesie koder" czyta sie odwrotnie do intencji. Po odjeciu od podlogi
    # dluzszy slupek = wiecej informacji, a zero ma znaczenie (= koder losowy).
    def poprawa(e):
        v = np.array(P[e]["values"], dtype=float)
        # Blad standardowy ROZNICY dwoch niezaleznych srednich, nie sd jednej
        # z nich -- slupek bledu dotyczy wielkosci, ktora jest narysowana.
        se = float(np.sqrt(v.std(ddof=1) ** 2 / v.size + rnd.std(ddof=1) ** 2 / rnd.size))
        return float(rnd.mean() - v.mean()), se

    encs = [("pretext_K4", "pretrening K=4\n16 par/lok."),
            ("pretext_K36_p16", "pretrening K=36 @ 16 par\n16 par/lok., siatka 36"),
            ("pretext_K36", "pretrening K=36\n1 296 par/lok.")]
    vals, errs = zip(*(poprawa(e) for e, _ in encs))
    y = np.arange(len(encs))

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    _clean(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Gorna granica: koder, ktory PRZESZEDL pelne uczenie na glebi. Etykieta
    # PIONOWO przy samej linii -- poziomo nie miesci sie miedzy linia a krawedzia
    # i wypada poza obszar rysunku.
    ax.axvline(span, color=INK_2, linewidth=2, ls=(0, (5, 3)), zorder=2)
    # Etykieta PIONOWO przy samej linii i przy DOLNEJ czesci osi: poziomo nie
    # miesci sie miedzy linia a krawedzia, a wysrodkowana pionowo wchodzila
    # na podtytul.
    ax.annotate("koder uczony na głębi (100 %)",
                xy=(span, -0.34), xytext=(-10, 0),
                textcoords="offset points", rotation=90, ha="center", va="bottom",
                fontsize=8.5, color=INK_2, fontweight="bold", annotation_clip=False)

    # Slupki wezsze niz domyslnie: miedzy nimi musi zmiescic sie strzalka
    # rozkladu z dwuwierszowym opisem.
    ax.barh(y, vals, xerr=errs, height=0.44, color=C_MAIN, capsize=5,
            error_kw=dict(elinewidth=1.4, ecolor=INK_2), zorder=3)
    for yi, v, e in zip(y, vals, errs):
        ax.annotate(f"{_pl(100 * v / span, 1)} %", xy=(v + e, yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10,
                    fontweight="bold", color=INK)

    # Rozklad: co dokłada gestosc katowa, a co sama liczba par.
    opisy = ((("gęstość kątowa", vals[1] - vals[0], rozklad["gestosc_katowa_danych"]["p"]), C_ALT),
             (("liczba par", vals[2] - vals[1], rozklad["liczba_par"]["p"]), C_THIRD))
    for i, ((nazwa, delta, pval), col) in enumerate(opisy):
        lab = f"{nazwa}\n+{_pl(100 * delta / span, 1)} pp   p = {_pl(pval, 4)}"
        x0, x1 = vals[i], vals[i + 1]
        ax.annotate("", xy=(x1, i + 0.58), xytext=(x0, i + 0.58),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.6))
        # Opis POD strzalka, nie nad nia: nad strzalka wchodzil na slupek
        # lezacy wyzej.
        ax.annotate(lab, xy=((x0 + x1) / 2, i + 0.58), xytext=(0, -9),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8.5, color=col, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n in encs], fontsize=9)
    ax.set_xlim(0, span * 1.13)
    _przecinki(ax, "x", 2)
    ax.set_xlabel(f"spadek: {OS_RMSE} względem kodera losowego (0 = podłoga)")
    _naglowek(ax, "Zamrożony koder z pretreningu niesie informację o głębi",
              "sonda: koder zamrożony, uczony wyłącznie dekoder",
              "3 ziarna, wąsy = błąd standardowy różnicy · pp = punkty procentowe rozpiętości")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "rys_4_sonda_glebi.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    _style()
    for fn in (rysunek_1_krzywa_nasycenia, rysunek_2_generalizacja_katowa,
               rysunek_3_rozklad_efektu, rysunek_4_sonda_glebi):
        print(f"zapisano: {fn()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
