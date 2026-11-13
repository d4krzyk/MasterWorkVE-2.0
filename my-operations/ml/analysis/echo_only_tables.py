#!/usr/bin/env python
"""Tabele §5.3-5.5 (siec glebi tylko-echo, `echo2depth`) z surowych `eval.json`. Zero GPU.

    python my-operations/ml/analysis/echo_only_tables.py

Czyta WYLACZNIE `outputs/ml/eval/<warunek>_seed{0,1,2}/eval.json` (jawne nazwy, BEZ globow --
przebiegi `_nf`, `_reeval` i `mask-*` nie wchodza) oraz stala rozstepu z
`outputs/ml/determinism/noise_floor_ED_40k.json`. Zapisuje do nowego katalogu
`outputs/ml/tables_5_3_5_5/` (`tables.json`, `tables.md`); zadnego istniejacego pliku nie zmienia.

Konwencje: etykieta `X − Y` znaczy dosłownie X − Y (docs/KONWENCJE.md §2). Test t na roznicach po
ziarnach (to samo ziarno = ta sama inicjalizacja) to `scipy.stats.ttest_1samp(d, 0)`; porownania
niezaleznych przebiegow (rozne warunki z roznych serii) to Welch `ttest_ind(equal_var=False)`.
Wszystkie metryki: warstwa `all` (bez masek krawedz/gladkie), RMSE glebi [m], zbior test@36.

Na koncu skrypt porownuje wynik z liczbami z STAN_WYNIKOW.md (`KONTROLA`) i WYPISUJE niezgodnosci;
niczego nie poprawia.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402

SEEDS = (0, 1, 2)
EVAL = paths.ML_OUTPUTS / "eval"
OUT = paths.ML_OUTPUTS / "tables_5_3_5_5"
NOISE = paths.ML_OUTPUTS / "determinism" / "noise_floor_ED_40k.json"
CONDS = ("EA", "EK6", "EK9", "EK12", "EK18", "ED", "EB", "ESE")
KGRID = (("EA", 4), ("EK6", 6), ("EK9", 9), ("EK12", 12), ("EK18", 18), ("ED", 36))


# ------------------------------------------------------------------ wczytywanie

def _load(cond: str, seed: int) -> dict:
    p = EVAL / f"{cond}_seed{seed}" / "eval.json"          # jawna nazwa, bez globow
    d = json.loads(p.read_text(encoding="utf-8"))
    if d["condition"] != cond or int(d["seed"]) != seed:
        raise ValueError(f"{p}: condition/seed w pliku = {d['condition']}/{d['seed']}")
    return d


DATA = {c: {s: _load(c, s) for s in SEEDS} for c in CONDS}


def rmse(cond: str, ts: str = "test@36", scene: str | None = None) -> np.ndarray:
    out = []
    for s in SEEDS:
        t = DATA[cond][s]["test_sets"][ts]
        blok = t["overall"] if scene is None else t["per_scene"][scene]
        out.append(blok["all"]["RMSE"])
    return np.array(out, dtype=float)


def by_dist(cond: str) -> dict[int, dict]:
    """{odleglosc: {"n": liczba probek, "vals": RMSE per ziarno}} -- odleglosc od siatki treningowej."""
    first = DATA[cond][SEEDS[0]]["test_sets"]["test@36"]["by_angular_distance"]
    out = {}
    for k in sorted(first, key=int):
        ns = {DATA[cond][s]["test_sets"]["test@36"]["by_angular_distance"][k]["n_samples"] for s in SEEDS}
        if len(ns) != 1:
            raise ValueError(f"{cond}: rozna liczba probek w kubelku {k} miedzy ziarnami: {ns}")
        out[int(k)] = {"n": ns.pop(),
                       "vals": np.array([DATA[cond][s]["test_sets"]["test@36"]["by_angular_distance"][k]
                                         ["all"]["RMSE"] for s in SEEDS], dtype=float)}
    return out


# ------------------------------------------------------------------ pomocnicze

def ms(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"values": [float(x) for x in v], "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else None}


def paired(x, y) -> dict:
    """x − y per ziarno + test t na roznicach (ttest_1samp)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    t = stats.ttest_1samp(d, 0.0)
    return {**ms(d), "t": float(t.statistic), "p": float(t.pvalue), "df": int(d.size - 1)}


def welch(x, y) -> dict:
    """x − y z roznicy srednich, test Welcha (niezalezne probki)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    t = stats.ttest_ind(x, y, equal_var=False)
    return {"delta": float(x.mean() - y.mean()), "t": float(t.statistic),
            "df": float(t.df), "p": float(t.pvalue)}


# ------------------------------------------------------------------ 5.3

def tabela_5_3() -> dict:
    R = {c: rmse(c) for c in ("EA", "ED", "EB", "ESE")}
    out = {"RMSE_test36": {c: ms(v) for c, v in R.items()}}
    ctr = {"D−A": ("ED", "EA"), "B−D": ("EB", "ED"), "B−A": ("EB", "EA"), "ESE−EB": ("ESE", "EB")}
    out["kontrasty"] = {lab: paired(R[a], R[b]) for lab, (a, b) in ctr.items()}
    da = R["ED"] - R["EA"]
    ba = R["EB"] - R["EA"]
    out["udzial_gestosci_pct"] = {
        "opis": "(D−A)/(B−A) × 100 per ziarno",
        **ms(100 * da / ba),
        "z_srednich": float(100 * da.mean() / ba.mean())}
    out["spadek_wzgledny_pct"] = {
        "opis": "100·(X−Y)/Y; z ŚREDNICH po ziarnach oraz per ziarno",
        "D−A": {"z_srednich": float(100 * (R["ED"].mean() - R["EA"].mean()) / R["EA"].mean()),
                "per_ziarno": [float(x) for x in 100 * (R["ED"] - R["EA"]) / R["EA"]]},
        "B−D": {"z_srednich": float(100 * (R["EB"].mean() - R["ED"].mean()) / R["ED"].mean()),
                "per_ziarno": [float(x) for x in 100 * (R["EB"] - R["ED"]) / R["ED"]]},
        "B−A": {"z_srednich": float(100 * (R["EB"].mean() - R["EA"].mean()) / R["EA"].mean()),
                "per_ziarno": [float(x) for x in 100 * (R["EB"] - R["EA"]) / R["EA"]]}}
    scenes = list(DATA["EA"][0]["test_sets"]["test@36"]["per_scene"])
    ps = {}
    for sc in scenes:
        a, d = rmse("EA", scene=sc), rmse("ED", scene=sc)
        ps[sc] = {"EA": ms(a), "ED": ms(d), "D−A": paired(d, a)}
    out["D−A_per_scena"] = ps
    return out


# ------------------------------------------------------------------ 5.4

def tabela_5_4() -> dict:
    floor = json.loads(NOISE.read_text(encoding="utf-8"))["rozstep"]
    R = {k: rmse(c) for c, k in KGRID}
    gain = R[4].mean() - R[36].mean()
    rows = []
    prev = None
    for c, k in KGRID:
        v = R[k]
        row = {"K": k, "warunek": c, "RMSE": ms(v)}
        row["K−K36"] = None if k == 36 else welch(v, R[36])
        row["udzial_zysku_4_36_pct"] = float(100 * (R[4].mean() - v.mean()) / gain)
        if prev is not None:
            step = float(v.mean() - R[prev].mean())
            row["krok_od_poprzedniego_K"] = {"od_K": prev, "delta": step,
                                              "krotnosc_rozstepu": float(abs(step) / floor)}
        rows.append(row)
        prev = k
    return {"rozstep_podlogi_szumu_GPU": floor, "zrodlo_rozstepu": str(NOISE.relative_to(paths.REPO_ROOT)),
            "zysk_4_36": float(gain), "wiersze": rows,
            "uwaga": "K−K36: Welch (3 vs 3 ziarna) na RMSE test@36; K=36 to warunek ED"}


# ------------------------------------------------------------------ 5.5

def tabela_5_5() -> dict:
    out = {"uwaga": "odleglosc = od najblizszego kata SIATKI treningowej K (nie od faktycznie "
                    "wylosowanych 4 katow lokalizacji), pole by_angular_distance z eval.json"}
    krzywe = {}
    for c in ("EA", "EK6", "EK9", "EK12", "EK18"):
        b = by_dist(c)
        v0 = b[0]["vals"]
        krzywe[c] = {str(d): {"n_probek": x["n"], "RMSE": ms(x["vals"]),
                              "kara_wzgledem_0": None if d == 0 else paired(x["vals"], v0)}
                     for d, x in b.items()}
        dmax = max(b)
        gap = b[dmax]["vals"] - v0
        mono = []
        for i, s in enumerate(SEEDS):
            seq = [b[d]["vals"][i] for d in sorted(b)]
            mono.append(bool(all(x < y for x, y in zip(seq, seq[1:]))))
        krzywe[c]["_luka"] = {"od_0_do": dmax, **ms(gap), "monotoniczna_scisle_per_ziarno": mono,
                              "monotoniczna_w_kazdym_ziarnie": all(mono),
                              "flaga_monotonic_z_eval_json": [
                                  DATA[c][s]["test_sets"]["test@36"]["angular_gap"]["monotonic"]
                                  for s in SEEDS]}
    out["krzywe"] = krzywe
    out["test36_minus_test4"] = {
        c: ms(rmse(c, "test@36") - rmse(c, "test@4")) for c in ("EA", "ED", "EB")}
    d36 = rmse("EA", "test@36") - rmse("EB", "test@36")
    d4 = rmse("EA", "test@4") - rmse("EB", "test@4")
    share = 100 * (1 - d4 / d36)
    out["EA−EB"] = {"test@36": ms(d36), "test@4": ms(d4),
                    "udzial_kary_na_niewidzianych_pct": {
                        "opis": "100·(1 − (EA−EB)@test4 / (EA−EB)@test36)",
                        "per_ziarno": [float(x) for x in share],
                        "z_srednich": float(100 * (1 - d4.mean() / d36.mean()))}}
    return out


# ------------------------------------------------------------------ kontrola

def kontrola(T: dict) -> list[dict]:
    t53, t54, t55 = T["5.3"], T["5.4"], T["5.5"]
    k9 = next(r for r in t54["wiersze"] if r["K"] == 9)["K−K36"]
    oczek = [
        ("EA test@36 (średnia)", 0.79104, t53["RMSE_test36"]["EA"]["mean"], 5e-6),
        ("ED test@36 (średnia)", 0.64432, t53["RMSE_test36"]["ED"]["mean"], 5e-6),
        ("EB test@36 (średnia)", 0.58223, t53["RMSE_test36"]["EB"]["mean"], 5e-6),
        ("ESE test@36 (średnia)", 1.16292, t53["RMSE_test36"]["ESE"]["mean"], 5e-6),
        ("D−A (średnia)", -0.14672, t53["kontrasty"]["D−A"]["mean"], 5e-6),
        ("D−A p (ttest_1samp)", 0.0026, t53["kontrasty"]["D−A"]["p"], 5e-5),
        ("udział gęstości, średnia [%]", 70.2, t53["udzial_gestosci_pct"]["mean"], 0.05),
        ("udział gęstości, sd [pp]", 3.0, t53["udzial_gestosci_pct"]["sd"], 0.05),
        ("K=9 − K=36 (Welch Δ)", 0.01910, k9["delta"], 5e-6),
        ("K=9 − K=36 (Welch p)", 0.0032, k9["p"], 5e-5),
        ("luka EA 0→40° (średnia)", 0.31477, t55["krzywe"]["EA"]["_luka"]["mean"], 5e-6),
        ("luka EA 0→40° (sd)", 0.02860, t55["krzywe"]["EA"]["_luka"]["sd"], 5e-6),
        ("ED test@36−test@4", 0.00691, t55["test36_minus_test4"]["ED"]["mean"], 5e-6),
        ("udział 91,4 % (z średnich)", 91.4, t55["EA−EB"]["udzial_kary_na_niewidzianych_pct"]["z_srednich"], 0.05),
    ]
    return [{"wielkosc": n, "oczekiwane": e, "obliczone": float(o), "tolerancja": tol,
             "zgodne": bool(abs(e - o) <= tol)} for n, e, o, tol in oczek]


# ------------------------------------------------------------------ render

def f5(x) -> str:
    return "—" if x is None else f"{x:.5f}"


def _pm(d: dict) -> str:
    return f"{d['mean']:.5f} ± {d['sd']:.5f}"


def _o(d, key: str, spec: str) -> str:
    return "—" if d is None else format(d[key], spec)


def _p(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def render_md(T: dict, K: list[dict]) -> str:
    L = ["# Tabele §5.3–5.5 — sieć głębi (tylko echo), test@36", "",
         "Generowane przez `my-operations/ml/analysis/echo_only_tables.py` z "
         "`outputs/ml/eval/<warunek>_seed{0,1,2}/eval.json`. Nie edytować ręcznie.",
         "Konwencja: `X − Y` = dosłownie X − Y (ujemne = mniejszy błąd = lepiej). RMSE głębi [m], "
         "średnia ± sd po 3 ziarnach. t i p: `ttest_1samp` na różnicach per ziarno (df = 2), "
         "o ile nie zaznaczono Welcha.", ""]
    t = T["5.3"]
    L += ["## 5.3 Warunki i kontrasty", "", "| warunek | RMSE | per ziarno |", "|---|---|---|"]
    for c, v in t["RMSE_test36"].items():
        L.append(f"| {c} | {_pm(v)} | {' · '.join(f5(x) for x in v['values'])} |")
    L += ["", "| kontrast | średnia ± sd | per ziarno | t | p |", "|---|---|---|---|---|"]
    for lab, v in t["kontrasty"].items():
        L.append(f"| {lab} | {v['mean']:+.5f} ± {v['sd']:.5f} | "
                 f"{' · '.join(f'{x:+.5f}' for x in v['values'])} | {v['t']:+.2f} | {_p(v['p'])} |")
    u = t["udzial_gestosci_pct"]
    L += ["", f"**Udział gęstości (D−A)/(B−A):** per ziarno {' · '.join(f'{x:.1f}' for x in u['values'])} %; "
              f"średnia ± sd = **{u['mean']:.1f} ± {u['sd']:.1f} %**; z średnich {u['z_srednich']:.1f} %.", "",
          "**Spadek względny** (z średnich; w nawiasie per ziarno):", ""]
    for lab, v in t["spadek_wzgledny_pct"].items():
        if lab == "opis":
            continue
        L.append(f"- {lab}: {v['z_srednich']:+.2f} % ({' · '.join(f'{x:+.2f}' for x in v['per_ziarno'])})")
    L += ["", "**D−A per scena testowa** (ED − EA):", "",
          "| scena | EA | ED | D−A | per ziarno | t | p |", "|---|---|---|---|---|---|---|"]
    for sc, v in t["D−A_per_scena"].items():
        d = v["D−A"]
        L.append(f"| {sc} | {_pm(v['EA'])} | {_pm(v['ED'])} | {d['mean']:+.5f} ± {d['sd']:.5f} | "
                 f"{' · '.join(f'{x:+.5f}' for x in d['values'])} | {d['t']:+.2f} | {_p(d['p'])} |")

    t = T["5.4"]
    L += ["", "## 5.4 Krzywa stałego budżetu", "",
          f"Rozstęp podłogi szumu GPU: **{t['rozstep_podlogi_szumu_GPU']:.5f}** (`{t['zrodlo_rozstepu']}`). "
          f"Zysk 4→36: {t['zysk_4_36']:.5f}. K−K36: Welch 3 vs 3.", "",
          "| K | warunek | RMSE | K − K36 | t | df | p | udział zysku 4→36 | krok od poprzedniego K | krotność rozstępu |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["wiersze"]:
        w = r["K−K36"]
        st = r.get("krok_od_poprzedniego_K")
        c_d, c_t = _o(w, "delta", "+.5f"), _o(w, "t", "+.2f")
        c_df, c_p = _o(w, "df", ".2f"), ("—" if w is None else _p(w["p"]))
        c_st = "—" if st is None else _o(st, "delta", "+.5f") + " (od K=" + str(st["od_K"]) + ")"
        c_kr = "—" if st is None else _o(st, "krotnosc_rozstepu", ".1f") + "×"
        L.append(f"| {r['K']} | {r['warunek']} | {_pm(r['RMSE'])} | {c_d} | {c_t} | {c_df} | {c_p} | "
                 f"{r['udzial_zysku_4_36_pct']:.1f} % | {c_st} | {c_kr} |")

    t = T["5.5"]
    L += ["", "## 5.5 RMSE wg odległości od siatki treningowej", "", t["uwaga"] + ".", ""]
    for c, kr in t["krzywe"].items():
        L += [f"**{c}**", "", "| odległość | próbek | RMSE | kara względem 0° | t | p |", "|---|---|---|---|---|---|"]
        for d, v in kr.items():
            if d == "_luka":
                continue
            kara = v["kara_wzgledem_0"]
            if kara is None:
                c_k, c_t, c_p = "—", "—", "—"
            else:
                c_k = f"{kara['mean']:+.5f} ± {kara['sd']:.5f}"
                c_t, c_p = f"{kara['t']:+.2f}", _p(kara["p"])
            L.append(f"| {d}° | {v['n_probek']} | {_pm(v['RMSE'])} | {c_k} | {c_t} | {c_p} |")
        g = kr["_luka"]
        L += ["", f"Luka 0°→{g['od_0_do']}°: **{g['mean']:+.5f} ± {g['sd']:.5f}** "
                  f"(per ziarno {' · '.join(f'{x:+.5f}' for x in g['values'])}); monotoniczna ściśle w każdym "
                  f"ziarnie: **{g['monotoniczna_w_kazdym_ziarnie']}** {g['monotoniczna_scisle_per_ziarno']}.", ""]
    L += ["**test@36 − test@4:**", "", "| warunek | średnia ± sd | per ziarno |", "|---|---|---|"]
    for c, v in t["test36_minus_test4"].items():
        L.append(f"| {c} | {v['mean']:+.5f} ± {v['sd']:.5f} | {' · '.join(f'{x:+.5f}' for x in v['values'])} |")
    e = t["EA−EB"]
    sh = e["udzial_kary_na_niewidzianych_pct"]
    L += ["", f"**EA − EB:** test@36 = {e['test@36']['mean']:+.5f} ± {e['test@36']['sd']:.5f}; "
              f"test@4 = {e['test@4']['mean']:+.5f} ± {e['test@4']['sd']:.5f}.",
          f"**Udział kary na niewidzianych kątach** 1 − (EA−EB)@4/(EA−EB)@36: per ziarno "
          f"{' · '.join(f'{x:.1f}' for x in sh['per_ziarno'])} %; z średnich **{sh['z_srednich']:.1f} %**.", "",
          "## Kontrola względem liczb z `STAN_WYNIKOW.md`", "",
          "| wielkość | oczekiwane | obliczone | |", "|---|---|---|---|"]
    for k in K:
        L.append(f"| {k['wielkosc']} | {k['oczekiwane']} | {k['obliczone']:.5f} | "
                 f"{'zgodne' if k['zgodne'] else '**NIEZGODNE**'} |")
    return "\n".join(L) + "\n"


def main() -> int:
    T = {"5.3": tabela_5_3(), "5.4": tabela_5_4(), "5.5": tabela_5_5()}
    K = kontrola(T)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.json").write_text(json.dumps({"tabele": T, "kontrola": K}, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    (OUT / "tables.md").write_text(render_md(T, K), encoding="utf-8")
    print(f"zapisano: {OUT / 'tables.json'}\n          {OUT / 'tables.md'}\n")
    print("KONTROLA wobec STAN_WYNIKOW.md:")
    for k in K:
        print(f"  [{'OK ' if k['zgodne'] else 'NIEZGODNE'}] {k['wielkosc']:<34s} oczekiwane {k['oczekiwane']:<9} "
              f"obliczone {k['obliczone']:.5f}")
    bad = [k for k in K if not k["zgodne"]]
    print(f"\nniezgodnosci: {len(bad)}" + ("" if not bad else " -> " + "; ".join(k["wielkosc"] for k in bad)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
