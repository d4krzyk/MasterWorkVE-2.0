#!/usr/bin/env python
"""MODEL 2: zestawienie wynikow w dwie tabele do pracy. ZERO GPU.

    python my-operations/ml/pretext_model/summarize.py

Czyta to, co zostawily `train_pretext.py` i `transfer.py`, i sklada:

  TABELA A -- jakosc samego zadania pretekstowego w funkcji K.
              Metryka porownywalna to MAAE, NIE trafnosc top-1: poziom losowy
              top-1 spada z 25 % przy K=4 do 2,8 % przy K=36, wiec sam spadek
              trafnosci nic nie mowi o tym, czy zadanie zostalo rozwiazane
              gorzej. MAAE poziomu losowego wynosi 90 stopni niezaleznie od K.

  TABELA B -- zadanie DOCELOWE (RGB2Depth bez audio) w funkcji inicjalizacji
              enkodera. To jest liczba do pracy.

Kolumna "odniesienie Gao" jest wypelniana TYLKO dla `Scratch` i K=4, bo tylko te
dwa warunki Gao raportuje (tabela 3: 0,360 i 0,332). NIE jest to baseline do
przepisania -- silnik akustyczny jest inny, a `geometry_check.py` pokazal, ze sam
wariant geometrii zmienia energie pozna o 46 %. Sluzy wylacznie do sprawdzenia,
czy odtwarzamy wlasciwy PORZADEK warunkow i rzad wielkosci efektu.

Rozbicie z punktu 4.6 (MAAE osobno dla przesuniec <= 20 i > 20 stopni) jest
raportowane zawsze, bo to ono wyznacza faktyczna granice rozdzielczosci metody.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.pretext_model"

from .. import paths  # noqa: E402
from .metrics import FINE_SHIFT_LIMIT_DEG, TOLERANCES_DEG  # noqa: E402

# Liczby, ktore Gao faktycznie raportuje dla Repliki (tabela 3 pracy glownej).
# Wpisane wylacznie jako kolumna odniesienia -- patrz naglowek modulu.
GAO_REFERENCE = {"scratch": 0.360, "K4": 0.332}

_RUN_RE = re.compile(r"^pretext_K(\d+)(?:_p(\d+))?_seed(\d+)$")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_pretext(root: Path) -> list[dict]:
    rows = []
    for d in sorted(root.glob("pretext_*")):
        m = _RUN_RE.match(d.name)
        if not m:
            continue
        best = _load(d / "best.json")
        status = _load(d / "status.json") or {}
        if not best:
            continue
        k, ppl, seed = int(m.group(1)), m.group(2), int(m.group(3))
        rows.append({
            "run": d.name, "K": k, "seed": seed,
            "pairs_per_location": int(ppl) if ppl else k * k,
            "subsampled": bool(ppl),
            "MAAE_deg": best.get("MAAE_deg"),
            "top1": best.get("top1"),
            "top1_chance": best.get("top1_chance"),
            **{f"acc_within_{t}deg": best.get(f"acc_within_{t}deg") for t in TOLERANCES_DEG},
            "MAAE_fine": (best.get("by_true_shift", {})
                          .get(f"fine_le_{FINE_SHIFT_LIMIT_DEG}deg", {}).get("MAAE_deg")),
            "MAAE_coarse": (best.get("by_true_shift", {})
                            .get(f"coarse_gt_{FINE_SHIFT_LIMIT_DEG}deg", {}).get("MAAE_deg")),
            "errors_adjacent": best.get("errors_to_adjacent_class_fraction"),
            "best_step": status.get("best_step"),
            "budget_warning": status.get("budget_ceiling_warning"),
            "finished": status.get("finished"),
        })
    return rows


def collect_transfer(root: Path) -> list[dict]:
    rows = []
    for d in sorted(root.glob("transfer_*")):
        best = _load(d / "best.json")
        cfg = _load(d / "config.json") or {}
        status = _load(d / "status.json") or {}
        if not best:
            continue
        rows.append({
            "run": d.name,
            "label": cfg.get("label", d.name),
            "seed": cfg.get("seed"),
            "init": cfg.get("init"),
            "RMSE": best.get("all", {}).get("RMSE"),
            "RMSE_edge": best.get("edge", {}).get("RMSE"),
            "RMSE_smooth": best.get("smooth", {}).get("RMSE"),
            "DELTA1": best.get("all", {}).get("DELTA1"),
            "transfer_ok": cfg.get("transfer", {}).get("ok"),
            "n_loaded": cfg.get("transfer", {}).get("n_loaded"),
            "best_step": status.get("best_step"),
            "budget_warning": status.get("budget_ceiling_warning"),
            "finished": status.get("finished"),
        })
    return rows


def _agg(rows: list[dict], key: str, field: str) -> dict:
    """Srednia i odchylenie po ziarnach. Pojedyncze ziarno -> sd = None, a nie 0:
    z n=1 nie da sie oszacowac rozrzutu i udawanie zera bylo by falszem."""
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    res = {}
    for k, group in out.items():
        vals = [g[field] for g in group if g.get(field) is not None]
        if not vals:
            continue
        res[k] = {
            "n_seeds": len(vals),
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "values": [float(v) for v in vals],
        }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    pre = collect_pretext(paths.ML_OUTPUTS / "pretext")
    tra = collect_transfer(paths.ML_OUTPUTS / "pretext_transfer")

    print("=" * 92)
    print("TABELA A -- zadanie pretekstowe (MAAE jest metryka porownywalna miedzy K, top-1 NIE JEST)")
    print("=" * 92)
    if not pre:
        print("  (brak przebiegow pretreningu w outputs/ml/pretext_model/)")
    else:
        print(f"  {'wariant':16s} {'ziarn':>5s} {'MAAE':>8s} {'losowo':>7s} {'top1':>7s} "
              f"{'losowo':>7s} {'+/-30':>7s} {'MAAE<=20':>9s} {'MAAE>20':>8s} {'sasiad':>7s}")
        for r in sorted(pre, key=lambda x: (x["K"], x["subsampled"], x["seed"])):
            tag = f"K{r['K']}" + (f"@{r['pairs_per_location']}par" if r["subsampled"] else "")
            warn = "  [SUFIT BUDZETU]" if r.get("budget_warning") else ""
            print(f"  {tag:16s} {r['seed']:>5d} {r['MAAE_deg']:>8.2f} {90.0:>7.1f} "
                  f"{r['top1']*100:>6.1f}% {r['top1_chance']*100:>6.1f}% "
                  f"{r['acc_within_30deg']*100:>6.1f}% {r['MAAE_fine']:>9.2f} "
                  f"{r['MAAE_coarse']:>8.2f} {r['errors_adjacent']*100:>6.1f}%{warn}")

    print()
    print("=" * 92)
    print("TABELA B -- zadanie DOCELOWE: RGB2Depth bez audio (to jest liczba do pracy)")
    print("=" * 92)
    if not tra:
        print("  (brak przebiegow zadania docelowego w outputs/ml/pretext_transfer/)")
    else:
        agg = _agg(tra, "label", "RMSE")
        print(f"  {'inicjalizacja enkodera':28s} {'ziarn':>5s} {'RMSE':>9s} {'sd':>8s} "
              f"{'krawedzie':>10s} {'odniesienie Gao':>16s}")
        for label in sorted(agg, key=lambda l: agg[l]["mean"]):
            a = agg[label]
            edge = np.mean([r["RMSE_edge"] for r in tra
                            if r["label"] == label and r["RMSE_edge"] is not None])
            ref = GAO_REFERENCE.get("scratch" if "scratch" in label.lower() else
                                    ("K4" if re.search(r"_K4_", label) else ""), None)
            sd = f"{a['sd']:.5f}" if a["sd"] is not None else "   n=1"
            print(f"  {label:28s} {a['n_seeds']:>5d} {a['mean']:>9.5f} {sd:>8s} "
                  f"{edge:>10.5f} {(f'{ref:.3f}' if ref else '—'):>16s}")
        # `scratch` z definicji nie przenosi wag, wiec nie moze "nie przeniesc" --
        # ostrzezenie dotyczy wylacznie przebiegow, ktore mialy wczytac enkoder.
        bad = [r for r in tra
               if r.get("transfer_ok") is False and str(r.get("init", "")).lower() != "scratch"]
        if bad:
            print(f"\n  UWAGA: {len(bad)} przebieg(ow) z NIEUDANYM przeniesieniem wag "
                  f"-- ich wynik jest nieodroznialny od 'scratch': "
                  f"{[r['run'] for r in bad]}")

    # Macierz pomylek najlepszego przebiegu kazdego K -- oczekiwanie: bledy
    # skupione przy przekatnej (klasy sasiednie). Trzymana w JSON-ie, NIE
    # drukowana: przy K=36 to 1 296 liczb.
    cm = {}
    for d in sorted((paths.ML_OUTPUTS / "pretext").glob("pretext_*")):
        best = _load(d / "best.json")
        if best and best.get("confusion_matrix"):
            cm[d.name] = {"K": best.get("K"), "matrix": best["confusion_matrix"],
                          "errors_to_adjacent_fraction": best.get("errors_to_adjacent_class_fraction")}

    # Rozklad efektu wg 4.4: K36 - K36@16par izoluje ILOSC PAR,
    # K36@16par - K4 izoluje SAMA ROZDZIELCZOSC KATOWA zadania.
    by_lab = _agg(tra, "label", "RMSE")
    def pick(frag):
        for k in by_lab:
            if frag in k:
                return by_lab[k]["mean"]
        return None
    k4, k36, k36p = pick("K4_"), None, pick("K36_p16")
    for k in by_lab:
        if "K36_" in k and "p16" not in k:
            k36 = by_lab[k]["mean"]
    rozklad = {}
    if None not in (k4, k36, k36p):
        rozklad = {"ilosc_par_K36_minus_K36p16": k36 - k36p,
                   "rozdzielczosc_K36p16_minus_K4": k36p - k4,
                   "laczny_K36_minus_K4": k36 - k4,
                   "uwaga": "wartosci UJEMNE znacza poprawe (nizsze RMSE)"}

    payload = {"pretext": pre, "transfer": tra,
               "pretext_by_K": _agg(pre, "K", "MAAE_deg"),
               "transfer_by_label": by_lab,
               "confusion_matrices": cm,
               "rozklad_efektu_pretreningu": rozklad,
               "gao_reference": GAO_REFERENCE,
               "uwaga": "kolumna odniesienia Gao NIE jest baseline'em -- inny silnik akustyczny; "
                        "sluzy do sprawdzenia porzadku warunkow i rzedu wielkosci efektu"}
    out = args.out or (paths.ML_OUTPUTS / "pretext" / "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nzapisano: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
