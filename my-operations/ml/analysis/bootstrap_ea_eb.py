#!/usr/bin/env python
"""Bootstrap sparowany po LOKALIZACJACH: EA vs EB, ziarna 0/1/2, test@36, n_boot=2000. Zero GPU.

    python my-operations/ml/analysis/bootstrap_ea_eb.py [--out DIR] [--dry-run] [--n-boot N]

Ten sam rachunek co `evaluate.py --compare` (`compare()`), ograniczony do warstwy
`all` i metryki RMSE: `metrics.bootstrap_paired_by_location` na tabelach
`SampleTable` z `outputs/ml/eval/{EA,EB}_seed{s}/samples_test@36.npz` (jawne
nazwy katalogow, bez globow). Ziarno bootstrapu `boot_seed=0` jak w `evaluate.py`.

DLACZEGO OSOBNO. Istniejace `compare_EA_seed{s}_vs_EB_seed{s}.json` maja ROZNE
n_boot (200 / 1000 / 1000), wiec ich przedzialy nie sa liczone jednolicie. Ten
skrypt liczy wszystkie trzy przy n_boot = 2000 do NOWYCH plikow; istniejacych
`compare_*.json` nie dotyka -- czyta je wylacznie do zestawienia w MD.

Konwencja: delta = EA − EB (doslownie X − Y; dodatnia = EA ma wiekszy blad).

Wyniki: `<DIR>/bootstrap_EA_vs_EB_seed{s}_nboot{N}.json` + `<DIR>/bootstrap_EA_vs_EB_summary.md`;
`--dry-run` -> `<DIR>/dryrun/`, n_boot = 50. Nie nadpisuje (blad, jesli plik istnieje).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402
from ..depth_model.metrics import SampleTable, bootstrap_paired_by_location  # noqa: E402

DEFAULT_OUT = paths.ML_OUTPUTS / "overnight_2026-11"
EVAL = paths.ML_OUTPUTS / "eval"
SEEDS = (0, 1, 2)
TEST_SET = "test@36"
STRATUM = "all"
METRIC = "RMSE"
BOOT_SEED = 0
N_BOOT = 2000
N_BOOT_DRY = 50
A, B = "EA", "EB"


def _tabela(cond: str, seed: int) -> SampleTable:
    p = EVAL / f"{cond}_seed{seed}" / f"samples_{TEST_SET}.npz"   # jawna nazwa, bez globow
    if not p.exists():
        raise FileNotFoundError(p)
    return SampleTable.load(p)


def _istniejacy(seed: int) -> dict | None:
    p = EVAL / f"compare_{A}_seed{seed}_vs_{B}_seed{seed}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    a = d.get(STRATUM, {})
    return {"plik": str(p.relative_to(paths.REPO_ROOT)), "n_boot": d.get("n_boot"),
            "test_set": d.get("test_set"), "seed": a.get("seed"),
            **{k: a.get(k) for k in ("delta_point", "ci_low", "ci_high", "boot_sd", "n_locations")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"katalog wynikow (domyslnie {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help=f"wyniki do <DIR>/dryrun/, n_boot={N_BOOT_DRY} (o ile nie podano --n-boot)")
    ap.add_argument("--n-boot", type=int, default=None,
                    help=f"liczba replik bootstrapu (domyslnie {N_BOOT}; przy --dry-run {N_BOOT_DRY})")
    args = ap.parse_args(argv)
    n_boot = args.n_boot if args.n_boot is not None else (N_BOOT_DRY if args.dry_run else N_BOOT)

    out_dir = args.out / "dryrun" if args.dry_run else args.out
    pliki = {s: out_dir / f"bootstrap_{A}_vs_{B}_seed{s}_nboot{n_boot}.json" for s in SEEDS}
    f_md = out_dir / f"bootstrap_{A}_vs_{B}_summary.md"
    istnieje = [str(p) for p in [*pliki.values(), f_md] if p.exists()]
    if istnieje:
        print("BLAD: plik(i) wynikowe juz istnieja, nie nadpisuje:\n  " + "\n  ".join(istnieje),
              file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    wyniki = {}
    for s in SEEDS:
        t0 = time.time()
        r = bootstrap_paired_by_location(_tabela(A, s), _tabela(B, s), stratum=STRATUM,
                                         metric=METRIC, n_boot=n_boot, seed=BOOT_SEED)
        rec = {"a": f"{A}_seed{s}", "b": f"{B}_seed{s}", "konwencja": f"delta = {A} − {B}",
               "test_set": TEST_SET, "n_boot": n_boot, "seed": BOOT_SEED,
               "delta_point": r["delta_point"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
               "n_locations": r["n_locations"], "boot_sd": r["boot_sd"],
               "ci_excludes_zero": r["ci_excludes_zero"], "n_samples": r["n_samples"],
               "alpha": r["alpha"], "stratum": STRATUM, "metric": METRIC,
               "dry_run": bool(args.dry_run), "czas_s": round(time.time() - t0, 1)}
        with open(pliki[s], "x", encoding="utf-8") as fh:     # "x" = blad, jesli plik istnieje
            fh.write(json.dumps(rec, indent=2, ensure_ascii=False))
        wyniki[s] = rec
        print(f"seed{s}: delta={rec['delta_point']:+.5f} CI95=[{rec['ci_low']:+.5f}, {rec['ci_high']:+.5f}] "
              f"sd={rec['boot_sd']:.5f} n_loc={rec['n_locations']} ({rec['czas_s']} s) -> {pliki[s].name}")

    L = [f"# Bootstrap po lokalizacjach: {A} vs {B}, {TEST_SET}", "",
         "Generowane przez `my-operations/ml/analysis/bootstrap_ea_eb.py`. Nie edytować ręcznie.",
         f"delta = {A} − {B} (dodatnia = {A} ma większy błąd), RMSE [m], warstwa `{STRATUM}`, "
         f"bootstrap sparowany po lokalizacjach, boot_seed = {BOOT_SEED}, α = 0,05.", ""]
    if args.dry_run:
        L += [f"**DRY-RUN** — n_boot = {n_boot}; przedziały NIE są wynikiem.", ""]
    L += [f"## Nowe przebiegi (n_boot = {n_boot})", "",
          "| ziarno | delta | CI 95 % | boot_sd | lokalizacji | plik |", "|---|---|---|---|---|---|"]
    for s, r in wyniki.items():
        L.append(f"| {s} | {r['delta_point']:+.5f} | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] | "
                 f"{r['boot_sd']:.5f} | {r['n_locations']} | `{pliki[s].name}` |")
    L += ["", "## Zestawienie z istniejącymi `compare_*.json` (tylko zestawienie)", "",
          "| ziarno | n_boot (istn.) | delta (istn.) | CI 95 % (istn.) | boot_sd (istn.) | "
          f"CI 95 % (n_boot = {n_boot}) | boot_sd (n_boot = {n_boot}) |", "|---|---|---|---|---|---|---|"]
    for s, r in wyniki.items():
        e = _istniejacy(s)
        if e is None:
            L.append(f"| {s} | brak pliku | — | — | — | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] | "
                     f"{r['boot_sd']:.5f} |")
            continue
        L.append(f"| {s} | {e['n_boot']} | {e['delta_point']:+.5f} | [{e['ci_low']:+.5f}, {e['ci_high']:+.5f}] | "
                 f"{e['boot_sd']:.5f} | [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] | {r['boot_sd']:.5f} |")
    L.append("")
    with open(f_md, "x", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"zapisano: {f_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
