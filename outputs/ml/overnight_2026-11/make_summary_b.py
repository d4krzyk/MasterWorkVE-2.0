#!/usr/bin/env python
"""SUMMARY_B.md: tabela krokow B (z PROGRESS_B.md) + wklejki step6/step7 + sciezki. Nie nadpisuje SUMMARY.md (starszy zestaw)."""
from pathlib import Path
D = Path(__file__).resolve().parent
def rd(n): p = D / n; return p.read_text(encoding="utf-8") if p.exists() else None
o = ["# SUMMARY_B — część B (GPU): kroki 6 i 7", ""]
pr = rd("PROGRESS_B.md"); o += ["## Kroki", "", "\n".join(l for l in (pr or "brak PROGRESS_B.md").splitlines() if l.startswith(("|", "start", "koniec"))), ""]
for t, f in (("6. ED z angle_seed 1–3", "step6_summary.md"), ("7. RGB-only", "step7_summary.md"), ("B1. Ponowna ewaluacja 14 przebiegów", "reeval_14_runs.md")):
    s = rd(f); o += [f"## {t}", "", s if s else "_brak pliku (krok nie zakończył się sukcesem — patrz PROGRESS_B.md i stepB*.log)_", ""]
o += ["## Ścieżki", ""] + [f"- `{p.name}`" for p in sorted(D.iterdir()) if p.is_file()] + ["- `rgbonly/RGBONLY_<A|D|B>_seed<N>/`", "- `runs/ED_seed0_as{1,2,3}`, `eval/ED_seed0_as{1,2,3}`"]
(D / "SUMMARY_B.md").write_text("\n".join(o) + "\n", encoding="utf-8"); print("zapisano SUMMARY_B.md")
