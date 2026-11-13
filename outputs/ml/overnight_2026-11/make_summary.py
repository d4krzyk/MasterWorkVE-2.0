#!/usr/bin/env python
"""SUMMARY.md z wynikow nocnego zestawu: tabela krokow (z PROGRESS.md), najwazniejsze liczby
(wyciagane defensywnie z plikow wynikowych -- brak pliku nie przerywa), sciezki. Tylko odczyt."""
import json
import re
from pathlib import Path

D = Path(__file__).resolve().parent
out = ["# SUMMARY — nocny zestaw 2026-11", ""]


def read(name):
    p = D / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def section(title, body):
    out.extend([f"## {title}", "", body if body else "_brak pliku wynikowego (krok nie zakończył się sukcesem — patrz PROGRESS.md i step*.log)_", ""])


prog = read("PROGRESS.md")
tab = "\n".join(l for l in (prog or "").splitlines() if l.startswith("|") or l.startswith("start") or l.startswith("koniec"))
section("Kroki", tab)

# 1. Holm
try:
    j = json.loads(read("holm_families.json"))
    L = ["| rodzina | test | p surowe | p Holm | p_Holm < 0,05 |", "|---|---|---|---|---|"]
    for k, fam in j["rodziny"].items():
        for t in fam["testy"]:
            keys = list(t.keys())
            nm = t.get("test") or t.get("nazwa") or t.get("etykieta") or keys[0]
            pr = t.get("p_surowe", t.get("p_raw"))
            ph = t.get("p_holm", t.get("p_holm_rodzina"))
            L.append(f"| {k} | {nm} | {pr if pr is None else f'{pr:.4g}'} | {ph if ph is None else f'{ph:.4g}'} | "
                     f"{'tak' if (ph is not None and ph < 0.05) else 'nie'} |")
    section("1. Holm (w rodzinach)", "\n".join(L) + "\n\nPełne: `holm_families.md`.")
except Exception as e:  # noqa: BLE001
    section("1. Holm (w rodzinach)", read("holm_families.md") and "\n".join(read("holm_families.md").splitlines()[:80]) or None)

# 2-4: wklejka poczatku plikow md (bez dlugich tabel)
for title, f, n in (("2. Piksele > 10,29 m", "depth_range_stats.md", 70),
                    ("3. Bootstrap EA vs EB (n_boot = 2000)", "bootstrap_EA_vs_EB_summary.md", 40),
                    ("4. Diagnostyka gałęzi (A, D, B × ziarna 0–2)", "branch_diagnostics_summary.md", 90)):
    s = read(f)
    section(title, "\n".join(s.splitlines()[:n]) if s else None)

section("Ścieżki", "\n".join(f"- `{p.name}`" for p in sorted(D.iterdir()) if p.is_file()))
(D / "SUMMARY.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("zapisano", D / "SUMMARY.md")
