# PLAN — nocny zestaw sprawdzen metodologicznych (2026-11)

Przygotowano 2026-11-13. Uruchomienie: `outputs/ml/overnight_2026-11/overnight.sh` (nohup setsid; tmux nie jest zainstalowany).
Wyniki wylacznie w tym katalogu; podglad: `PROGRESS.md`, `step*.log`, na koncu `SUMMARY.md`.

## Zakres
W poleceniu zdefiniowano kroki **1-4**; tresc mowila o „krokach 1->7" i „skroc krok 7", ale kroków 5-7 nie bylo.
Wykonywane sa 1-4; `overnight.sh` ma wzorzec `run_step` do dopisania kolejnych.

| # | krok | skrypt (nowy) | CPU/GPU | wyniki | szacunek |
|---|---|---|---|---|---|
| 1 | Holm w rodzinach (a-e) + dodatkowo po wszystkich | `my-operations/ml/analysis/holm_families.py` | CPU | `holm_families.json/.md` | <1 min (dry-run 0,2 s) |
| 2 | odsetek pikseli > 10,29 m (test@36 ogolem/per scena, train B/A/ED; dodatkowo progi 8, 12 m) | `depth_range_stats.py` | CPU | `depth_range_stats.json/.md` | kilka min (odczyt HDF5 po scenach); limit 2 h |
| 3 | bootstrap EA vs EB, ziarna 0,1,2, n_boot=2000 | `bootstrap_ea_eb.py` | CPU | `bootstrap_EA_vs_EB_seed{s}_nboot2000.json`, `..._summary.md` | ~1-3 min |
| 4 | diagnostyka galezi A,D,B x 0,1,2 (alfa, alfa stale, material 0/shuffle) | `branch_diagnostics.py` | GPU | `branch_diag_*.json`, `branch_diagnostics_summary.md/.csv` | ~5 min (1 model: 30 s) |

Suma szacunkow: **< 30 min**, wyraznie ponizej 10 h (nie trzeba skracac zadnego kroku). Podstawa: czasy dry-run/pomiarow
skryptów; w nocy nie jest uruchamiany zaden trening (dla kontekstu: `runs/ED_seed0/status.json` = ~10,5 min treningu).
Globalny termin: po 10 h od startu nie zaczynaja sie nowe kroki (sprawdzane w `run_step`).

## Zmiany w istniejacych plikach
**Zadnych.** Wszystkie skrypty sa nowymi plikami (importuja z `ml.*`, `final_results.py`, `echo_only_tables.py`);
`beyond-image-to-depth/` nietkniety; zadnych operacji gita. (`git diff` ssledzonych plikow pokazuje tylko
`my-operations/ml/analysis/figures.py` — zmiana z wczesniejszego zadania o rysunkach v2, nie z tego.)
Diff: brak (nic do wypisania).

## Dry-run (wykonane przed startem)
- 1: 0,2 s; surowe p zgodne (D-A 0,00262, B-D 0,00163, B-A 0,00098; K=9 0,00316; model pelny 0,00958/0,00099/0,00175; EPB-EB 0,01263).
- 2: 1,5 s (2 lokalizacje/scena); indeksy pelne: test@36 = 6588, train B 49464, A 5496, ED 5496.
- 3: 0,9 s (n_boot=50); delta_point zgodna z istniejacymi compare_*.
- 4: `--check` A_seed0: RMSE depth_predicted 0,28739149 vs eval.json 0,28739153 (roznica -4,3e-8 < 1e-4); `--dry-run` 5,7 s.
Wyniki dry-run: `dryrun/`, `check/` (nie wchodza do wlasciwych wynikow).

## Uwagi do interpretacji kroku 4
- Przepisany forward wzgledem `model(dev)` rozni sie do 0,0039 m pod AMP (niedeterminizm cuDNN; `model(dev)` rozni sie od siebie tyle samo); bez AMP ~1e-6.
- Stala alfa = srednia alfa po pikselach waznych TEGO modelu na test@36 (nie z treningu).
- Przetasowanie cech materialu: jedna permutacja na batch, jeden generator (ziarno 12345) — wynik zalezy od `--batch-size` (32); ma punkty stale (206 z 6588 pozostaje na miejscu).
- Brak `statsmodels` — Holm wlasna implementacja (sprawdzona na przykladach).
