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

## Kontynuacja (część B/A) — 2026-11-13

### Decyzja do kroku 7 (RGB-only)

- `scratch` w `pretext_model/transfer.py`: 36 orientacji (`angle_subset=all`), n_train 49 464, 40 000 kroków, batch 32, lr 1e-4, wd 5e-4, AMP, walidacja na 36 (n_val 6 588). **Ale** jego `best.json` to RMSE walidacyjne (checkpoint wybrany na tym samym zbiorze), bez ewaluacji test@36 i test@4 — warunek reguły ("walidacja na 36 i test@36") nie jest spełniony.
- **Decyzja: trenowane są A, D i B** (RGB-only, `ml/depth_model/train_rgbonly.py`), 9 przebiegów, ziarno na zewnątrz, warunek wewnątrz; pary (lokalizacja, kąt) zweryfikowane jako identyczne z A/D/B (`rgbonly/check_pairs.json`: A 5496, D 5496, B 49464; walidacja 6588, test@36 6588, test@4 732).
- Szacunek czasu: dry-run 0,0155 s/krok (A) i 0,0130 (B) → trening ok. 9–10 min + 40 walidacji ok. 5 min ≈ **15 min na przebieg**, razem 9 × 15 ≈ **2,3 h**; krok 6: 3 × ok. 11–12 min ≈ **35 min**. Suma ≈ **3 h** (< 10 h).
- Uwaga o `transfer_scratch`: jego RMSE (np. 0,28986) jest walidacyjne, więc NIE jest porównywalne z test@36 pełnych modeli (A/D/B) bez tej adnotacji.
- Znalezisko agenta: `train_condition._plateau_diagnostics` czyta `r["overall"]["RMSE"]` zamiast `r["all"]`, więc `status.json` ma zawsze `n_validations: 0`/`budget_ceiling_warning: false` we wszystkich przebiegach sieci głębi (nie poprawiane; w `train_rgbonly.py` diagnostyka plateau pominięta).

### Diff istniejących plików (jedyny: nowa opcja z domyślnym zachowaniem bez zmian)

`my-operations/ml/depth_model/train_condition.py` (`--angle-seed`, domyślnie None = angle_seed z definicji warunku):

```diff
@@ -301,18 +301,23 @@
     p.add_argument("--tag", default="",
                    help="dopisek do nazwy przebiegu (<warunek>_seed<N>_<tag>); domyslnie pusty, "
                         "wtedy katalog i zachowanie jak dotad")
+    p.add_argument("--angle-seed", type=int, default=None,
+                   help="nadpisuje angle_seed warunku (losowanie katow random_K w ZBIORZE TRENINGOWYM); "
+                        "domyslnie None = wartosc z definicji warunku (dla ED/D: 0), zachowanie jak dotad")
     p.add_argument("--force", action="store_true", help="nadpisz istniejacy katalog przebiegu")
     p.add_argument("--dry-run", action="store_true",
                    help="pokaz konfiguracje i licznosci, NIE trenuj (nie dotyka GPU)")
     args = p.parse_args(argv)
 
     cond = CONDITIONS_BY_ID[args.condition]
+    angle_seed = cond.angle_seed if args.angle_seed is None else args.angle_seed
     spec = RunSpec(
         condition=cond.id, seed=args.seed, total_steps=args.steps,
         batch_size=args.batch_size, num_workers=args.num_workers,
         amp=not args.no_amp, validation_freq=args.validation_freq,
         display_freq=args.display_freq, edge_threshold_m=args.edge_threshold,
-        extra={"fast_bilinear": args.fast_bilinear}, tag=args.tag,
+        extra={"fast_bilinear": args.fast_bilinear,
+               **({"angle_seed": angle_seed} if args.angle_seed is not None else {})}, tag=args.tag,
     )
     run_dir = spec.run_dir()
     splits = load_splits(variant=cond.geometry)
@@ -329,6 +334,8 @@
     print(f"  krokow         : {spec.total_steps} (batch {spec.batch_size})")
     print(f"  rownowaznik ep.: {cond.epochs_equivalent(splits, spec.total_steps, spec.batch_size):.1f}")
     print(f"  AMP            : {spec.amp}")
+    if args.angle_seed is not None:
+        print(f"  angle_seed     : {angle_seed}  (nadpisany; definicja warunku: {cond.angle_seed})")
     print(f"  katalog        : {run_dir}")
     print(f"  odcisk podzialu: {splits.meta.get('location_fingerprint')}")
 
@@ -359,7 +366,7 @@
     shuffle_seed = getattr(cond, "shuffle_echo_seed", None)
     train_loader, train_ds = build_dataloader(
         DatasetConfig(variant=cond.geometry, mode="train",
-                      angle_subset=cond.angle_subset, angle_seed=cond.angle_seed,
+                      angle_subset=cond.angle_subset, angle_seed=angle_seed,
                       shuffle_echo_seed=shuffle_seed),
         batch_size=spec.batch_size, num_workers=spec.num_workers, splits=splits,
     )
```

Nowe pliki: `my-operations/ml/depth_model/train_rgbonly.py`, w tym katalogu `overnight_b.sh`, `step6_summary.py`, `step7_summary.py`, `make_summary_b.py`, `angle_diversity.py`, `RESUME.md`.
