# SUMMARY_B — część B (GPU): kroki 6 i 7

## Kroki

start: 2026-11-13 07:19:40  (limit nowych przebiegow: 10 h)
| krok | start | koniec | status | wyniki |
|---|---|---|---|---|
| 6_as1_train ED angle_seed=1: trening | 2026-11-13 07:19:40 | 2026-11-13 07:30:08 | OK | runs/ED_seed0_as1; log: stepB_6_as1_train.log |
| 6_as1_eval ED angle_seed=1: ewaluacja test@36 | 2026-11-13 07:30:08 | 2026-11-13 07:30:12 | OK | eval/ED_seed0_as1/eval.json; log: stepB_6_as1_eval.log |
| 6_as2_train ED angle_seed=2: trening | 2026-11-13 07:30:12 | 2026-11-13 07:40:41 | OK | runs/ED_seed0_as2; log: stepB_6_as2_train.log |
| 6_as2_eval ED angle_seed=2: ewaluacja test@36 | 2026-11-13 07:40:41 | 2026-11-13 07:40:45 | OK | eval/ED_seed0_as2/eval.json; log: stepB_6_as2_eval.log |
| 6_as3_train ED angle_seed=3: trening | 2026-11-13 07:40:45 | 2026-11-13 07:51:17 | OK | runs/ED_seed0_as3; log: stepB_6_as3_train.log |
| 6_as3_eval ED angle_seed=3: ewaluacja test@36 | 2026-11-13 07:51:17 | 2026-11-13 07:51:21 | OK | eval/ED_seed0_as3/eval.json; log: stepB_6_as3_eval.log |
| 6_summary krok 6: zestawienie | 2026-11-13 07:51:21 | 2026-11-13 07:51:21 | OK | step6_summary.md; log: stepB_6_summary.log |
| 7_A_seed0 RGB-only A ziarno 0 | 2026-11-13 07:51:21 | 2026-11-13 08:02:42 | OK | rgbonly/RGBONLY_A_seed0/; log: stepB_7_A_seed0.log |
| 7_D_seed0 RGB-only D ziarno 0 | 2026-11-13 08:02:42 | 2026-11-13 08:14:08 | OK | rgbonly/RGBONLY_D_seed0/; log: stepB_7_D_seed0.log |
| 7_B_seed0 RGB-only B ziarno 0 | 2026-11-13 08:14:08 | 2026-11-13 08:25:16 | OK | rgbonly/RGBONLY_B_seed0/; log: stepB_7_B_seed0.log |
| 7_A_seed1 RGB-only A ziarno 1 | 2026-11-13 08:25:16 | 2026-11-13 08:36:39 | OK | rgbonly/RGBONLY_A_seed1/; log: stepB_7_A_seed1.log |
| 7_D_seed1 RGB-only D ziarno 1 | 2026-11-13 08:36:39 | 2026-11-13 08:48:00 | OK | rgbonly/RGBONLY_D_seed1/; log: stepB_7_D_seed1.log |
| 7_B_seed1 RGB-only B ziarno 1 | 2026-11-13 08:48:01 | 2026-11-13 08:59:04 | OK | rgbonly/RGBONLY_B_seed1/; log: stepB_7_B_seed1.log |
| 7_A_seed2 RGB-only A ziarno 2 | 2026-11-13 08:59:04 | 2026-11-13 09:10:26 | OK | rgbonly/RGBONLY_A_seed2/; log: stepB_7_A_seed2.log |
| 7_D_seed2 RGB-only D ziarno 2 | 2026-11-13 09:10:26 | 2026-11-13 09:21:47 | OK | rgbonly/RGBONLY_D_seed2/; log: stepB_7_D_seed2.log |
| 7_B_seed2 RGB-only B ziarno 2 | 2026-11-13 09:21:47 | 2026-11-13 09:32:49 | OK | rgbonly/RGBONLY_B_seed2/; log: stepB_7_B_seed2.log |
| 7_summary krok 7: zestawienie | 2026-11-13 09:32:49 | 2026-11-13 09:32:50 | OK | step7_summary.md; log: stepB_7_summary.log |
koniec: 2026-11-13 09:32:50  (czas calkowity: 133 min)

## 6. ED z angle_seed 1–3

# Krok 6: ED z różnym `angle_seed` (training seed 0, 40 000 kroków, AMP)

| przebieg | angle_seed | RMSE test@36 |
|---|---|---|
| ED_seed0 (oryginał, sierpień) | 0 | 0.64695 |
| ED_seed0_nf0 (powtórzenie) | 0 | 0.65143 |
| ED_seed0_nf1 (powtórzenie) | 0 | 0.64986 |
| ED_seed0_nf2 (powtórzenie) | 0 | 0.65477 |
| ED_seed0_nf3 (powtórzenie) | 0 | 0.65331 |
| ED_seed0_nf4 (powtórzenie) | 0 | 0.65288 |
| ED_seed0_as1 | 1 | 0.64461 |
| ED_seed0_as2 | 2 | 0.66028 |
| ED_seed0_as3 | 3 | 0.64627 |

- powtórzenia `nf` (angle_seed 0): średnia 0.65245, sd 0.00188, rozstęp 0.00491.
- `as1..3` (n = 3): średnia 0.65039, sd 0.00861, rozstęp 0.01568; średnia `as` − średnia `nf` = -0.00207; oryginał − średnia `as` = -0.00344.
- czy wynik mieści się w zakresie powtórzeń `nf`: [('as1', False), ('as2', False), ('as3', False)]

Interpretacja (hipotezy, nie wnioski): jeśli `as1..3` mają rozrzut podobny do `nf`, to losowanie 4 orientacji na lokalizację (angle_seed) nie wnosi wykrywalnej dodatkowej zmienności ponad niedeterminizm GPU; jeśli rozstęp `as` jest wyraźnie większy — angle_seed jest osobnym źródłem rozrzutu.


## 7. RGB-only

# Krok 7: kontrola RGB-only (RGB2Depth od zera) wobec modelu pełnego

RGB-only = sama `RGBDepthNet`, te same pary (lokalizacja, kąt) co A/D/B, 40 000 kroków, batch 32, wybór checkpointu po walidacji na 36 orientacjach. Pełny model = A/D/B z `outputs/ml/eval`. RMSE głębi [m], 3 ziarna.

## test@36

| warunek | RGB-only | pełny model | pełny − RGB-only (wkład echa+materiału+uwagi, sparowany po ziarnie) |
|---|---|---|---|
| A | 0.36365 ± 0.00359 | 0.29248 ± 0.00488 | -0.07117 ± 0.00157 (p = 0.0002) |
| D | 0.34549 ± 0.00243 | 0.27199 ± 0.00265 | -0.07349 ± 0.00354 (p = 0.0008) |
| B | 0.28710 ± 0.00345 | 0.24367 ± 0.00142 | -0.04343 ± 0.00482 (p = 0.0041) |

| kontrast | RGB-only | pełny model |
|---|---|---|
| D − A | -0.01817 ± 0.00525 (p = 0.0267) | -0.02048 ± 0.00350 (p = 0.0096) |
| B − D | -0.05839 ± 0.00255 (p = 0.0006) | -0.02833 ± 0.00154 (p = 0.0010) |
| B − A | -0.07655 ± 0.00700 (p = 0.0028) | -0.04881 ± 0.00355 (p = 0.0018) |

## test@4

| warunek | RGB-only | pełny model | pełny − RGB-only (wkład echa+materiału+uwagi, sparowany po ziarnie) |
|---|---|---|---|
| A | 0.35857 ± 0.00291 | 0.28422 ± 0.00403 | -0.07436 ± 0.00383 (p = 0.0009) |
| D | 0.35387 ± 0.00278 | 0.27657 ± 0.00204 | -0.07730 ± 0.00079 (p = 0.0000) |
| B | 0.29620 ± 0.00334 | 0.24739 ± 0.00043 | -0.04881 ± 0.00376 (p = 0.0020) |

| kontrast | RGB-only | pełny model |
|---|---|---|
| D − A | -0.00471 ± 0.00454 (p = 0.2143) | -0.00765 ± 0.00262 (p = 0.0368) |
| B − D | -0.05767 ± 0.00570 (p = 0.0032) | -0.02917 ± 0.00172 (p = 0.0012) |
| B − A | -0.06237 ± 0.00533 (p = 0.0024) | -0.03682 ± 0.00361 (p = 0.0032) |

Odczyt: jeśli `D − A` występuje także w RGB-only, to część „efektu gęstości kątowej” w modelu pełnym pochodzi z różnorodności obrazów (orientacji kamery), a nie z echa; różnica kontrastów (pełny − RGB-only) szacuje część przypisywalną echu. To są hipotezy do rozstrzygnięcia na liczbach powyżej (3 ziarna, df = 2).


## B1. Ponowna ewaluacja 14 przebiegów

# Ponowna ewaluacja 14 przebiegów trenowanych 2026-08-11 (stan kodu sprzed commita e5eff1a1)

Metoda: `runs/<id>_reeval` z kopią `config.json` i dowiązaniami do `best_*.pth` oryginału (bez kopiowania wag), `evaluate.py --test-set test@36`, wynik w `eval/<id>_reeval/`; oryginały nietknięte. `ED_seed0/1/2` policzone wcześniej (ten sam sposób); tutaj dołączone z istniejących plików.

| przebieg | RMSE test@36 oryginał | RMSE test@36 re-ewaluacja | różnica | max \|Δ\| per scena | trening (start UTC) | eval oryginału | best_step | ESE: shuffled_echo (oryg./re) |
|---|---|---|---|---|---|---|---|---|
| SE_seed0 | 0.26433380 | 0.26433378 | -2.63e-08 | 9.7e-08 | 2026-08-11 19:05 | 08-11 23:21 | 38000 | False/False |
| B_seed0 | 0.24205105 | 0.24205106 | +7.14e-09 | 1.0e-07 | 2026-08-11 20:01 | 08-11 23:21 | 38000 | False/False |
| EA_seed0 | 0.77895299 | 0.77895297 | -2.12e-08 | 1.2e-07 | 2026-08-11 20:58 | 08-12 13:30 | 2000 | False/False |
| EA_seed1 | 0.79910485 | 0.79910505 | +1.96e-07 | 3.1e-07 | 2026-08-11 21:40 | 08-12 13:30 | 4000 | False/False |
| EA_seed2 | 0.79507569 | 0.79507568 | -1.19e-08 | 7.4e-08 | 2026-08-11 22:21 | 08-12 13:30 | 2000 | False/False |
| ED_seed0 | 0.64695049 | 0.64695050 | +1.48e-08 | 1.5e-07 | 2026-08-11 21:18 | 08-12 13:30 | 9000 | False/False |
| ED_seed1 | 0.64232286 | 0.64232275 | -1.13e-07 | 1.9e-07 | 2026-08-11 22:00 | 08-12 13:30 | 12000 | False/False |
| ED_seed2 | 0.64368941 | 0.64368939 | -2.66e-08 | 2.6e-07 | 2026-08-11 22:42 | 08-12 13:31 | 7000 | False/False |
| EB_seed0 | 0.58311155 | 0.58311158 | +3.55e-08 | 5.0e-08 | 2026-08-11 21:08 | 08-12 13:30 | 36000 | False/False |
| EB_seed1 | 0.58518575 | 0.58518582 | +7.24e-08 | 1.7e-07 | 2026-08-11 21:50 | 08-12 13:30 | 25000 | False/False |
| EB_seed2 | 0.57838601 | 0.57838577 | -2.41e-07 | 2.8e-07 | 2026-08-11 22:31 | 08-12 13:31 | 38000 | False/False |
| ESE_seed0 | 1.16467736 | 1.16467732 | -3.59e-08 | 7.8e-08 | 2026-08-11 21:29 | 08-12 13:30 | 6000 | False/False |
| ESE_seed1 | 1.16533418 | 1.16533403 | -1.51e-07 | 2.7e-07 | 2026-08-11 22:11 | 08-12 13:30 | 15000 | False/False |
| ESE_seed2 | 1.15875979 | 1.15875971 | -8.00e-08 | 1.2e-07 | 2026-08-11 22:52 | 08-12 13:31 | 5000 | False/False |

**Maksymalna różnica |Δ RMSE test@36| w 14 przebiegach: 2.41e-07** (oczekiwane ≈ 1e-7).

Wniosek: ewaluacja wag z 11.08 obecnym kodem i sterownikiem (580.178.04) odtwarza zapisane wyniki; rozbieżność ED_seed0 vs powtórzenia `nf` nie wynika z ewaluacji.


## Ścieżki

- `ED_seed0_vs_nf.md`
- `PLAN.md`
- `PROGRESS.md`
- `PROGRESS_B.md`
- `RESUME.md`
- `SUMMARY.md`
- `angle_diversity.json`
- `angle_diversity.md`
- `angle_diversity.py`
- `bootstrap_EA_vs_EB_seed0_nboot2000.json`
- `bootstrap_EA_vs_EB_seed1_nboot2000.json`
- `bootstrap_EA_vs_EB_seed2_nboot2000.json`
- `bootstrap_EA_vs_EB_summary.md`
- `branch_diag_A_seed0.json`
- `branch_diag_A_seed1.json`
- `branch_diag_A_seed2.json`
- `branch_diag_B_seed0.json`
- `branch_diag_B_seed1.json`
- `branch_diag_B_seed2.json`
- `branch_diag_D_seed0.json`
- `branch_diag_D_seed1.json`
- `branch_diag_D_seed2.json`
- `branch_diagnostics_summary.csv`
- `branch_diagnostics_summary.md`
- `depth_range_stats.json`
- `depth_range_stats.md`
- `ed_vs_nf_analysis.py`
- `holm_families.json`
- `holm_families.md`
- `make_summary.log`
- `make_summary.py`
- `make_summary_b.log`
- `make_summary_b.py`
- `overnight.log`
- `overnight.sh`
- `overnight_b.log`
- `overnight_b.sh`
- `reeval_14_runs.md`
- `step1.log`
- `step2.log`
- `step3.log`
- `step4.log`
- `step6_summary.md`
- `step6_summary.py`
- `step7_summary.md`
- `step7_summary.py`
- `stepB_6_as1_eval.log`
- `stepB_6_as1_train.log`
- `stepB_6_as2_eval.log`
- `stepB_6_as2_train.log`
- `stepB_6_as3_eval.log`
- `stepB_6_as3_train.log`
- `stepB_6_summary.log`
- `stepB_7_A_seed0.log`
- `stepB_7_A_seed1.log`
- `stepB_7_A_seed2.log`
- `stepB_7_B_seed0.log`
- `stepB_7_B_seed1.log`
- `stepB_7_B_seed2.log`
- `stepB_7_D_seed0.log`
- `stepB_7_D_seed1.log`
- `stepB_7_D_seed2.log`
- `stepB_7_summary.log`
- `rgbonly/RGBONLY_<A|D|B>_seed<N>/`
- `runs/ED_seed0_as{1,2,3}`, `eval/ED_seed0_as{1,2,3}`
