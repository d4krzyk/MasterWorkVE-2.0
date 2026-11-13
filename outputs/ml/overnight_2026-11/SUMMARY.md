# SUMMARY — nocny zestaw 2026-11

## Kroki

start: 2026-11-13 06:20:56  (limit nowych krokow: 10 h)
| krok | start | koniec | status | wyniki |
|---|---|---|---|---|
| 1 Holm w rodzinach testow | 2026-11-13 06:20:56 | 2026-11-13 06:20:57 | OK | holm_families.json/.md; log: step1.log |
| 2 piksele > 10,29 m (test@36, train) | 2026-11-13 06:20:57 | 2026-11-13 06:21:25 | OK | depth_range_stats.json/.md; log: step2.log |
| 3 bootstrap EA vs EB, n_boot=2000 | 2026-11-13 06:21:25 | 2026-11-13 06:21:27 | OK | bootstrap_EA_vs_EB_seed{0,1,2}_nboot2000.json, bootstrap_EA_vs_EB_summary.md; log: step3.log |
| 4 branch_diagnostics A,D,B x 0,1,2 | 2026-11-13 06:21:27 | 2026-11-13 06:25:50 | OK | branch_diag_*.json, branch_diagnostics_summary.md/.csv; log: step4.log |
koniec: 2026-11-13 06:25:50  (czas calkowity: 4 min)

## 1. Holm (w rodzinach)

| rodzina | test | p surowe | p Holm | p_Holm < 0,05 |
|---|---|---|---|---|
| a | D−A (ED−EA) | 0.002619 | 0.00327 | tak |
| a | B−D (EB−ED) | 0.001635 | 0.00327 | tak |
| a | B−A (EB−EA) | 0.0009785 | 0.002935 | tak |
| b | K=6 − K=36 (EK6−ED) | 6.043e-06 | 2.417e-05 | tak |
| b | K=9 − K=36 (EK9−ED) | 0.003155 | 0.009466 | tak |
| b | K=12 − K=36 (EK12−ED) | 0.06752 | 0.135 | nie |
| b | K=18 − K=36 (EK18−ED) | 0.6846 | 0.6846 | nie |
| c | D−A | 0.009581 | 0.009581 | tak |
| c | B−D | 0.0009881 | 0.002964 | tak |
| c | B−A | 0.001754 | 0.003508 | tak |
| d | EPA−EA | 0.8698 | 0.8698 | nie |
| d | EPD−ED | 0.0905 | 0.181 | nie |
| d | EPB−EB | 0.01263 | 0.03788 | tak |
| e | pretext_K4_seed0 − scratch | 0.231 | 0.6218 | nie |
| e | pretext_K36_p16_seed0 − scratch | 0.7506 | 0.7506 | nie |
| e | pretext_K36_seed0 − scratch | 0.2073 | 0.6218 | nie |
| e | pretext_K12_seed0 − scratch | 0.07421 | 0.2969 | nie |

Pełne: `holm_families.md`.

## 2. Piksele > 10,29 m

# Głębia poza zasięgiem okna echa

Generowane przez `my-operations/ml/analysis/depth_range_stats.py`. Nie edytować ręcznie.
Odsetek WAŻNYCH pikseli (depth > 0) z głębią > próg. Próg główny **10.29 m** = promien dwustronny okna echa 60 ms: 343 m/s * 0,060 s / 2; progi 8 i 12 m jako kontekst. Wariant `main`, split `e0bf7547668d9e0a`.

## test@36

Liczba próbek: 6588 (oczekiwane w pełnym przebiegu 6588; zgodne: True). Sceny: apartment_2, frl_apartment_5, office_4.

| zbiór | próbek | ważnych pikseli | > 8 m | > 10.29 m | > 12 m |
|---|---|---|---|---|---|
| test@36 ogółem | 6588 | 96853180 | 0.143 % (138293/96853180) | 0.014 % (13456/96853180) | 0.001 % (1255/96853180) |
| apartment_2 | 2556 | 37090393 | 0.031 % (11473/37090393) | 0.000 % (35/37090393) | 0.000 % (0/37090393) |
| frl_apartment_5 | 2664 | 37349731 | 0.340 % (126820/37349731) | 0.036 % (13421/37349731) | 0.003 % (1255/37349731) |
| office_4 | 1368 | 22413056 | 0.000 % (0/22413056) | 0.000 % (0/22413056) | 0.000 % (0/22413056) |

## Zbiór treningowy

A i ED to podzbiory wierszy B wybrane przez angles.select_angles (cardinal; random_4 z angle_seed=0, jak warunek ED/D).

| zbiór | próbek | ważnych pikseli | > 8 m | > 10.29 m | > 12 m |
|---|---|---|---|---|---|
| B (36 orientacji) | 49464 | 745360080 | 0.196 % (1458335/745360080) | 0.018 % (131684/745360080) | 0.001 % (7461/745360080) |
| A (cardinal) | 5496 | 82899002 | 0.212 % (175403/82899002) | 0.022 % (18467/82899002) | 0.001 % (985/82899002) |
| ED (random_4, angle_seed=0) | 5496 | 82795377 | 0.197 % (162875/82795377) | 0.016 % (13489/82795377) | 0.001 % (743/82795377) |

Per scena treningowa (B, 36 orientacji):

| zbiór | próbek | ważnych pikseli | > 8 m | > 10.29 m | > 12 m |
|---|---|---|---|---|---|
| apartment_0 | 7596 | 124452027 | 0.287 % (357182/124452027) | 0.024 % (29661/124452027) | 0.003 % (3248/124452027) |
| apartment_1 | 6336 | 97531413 | 0.004 % (3694/97531413) | 0.000 % (0/97531413) | 0.000 % (0/97531413) |
| frl_apartment_0 | 4680 | 65082396 | 0.282 % (183493/65082396) | 0.024 % (15468/65082396) | 0.000 % (202/65082396) |
| frl_apartment_1 | 4680 | 65044640 | 0.374 % (243265/65044640) | 0.039 % (25303/65044640) | 0.001 % (801/65044640) |
| frl_apartment_2 | 4500 | 63693817 | 0.344 % (219414/63693817) | 0.034 % (21433/63693817) | 0.002 % (1244/63693817) |
| frl_apartment_3 | 5292 | 73844432 | 0.313 % (231327/73844432) | 0.025 % (18501/73844432) | 0.001 % (879/73844432) |
| frl_apartment_4 | 4572 | 64004384 | 0.337 % (215936/64004384) | 0.033 % (21318/64004384) | 0.002 % (1087/64004384) |
| hotel_0 | 1728 | 28311128 | 0.002 % (700/28311128) | 0.000 % (0/28311128) | 0.000 % (0/28311128) |
| office_0 | 936 | 15329668 | 0.000 % (0/15329668) | 0.000 % (0/15329668) | 0.000 % (0/15329668) |
| office_1 | 576 | 9437045 | 0.000 % (0/9437045) | 0.000 % (0/9437045) | 0.000 % (0/9437045) |
| office_2 | 1332 | 20457227 | 0.000 % (0/20457227) | 0.000 % (0/20457227) | 0.000 % (0/20457227) |
| office_3 | 2232 | 36187946 | 0.009 % (3226/36187946) | 0.000 % (0/36187946) | 0.000 % (0/36187946) |
| room_0 | 2052 | 33619629 | 0.000 % (98/33619629) | 0.000 % (0/33619629) | 0.000 % (0/33619629) |
| room_1 | 1260 | 20643748 | 0.000 % (0/20643748) | 0.000 % (0/20643748) | 0.000 % (0/20643748) |
| room_2 | 1692 | 27720580 | 0.000 % (0/27720580) | 0.000 % (0/27720580) | 0.000 % (0/27720580) |

Pikseli > max_depth (14.104 m): test@36 0, train B 131.


## 3. Bootstrap EA vs EB (n_boot = 2000)

# Bootstrap po lokalizacjach: EA vs EB, test@36

Generowane przez `my-operations/ml/analysis/bootstrap_ea_eb.py`. Nie edytować ręcznie.
delta = EA − EB (dodatnia = EA ma większy błąd), RMSE [m], warstwa `all`, bootstrap sparowany po lokalizacjach, boot_seed = 0, α = 0,05.

## Nowe przebiegi (n_boot = 2000)

| ziarno | delta | CI 95 % | boot_sd | lokalizacji | plik |
|---|---|---|---|---|---|
| 0 | +0.19584 | [+0.16936, +0.22596] | 0.01436 | 183 | `bootstrap_EA_vs_EB_seed0_nboot2000.json` |
| 1 | +0.21392 | [+0.18549, +0.24552] | 0.01527 | 183 | `bootstrap_EA_vs_EB_seed1_nboot2000.json` |
| 2 | +0.21669 | [+0.19016, +0.24446] | 0.01364 | 183 | `bootstrap_EA_vs_EB_seed2_nboot2000.json` |

## Zestawienie z istniejącymi `compare_*.json` (tylko zestawienie)

| ziarno | n_boot (istn.) | delta (istn.) | CI 95 % (istn.) | boot_sd (istn.) | CI 95 % (n_boot = 2000) | boot_sd (n_boot = 2000) |
|---|---|---|---|---|---|---|
| 0 | 200 | +0.19584 | [+0.17170, +0.22283] | 0.01420 | [+0.16936, +0.22596] | 0.01436 |
| 1 | 1000 | +0.21392 | [+0.18604, +0.24450] | 0.01502 | [+0.18549, +0.24552] | 0.01527 |
| 2 | 1000 | +0.21669 | [+0.18985, +0.24497] | 0.01366 | [+0.19016, +0.24446] | 0.01364 |

## 4. Diagnostyka gałęzi (A, D, B × ziarna 0–2)

# Diagnostyka galezi modelu pelnego (AudioVisualModel)

Wygenerowano: 2026-11-13T05:25:49.878730+00:00 -- `my-operations/ml/analysis/branch_diagnostics.py`
Wersja ustawien: `branch_diag_v1 (2026-11-13)`. Zbior: test@36 (6588 probek), AMP=tak, batch 32.
Metryka: RMSE [m] zagregowane po pikselach, warstwa `all` (gt > 0), jak w `evaluate.py`. Srednia ± sd (ddof=1) po ziarnach [0, 1, 2].
Roznice: `X − Y` doslownie (docs/KONWENCJE.md §2); ujemne = mniejszy blad.
`alpha_const_mean`: alpha := srednia alpha z przebiegu bazowego TEGO modelu na zbiorze testowym test@36 (stala wyznaczona na danych oceny).
`material_shuffle`: permutacja w obrebie wsadu, ziarno 12345; moze miec punkty stale (liczba w plikach JSON per model).

## Warunek A

| wielkosc | srednia ± sd | seed0 | seed1 | seed2 |
|---|---|---|---|---|
| `RMSE(baseline)` | 0.292478 ± 0.004881 | 0.287392 | 0.297125 | 0.292916 |
| `RMSE(branch_audio_depth)` | 4.538590 ± 0.266524 | 4.277099 | 4.809876 | 4.528793 |
| `RMSE(branch_img_depth)` | 1.491638 ± 0.066600 | 1.438743 | 1.566430 | 1.469742 |
| `alpha_mean_valid_px(baseline)` | 0.232426 ± 0.007463 | 0.240387 | 0.225589 | 0.231302 |
| `alpha_mean_all_px(baseline)` | 0.231231 ± 0.007063 | 0.237849 | 0.223794 | 0.232048 |
| `RMSE(alpha_const_mean)` | 0.739955 ± 0.048873 | 0.709003 | 0.796298 | 0.714565 |
| `RMSE(alpha_const_0.5)` | 1.754154 ± 0.116436 | 1.642116 | 1.874539 | 1.745807 |
| `RMSE(material_zeros)` | 0.764898 ± 0.036027 | 0.777800 | 0.792697 | 0.724196 |
| `RMSE(material_shuffle)` | 0.822472 ± 0.063139 | 0.777047 | 0.894571 | 0.795799 |
| `alpha_mean_valid_px(material_zeros)` | 0.202076 ± 0.033399 | 0.171546 | 0.237748 | 0.196934 |
| `alpha_mean_all_px(material_zeros)` | 0.202021 ± 0.033510 | 0.171210 | 0.237699 | 0.197155 |
| `alpha_mean_valid_px(material_shuffle)` | 0.232634 ± 0.006632 | 0.239209 | 0.225948 | 0.232746 |
| `alpha_mean_all_px(material_shuffle)` | 0.231462 ± 0.006720 | 0.237573 | 0.224265 | 0.232549 |
| `RMSE(branch_audio_depth) − RMSE(baseline)` | 4.246112 ± 0.261672 | 3.989708 | 4.512752 | 4.235877 |
| `RMSE(branch_img_depth) − RMSE(baseline)` | 1.199161 ± 0.062068 | 1.151351 | 1.269305 | 1.176825 |
| `RMSE(alpha_const_mean) − RMSE(baseline)` | 0.447478 ± 0.044770 | 0.421612 | 0.499173 | 0.421648 |
| `RMSE(alpha_const_0.5) − RMSE(baseline)` | 1.461677 ± 0.111605 | 1.354724 | 1.577415 | 1.452891 |
| `RMSE(material_zeros) − RMSE(baseline)` | 0.472420 ± 0.035722 | 0.490409 | 0.495573 | 0.431279 |
| `RMSE(material_shuffle) − RMSE(baseline)` | 0.529995 ± 0.058788 | 0.489655 | 0.597446 | 0.502883 |
| `alpha_mean_valid_px(material_zeros) − alpha_mean_valid_px(baseline)` | -0.030350 ± 0.040649 | -0.068841 | 0.012159 | -0.034368 |
| `alpha_mean_valid_px(material_shuffle) − alpha_mean_valid_px(baseline)` | 0.000208 ± 0.001317 | -0.001178 | 0.000359 | 0.001444 |
| `RMSE(baseline) − RMSE(eval.json)` | 0.000000 ± 0.000000 | -0.000000 | 0.000000 | 0.000000 |

## Warunek D

| wielkosc | srednia ± sd | seed0 | seed1 | seed2 |
|---|---|---|---|---|
| `RMSE(baseline)` | 0.271993 ± 0.002654 | 0.269086 | 0.272605 | 0.274288 |
| `RMSE(branch_audio_depth)` | 4.293471 ± 0.323137 | 4.014549 | 4.647570 | 4.218295 |
| `RMSE(branch_img_depth)` | 1.407942 ± 0.047351 | 1.366707 | 1.459654 | 1.397466 |
| `alpha_mean_valid_px(baseline)` | 0.238255 ± 0.008816 | 0.246326 | 0.228847 | 0.239593 |
| `alpha_mean_all_px(baseline)` | 0.237152 ± 0.007301 | 0.243720 | 0.229291 | 0.238445 |
| `RMSE(alpha_const_mean)` | 0.678740 ± 0.032353 | 0.652272 | 0.714806 | 0.669143 |
| `RMSE(alpha_const_0.5)` | 1.647330 ± 0.143721 | 1.524263 | 1.805282 | 1.612445 |
| `RMSE(material_zeros)` | 0.750902 ± 0.095962 | 0.857901 | 0.722348 | 0.672459 |
| `RMSE(material_shuffle)` | 0.758066 ± 0.041606 | 0.726222 | 0.805142 | 0.742834 |
| `alpha_mean_valid_px(material_zeros)` | 0.174275 ± 0.042526 | 0.125793 | 0.191766 | 0.205267 |
| `alpha_mean_all_px(material_zeros)` | 0.173836 ± 0.042710 | 0.125190 | 0.191133 | 0.205184 |
| `alpha_mean_valid_px(material_shuffle)` | 0.239728 ± 0.007410 | 0.246019 | 0.231559 | 0.241605 |
| `alpha_mean_all_px(material_shuffle)` | 0.238382 ± 0.007039 | 0.244159 | 0.230542 | 0.240444 |
| `RMSE(branch_audio_depth) − RMSE(baseline)` | 4.021478 ± 0.321822 | 3.745463 | 4.374965 | 3.944007 |
| `RMSE(branch_img_depth) − RMSE(baseline)` | 1.135950 ± 0.046061 | 1.097622 | 1.187049 | 1.123178 |
| `RMSE(alpha_const_mean) − RMSE(baseline)` | 0.406748 ± 0.031253 | 0.383187 | 0.442201 | 0.394855 |
| `RMSE(alpha_const_0.5) − RMSE(baseline)` | 1.375338 ± 0.142437 | 1.255177 | 1.532678 | 1.338158 |
| `RMSE(material_zeros) − RMSE(baseline)` | 0.478910 ± 0.098612 | 0.588815 | 0.449743 | 0.398171 |
| `RMSE(material_shuffle) − RMSE(baseline)` | 0.486074 ± 0.040641 | 0.457137 | 0.532537 | 0.468547 |
| `alpha_mean_valid_px(material_zeros) − alpha_mean_valid_px(baseline)` | -0.063980 ± 0.048996 | -0.120533 | -0.037081 | -0.034326 |
| `alpha_mean_valid_px(material_shuffle) − alpha_mean_valid_px(baseline)` | 0.001472 ± 0.001581 | -0.000308 | 0.002712 | 0.002012 |
| `RMSE(baseline) − RMSE(eval.json)` | -0.000000 ± 0.000000 | -0.000000 | -0.000000 | -0.000000 |

## Warunek B

| wielkosc | srednia ± sd | seed0 | seed1 | seed2 |
|---|---|---|---|---|
| `RMSE(baseline)` | 0.243667 ± 0.001417 | 0.242051 | 0.244698 | 0.244253 |
| `RMSE(branch_audio_depth)` | 4.624687 ± 0.285597 | 4.339047 | 4.910241 | 4.624771 |
| `RMSE(branch_img_depth)` | 1.401892 ± 0.027831 | 1.376272 | 1.431503 | 1.397899 |
| `alpha_mean_valid_px(baseline)` | 0.223046 ± 0.009263 | 0.232276 | 0.213752 | 0.223110 |
| `alpha_mean_all_px(baseline)` | 0.223303 ± 0.009286 | 0.232479 | 0.213911 | 0.223519 |
| `RMSE(alpha_const_mean)` | 0.663670 ± 0.017080 | 0.649542 | 0.682652 | 0.658816 |
| `RMSE(alpha_const_0.5)` | 1.796694 ± 0.129006 | 1.668131 | 1.926139 | 1.795812 |
| `RMSE(material_zeros)` | 0.703320 ± 0.025930 | 0.731188 | 0.679903 | 0.698869 |
| `RMSE(material_shuffle)` | 0.733399 ± 0.026302 | 0.718888 | 0.763760 | 0.717548 |
| `alpha_mean_valid_px(material_zeros)` | 0.176404 ± 0.021635 | 0.157827 | 0.200157 | 0.171227 |
| `alpha_mean_all_px(material_zeros)` | 0.176113 ± 0.021231 | 0.158121 | 0.199531 | 0.170688 |
| `alpha_mean_valid_px(material_shuffle)` | 0.225999 ± 0.009611 | 0.235456 | 0.216240 | 0.226302 |
| `alpha_mean_all_px(material_shuffle)` | 0.225453 ± 0.009602 | 0.234761 | 0.215582 | 0.226015 |
| `RMSE(branch_audio_depth) − RMSE(baseline)` | 4.381019 ± 0.284274 | 4.096996 | 4.665543 | 4.380519 |
| `RMSE(branch_img_depth) − RMSE(baseline)` | 1.158224 ± 0.026589 | 1.134221 | 1.186805 | 1.153647 |
| `RMSE(alpha_const_mean) − RMSE(baseline)` | 0.420003 ± 0.015943 | 0.407491 | 0.437954 | 0.414563 |
| `RMSE(alpha_const_0.5) − RMSE(baseline)` | 1.553027 ± 0.127687 | 1.426080 | 1.681442 | 1.551559 |
| `RMSE(material_zeros) − RMSE(baseline)` | 0.459653 ± 0.027316 | 0.489136 | 0.435205 | 0.454616 |
| `RMSE(material_shuffle) − RMSE(baseline)` | 0.489731 ± 0.025462 | 0.476837 | 0.519062 | 0.473296 |
| `alpha_mean_valid_px(material_zeros) − alpha_mean_valid_px(baseline)` | -0.046642 ± 0.030764 | -0.074449 | -0.013594 | -0.051883 |
| `alpha_mean_valid_px(material_shuffle) − alpha_mean_valid_px(baseline)` | 0.002953 ± 0.000402 | 0.003179 | 0.002489 | 0.003191 |
| `RMSE(baseline) − RMSE(eval.json)` | -0.000000 ± 0.000000 | 0.000000 | -0.000000 | -0.000000 |

## Ścieżki

- `PLAN.md`
- `PROGRESS.md`
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
- `holm_families.json`
- `holm_families.md`
- `make_summary.log`
- `make_summary.py`
- `overnight.log`
- `overnight.sh`
- `step1.log`
- `step2.log`
- `step3.log`
- `step4.log`

