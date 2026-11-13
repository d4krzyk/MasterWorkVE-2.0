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
