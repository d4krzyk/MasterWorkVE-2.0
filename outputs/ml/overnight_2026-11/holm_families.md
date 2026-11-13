# Poprawka Holma w rodzinach testów

Generowane przez `my-operations/ml/analysis/holm_families.py`. Nie edytować ręcznie.
Konwencja: `X − Y` = dosłownie X − Y (ujemne = mniejszy błąd = lepiej). RMSE głębi [m]. Holm–Bonferroni step-down, α = 0,05.

## Holm w obrębie rodzin (wynik podstawowy)

### (a) siec tylko-echo (echo2depth): kontrasty D−A, B−D, B−A, test@36, 3 ziarna

Liczba testów w rodzinie: **3**.

| test | Δ / średnia | p surowe | p Holm (rodzina) | p_Holm < 0,05 | n | metoda |
|---|---|---|---|---|---|---|
| D−A (ED−EA) | -0.14672 | 0.0026 | 0.0033 | tak | 3 vs 3 | ttest_1samp na roznicach per ziarno (df=2) |
| B−D (EB−ED) | -0.06209 | 0.0016 | 0.0033 | tak | 3 vs 3 | ttest_1samp na roznicach per ziarno (df=2) |
| B−A (EB−EA) | -0.20882 | 0.0010 | 0.0029 | tak | 3 vs 3 | ttest_1samp na roznicach per ziarno (df=2) |

### (b) krzywa stalego budzetu (echo2depth): K − K36, Welch 3 vs 3

Liczba testów w rodzinie: **4**.

| test | Δ / średnia | p surowe | p Holm (rodzina) | p_Holm < 0,05 | n | metoda |
|---|---|---|---|---|---|---|
| K=6 − K=36 (EK6−ED) | +0.06191 | 6.0e-06 | 2.4e-05 | tak | 3 vs 3 | Welch 3 vs 3 (df=3.83) |
| K=9 − K=36 (EK9−ED) | +0.01910 | 0.0032 | 0.0095 | tak | 3 vs 3 | Welch 3 vs 3 (df=3.40) |
| K=12 − K=36 (EK12−ED) | +0.00899 | 0.0675 | 0.1350 | nie | 3 vs 3 | Welch 3 vs 3 (df=2.89) |
| K=18 − K=36 (EK18−ED) | +0.00372 | 0.6846 | 0.6846 | nie | 3 vs 3 | Welch 3 vs 3 (df=2.12) |

### (c) model pelny (RGB+echo+material+uwaga): D−A, B−D, B−A, 3 ziarna

Liczba testów w rodzinie: **3**.

| test | Δ / średnia | p surowe | p Holm (rodzina) | p_Holm < 0,05 | n | metoda |
|---|---|---|---|---|---|---|
| D−A | -0.02048 | 0.0096 | 0.0096 | tak | 3 vs 3 | ttest_1samp na roznicach per ziarno (df=n−1) |
| B−D | -0.02833 | 0.0010 | 0.0030 | tak | 3 vs 3 | ttest_1samp na roznicach per ziarno (df=n−1) |
| B−A | -0.04881 | 0.0018 | 0.0035 | tak | 3 vs 3 | ttest_1samp na roznicach per ziarno (df=n−1) |

### (d) geometria: patched − main, Welch, maska pelna, 3 ziarna

Liczba testów w rodzinie: **3**.

| test | Δ / średnia | p surowe | p Holm (rodzina) | p_Holm < 0,05 | n | metoda |
|---|---|---|---|---|---|---|
| EPA−EA | -0.00123 | 0.8698 | 0.8698 | nie | 3 vs 3 | Welch na RMSE test@36, maska pelna (df=2.92) |
| EPD−ED | +0.01045 | 0.0905 | 0.1810 | nie | 3 vs 3 | Welch na RMSE test@36, maska pelna (df=2.55) |
| EPB−EB | +0.01235 | 0.0126 | 0.0379 | tak | 3 vs 3 | Welch na RMSE test@36, maska pelna (df=2.93) |

### (e) transfer pretrening vs scratch, 5 ziaren, RMSE best.json

Liczba testów w rodzinie: **4**.

| test | Δ / średnia | p surowe | p Holm (rodzina) | p_Holm < 0,05 | n | metoda |
|---|---|---|---|---|---|---|
| pretext_K4_seed0 − scratch | -0.00287 | 0.2310 | 0.6218 | nie | 5 vs 5 | Welch na RMSE best.json['all'] (pelna walidacja) (df=5.70) |
| pretext_K36_p16_seed0 − scratch | -0.00059 | 0.7506 | 0.7506 | nie | 5 vs 5 | Welch na RMSE best.json['all'] (pelna walidacja) (df=6.56) |
| pretext_K36_seed0 − scratch | +0.00453 | 0.2073 | 0.6218 | nie | 5 vs 5 | Welch na RMSE best.json['all'] (pelna walidacja) (df=4.75) |
| pretext_K12_seed0 − scratch | +0.00702 | 0.0742 | 0.2969 | nie | 5 vs 5 | Welch na RMSE best.json['all'] (pelna walidacja) (df=4.77) |

Status przebiegów transferu (`status.json` finished i obecny `best.json`):

- scratch: seed0=tak, seed1=tak, seed2=tak, seed3=tak, seed4=tak
- pretext_K4_seed0: seed0=tak, seed1=tak, seed2=tak, seed3=tak, seed4=tak
- pretext_K36_p16_seed0: seed0=tak, seed1=tak, seed2=tak, seed3=tak, seed4=tak
- pretext_K36_seed0: seed0=tak, seed1=tak, seed2=tak, seed3=tak, seed4=tak
- pretext_K12_seed0: seed0=tak, seed1=tak, seed2=tak, seed3=tak, seed4=tak

## DODATKOWO: Holm po wszystkich testach razem

Kontrola konserwatywna (m = 17); nie zastępuje korekty w rodzinach.

| rodzina | test | p surowe | p Holm (wszystkie) | p_Holm < 0,05 |
|---|---|---|---|---|
| a | D−A (ED−EA) | 0.0026 | 0.0314 | tak |
| a | B−D (EB−ED) | 0.0016 | 0.0229 | tak |
| a | B−A (EB−EA) | 0.0010 | 0.0157 | tak |
| b | K=6 − K=36 (EK6−ED) | 6.0e-06 | 0.0001 | tak |
| b | K=9 − K=36 (EK9−ED) | 0.0032 | 0.0347 | tak |
| b | K=12 − K=36 (EK12−ED) | 0.0675 | 0.5402 | nie |
| b | K=18 − K=36 (EK18−ED) | 0.6846 | 1.0000 | nie |
| c | D−A | 0.0096 | 0.0958 | nie |
| c | B−D | 0.0010 | 0.0157 | tak |
| c | B−A | 0.0018 | 0.0229 | tak |
| d | EPA−EA | 0.8698 | 1.0000 | nie |
| d | EPD−ED | 0.0905 | 0.5430 | nie |
| d | EPB−EB | 0.0126 | 0.1136 | nie |
| e | pretext_K4_seed0 − scratch | 0.2310 | 1.0000 | nie |
| e | pretext_K36_p16_seed0 − scratch | 0.7506 | 1.0000 | nie |
| e | pretext_K36_seed0 − scratch | 0.2073 | 1.0000 | nie |
| e | pretext_K12_seed0 − scratch | 0.0742 | 0.5402 | nie |

Implementacja Holma: statsmodels niedostepny -- tylko wlasna implementacja.

## Kontrola p surowych wobec dokumentacji

| rodzina | test | oczekiwane | obliczone | |
|---|---|---|---|---|
| a | D−A (ED−EA) | 0.0026 | 0.00262 | zgodne |
| a | B−D (EB−ED) | 0.0016 | 0.00163 | zgodne |
| a | B−A (EB−EA) | 0.001 | 0.00098 | zgodne |
| b | K=9 − K=36 (EK9−ED) | 0.0032 | 0.00316 | zgodne |
| c | D−A | 0.0096 | 0.00958 | zgodne |
| c | B−D | 0.001 | 0.00099 | zgodne |
| c | B−A | 0.0018 | 0.00175 | zgodne |
| d | EPB−EB | 0.013 | 0.01263 | zgodne |
