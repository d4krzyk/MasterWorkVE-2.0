# Tabele §5.3–5.5 — sieć głębi (tylko echo), test@36

Generowane przez `my-operations/ml/analysis/echo_only_tables.py` z `outputs/ml/eval/<warunek>_seed{0,1,2}/eval.json`. Nie edytować ręcznie.
Konwencja: `X − Y` = dosłownie X − Y (ujemne = mniejszy błąd = lepiej). RMSE głębi [m], średnia ± sd po 3 ziarnach. t i p: `ttest_1samp` na różnicach per ziarno (df = 2), o ile nie zaznaczono Welcha.

## 5.3 Warunki i kontrasty

| warunek | RMSE | per ziarno |
|---|---|---|
| EA | 0.79104 ± 0.01066 | 0.77895 · 0.79910 · 0.79508 |
| ED | 0.64432 ± 0.00238 | 0.64695 · 0.64232 · 0.64369 |
| EB | 0.58223 ± 0.00348 | 0.58311 · 0.58519 · 0.57839 |
| ESE | 1.16292 ± 0.00362 | 1.16468 · 1.16533 · 1.15876 |

| kontrast | średnia ± sd | per ziarno | t | p |
|---|---|---|---|---|
| D−A | -0.14672 ± 0.01303 | -0.13200 · -0.15678 · -0.15139 | -19.50 | 0.0026 |
| B−D | -0.06209 ± 0.00435 | -0.06384 · -0.05714 · -0.06530 | -24.70 | 0.0016 |
| B−A | -0.20882 ± 0.01132 | -0.19584 · -0.21392 · -0.21669 | -31.94 | 0.0010 |
| ESE−EB | +0.58070 ± 0.00076 | +0.58157 · +0.58015 · +0.58037 | +1320.55 | 5.7e-07 |

**Udział gęstości (D−A)/(B−A):** per ziarno 67.4 · 73.3 · 69.9 %; średnia ± sd = **70.2 ± 3.0 %**; z średnich 70.3 %.

**Spadek względny** (z średnich; w nawiasie per ziarno):

- D−A: -18.55 % (-16.95 · -19.62 · -19.04)
- B−D: -9.64 % (-9.87 · -8.90 · -10.15)
- B−A: -26.40 % (-25.14 · -26.77 · -27.25)

**D−A per scena testowa** (ED − EA):

| scena | EA | ED | D−A | per ziarno | t | p |
|---|---|---|---|---|---|---|
| apartment_2 | 0.78951 ± 0.00998 | 0.72619 ± 0.00687 | -0.06332 ± 0.01671 | -0.04427 · -0.07022 · -0.07547 | -6.57 | 0.0224 |
| frl_apartment_5 | 0.89923 ± 0.01289 | 0.63658 ± 0.00488 | -0.26265 ± 0.01557 | -0.24680 · -0.27792 · -0.26324 | -29.22 | 0.0012 |
| office_4 | 0.57003 ± 0.01357 | 0.49586 ± 0.00887 | -0.07417 ± 0.00517 | -0.07275 · -0.07990 · -0.06986 | -24.86 | 0.0016 |

## 5.4 Krzywa stałego budżetu

Rozstęp podłogi szumu GPU: **0.00491** (`outputs/ml/determinism/noise_floor_ED_40k.json`). Zysk 4→36: 0.14672. K−K36: Welch 3 vs 3.

| K | warunek | RMSE | K − K36 | t | df | p | udział zysku 4→36 | krok od poprzedniego K | krotność rozstępu |
|---|---|---|---|---|---|---|---|---|---|
| 4 | EA | 0.79104 ± 0.01066 | +0.14672 | +23.26 | 2.20 | 0.0011 | 0.0 % | — | — |
| 6 | EK6 | 0.70623 ± 0.00192 | +0.06191 | +35.12 | 3.83 | 6.0e-06 | 57.8 % | -0.08482 (od K=4) | 17.3× |
| 9 | EK9 | 0.66342 ± 0.00372 | +0.01910 | +7.49 | 3.40 | 0.0032 | 87.0 % | -0.04281 (od K=6) | 8.7× |
| 12 | EK12 | 0.65331 ± 0.00490 | +0.00899 | +2.86 | 2.89 | 0.0675 | 93.9 % | -0.01010 (od K=9) | 2.1× |
| 18 | EK18 | 0.64804 ± 0.01362 | +0.00372 | +0.47 | 2.12 | 0.6846 | 97.5 % | -0.00527 (od K=12) | 1.1× |
| 36 | ED | 0.64432 ± 0.00238 | — | — | — | — | 100.0 % | -0.00372 (od K=18) | 0.8× |

## 5.5 RMSE wg odległości od siatki treningowej

odleglosc = od najblizszego kata SIATKI treningowej K (nie od faktycznie wylosowanych 4 katow lokalizacji), pole by_angular_distance z eval.json.

**EA**

| odległość | próbek | RMSE | kara względem 0° | t | p |
|---|---|---|---|---|---|
| 0° | 732 | 0.59626 ± 0.00822 | — | — | — |
| 10° | 1464 | 0.66296 ± 0.00227 | +0.06670 ± 0.00662 | +17.45 | 0.0033 |
| 20° | 1464 | 0.78249 ± 0.00870 | +0.18623 ± 0.01655 | +19.49 | 0.0026 |
| 30° | 1464 | 0.87034 ± 0.01575 | +0.27408 ± 0.02267 | +20.94 | 0.0023 |
| 40° | 1464 | 0.91103 ± 0.02190 | +0.31477 ± 0.02860 | +19.06 | 0.0027 |

Luka 0°→40°: **+0.31477 ± 0.02860** (per ziarno +0.28434 · +0.34111 · +0.31886); monotoniczna ściśle w każdym ziarnie: **True** [True, True, True].

**EK6**

| odległość | próbek | RMSE | kara względem 0° | t | p |
|---|---|---|---|---|---|
| 0° | 1098 | 0.60357 ± 0.01369 | — | — | — |
| 10° | 2196 | 0.66226 ± 0.00520 | +0.05869 ± 0.01023 | +9.94 | 0.0100 |
| 20° | 2196 | 0.75190 ± 0.00497 | +0.14833 ± 0.01810 | +14.19 | 0.0049 |
| 30° | 1098 | 0.78758 ± 0.00702 | +0.18401 ± 0.02069 | +15.40 | 0.0042 |

Luka 0°→30°: **+0.18401 ± 0.02069** (per ziarno +0.17374 · +0.17045 · +0.20783); monotoniczna ściśle w każdym ziarnie: **True** [True, True, True].

**EK9**

| odległość | próbek | RMSE | kara względem 0° | t | p |
|---|---|---|---|---|---|
| 0° | 1647 | 0.62252 ± 0.01242 | — | — | — |
| 10° | 3294 | 0.66340 ± 0.00352 | +0.04088 ± 0.01303 | +5.43 | 0.0322 |
| 20° | 1647 | 0.70183 ± 0.01092 | +0.07931 ± 0.02186 | +6.28 | 0.0244 |

Luka 0°→20°: **+0.07931 ± 0.02186** (per ziarno +0.05412 · +0.09030 · +0.09349); monotoniczna ściśle w każdym ziarnie: **True** [True, True, True].

**EK12**

| odległość | próbek | RMSE | kara względem 0° | t | p |
|---|---|---|---|---|---|
| 0° | 2196 | 0.62907 ± 0.00348 | — | — | — |
| 10° | 4392 | 0.66511 ± 0.00558 | +0.03604 ± 0.00214 | +29.19 | 0.0012 |

Luka 0°→10°: **+0.03604 ± 0.00214** (per ziarno +0.03632 · +0.03802 · +0.03377); monotoniczna ściśle w każdym ziarnie: **True** [True, True, True].

**EK18**

| odległość | próbek | RMSE | kara względem 0° | t | p |
|---|---|---|---|---|---|
| 0° | 3294 | 0.64737 ± 0.01233 | — | — | — |
| 10° | 3294 | 0.64871 ± 0.01491 | +0.00133 ± 0.00260 | +0.89 | 0.4685 |

Luka 0°→10°: **+0.00133 ± 0.00260** (per ziarno +0.00428 · -0.00065 · +0.00037); monotoniczna ściśle w każdym ziarnie: **False** [True, False, True].

**test@36 − test@4:**

| warunek | średnia ± sd | per ziarno |
|---|---|---|
| EA | +0.19478 ± 0.01763 | +0.17701 · +0.21227 · +0.19507 |
| ED | +0.00691 ± 0.00297 | +0.00397 · +0.00684 · +0.00991 |
| EB | +0.00384 ± 0.00201 | +0.00608 · +0.00322 · +0.00221 |

**EA − EB:** test@36 = +0.20882 ± 0.01132; test@4 = +0.01787 ± 0.01128.
**Udział kary na niewidzianych kątach** 1 − (EA−EB)@4/(EA−EB)@36: per ziarno 87.3 · 97.7 · 89.0 %; z średnich **91.4 %**.

## Kontrola względem liczb z `STAN_WYNIKOW.md`

| wielkość | oczekiwane | obliczone | |
|---|---|---|---|
| EA test@36 (średnia) | 0.79104 | 0.79104 | zgodne |
| ED test@36 (średnia) | 0.64432 | 0.64432 | zgodne |
| EB test@36 (średnia) | 0.58223 | 0.58223 | zgodne |
| ESE test@36 (średnia) | 1.16292 | 1.16292 | zgodne |
| D−A (średnia) | -0.14672 | -0.14672 | zgodne |
| D−A p (ttest_1samp) | 0.0026 | 0.00262 | zgodne |
| udział gęstości, średnia [%] | 70.2 | 70.18541 | zgodne |
| udział gęstości, sd [pp] | 3.0 | 2.95699 | zgodne |
| K=9 − K=36 (Welch Δ) | 0.0191 | 0.01910 | zgodne |
| K=9 − K=36 (Welch p) | 0.0032 | 0.00316 | zgodne |
| luka EA 0→40° (średnia) | 0.31477 | 0.31477 | zgodne |
| luka EA 0→40° (sd) | 0.0286 | 0.02860 | zgodne |
| ED test@36−test@4 | 0.00691 | 0.00691 | zgodne |
| udział 91,4 % (z średnich) | 91.4 | 91.44125 | zgodne |
