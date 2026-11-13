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
