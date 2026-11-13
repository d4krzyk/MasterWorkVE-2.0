# Kontrole po części B (tylko odczyt)

## a) Przebiegi z najlepszym krokiem w ostatniej ćwiartce (≥ 31 000)

Razem 12 z 43 (źródło: `plateau_recomputed.md`, kolumna `najl_krok`).

| warunek | przebiegi z krokiem ≥ 31 000 / wszystkie | ziarno: najlepszy krok |
|---|---|---|
| A | 3/3 | s0: 37000, s1: 32000, s2: 36000 |
| B | 3/3 | s0: 38000, s1: 39000, s2: 35000 |
| D | 3/3 | s0: 39000, s1: 33000, s2: 34000 |
| EA | 0/3 | s0: 2000, s1: 4000, s2: 2000 |
| EB | 2/3 | s0: 36000, s1: 25000, s2: 38000 |
| ED | 0/3 | s0: 9000, s1: 12000, s2: 7000 |
| EK12 | 0/3 | s0: 9000, s1: 8000, s2: 9000 |
| EK18 | 0/3 | s0: 6000, s1: 7000, s2: 11000 |
| EK6 | 0/3 | s0: 3000, s1: 3000, s2: 5000 |
| EK9 | 0/3 | s0: 3000, s1: 5000, s2: 5000 |
| EPA | 0/3 | s0: 2000, s1: 3000, s2: 6000 |
| EPB | 0/3 | s0: 26000, s1: 14000, s2: 24000 |
| EPD | 0/3 | s0: 4000, s1: 7000, s2: 6000 |
| ESE | 0/3 | s0: 6000, s1: 15000, s2: 5000 |
| SE | 1/1 | s0: 38000 |

Lista: A_seed0 (37000), A_seed1 (32000), A_seed2 (36000), B_seed0 (38000), B_seed1 (39000), B_seed2 (35000), D_seed0 (39000), D_seed1 (33000), D_seed2 (34000), EB_seed0 (36000), EB_seed2 (38000), SE_seed0 (38000).

## b) Podział train/eval w `probe_aux`

- Plik: `my-operations/ml/pretext_model/probe.py`, funkcja `_aux_split` (l. 383–402), wywołana w `cmd_aux` (l. 445: `is_eval = _aux_split(scene_idx, loc_id)`).
- Mechanizm: klucz `scene_idx*100000 + loc_id` (l. 397) → `np.unique` → z unikalnych LOKALIZACJI losowane `round(0,2·N)` bez zwracania (`default_rng(20260817)`, l. 399–401) → maska `k in eval_locs` dla każdej próbki (l. 402). Podział jest **po lokalizacjach**, nie po próbkach (docstring l. 384 to potwierdza).
- Zgodność z `status.json`: n_train = 39564 = 1099·36, n_eval = 9900 = 275·36; 1099 + 275 = 1374 lokalizacji = 1374 treningowych (`angle_diversity.json`); round(0,2·1374) = 275 = 275 ✓.
- Lokalizacje występujące jednocześnie w obu zbiorach: **0** (maska zależy wyłącznie od klucza lokalizacji, więc wszystkie 36 próbek lokalizacji trafia do tej samej strony). Sprawdzone z kodu i arytmetyki; nie odtwarzałem cech na GPU.
- Zastrzeżenie: eval to inne lokalizacje, ale z tych samych scen i pomieszczeń co trening sondy, i to zbiór treningowy głębi (nie val/test) — to zamierzone (docstring l. 385–395: klasą jest tożsamość sceny).

## c) Krok 7 — dokładne liczby (RMSE [m]; średnia ± sd po 3 ziarnach, sd z ddof = 1; p = `ttest_1samp` na różnicach po ziarnach, df = 2)

### test@36

| warunek | RGB-only | model pełny | wartości RGB-only s0/s1/s2 | wartości pełne s0/s1/s2 |
|---|---|---|---|---|
| A | 0.36365 ± 0.00359 | 0.29248 ± 0.00488 | 0.35962 / 0.36650 / 0.36484 | 0.28739 / 0.29712 / 0.29292 |
| D | 0.34549 ± 0.00243 | 0.27199 ± 0.00265 | 0.34629 / 0.34276 / 0.34742 | 0.26909 / 0.27260 / 0.27429 |
| B | 0.28710 ± 0.00345 | 0.24367 ± 0.00142 | 0.29069 / 0.28381 / 0.28680 | 0.24205 / 0.24470 / 0.24425 |

Kontrasty (X − Y, ujemne = X lepsze):

| kontrast | RGB-only | model pełny |
|---|---|---|
| D − A | -0.01817 ± 0.00525 (p = 0.0267) | -0.02048 ± 0.00350 (p = 0.0096) |
| B − D | -0.05839 ± 0.00255 (p = 0.0006) | -0.02833 ± 0.00154 (p = 0.0010) |
| B − A | -0.07655 ± 0.00700 (p = 0.0028) | -0.04881 ± 0.00355 (p = 0.0018) |

Udział (D − A)/(B − A) per ziarno (para (D_s − A_s)/(B_s − A_s); ziarna sparowane indeksem):

| ziarno | RGB-only | model pełny |
|---|---|---|
| 0 | 0.193 | 0.404 |
| 1 | 0.287 | 0.468 |
| 2 | 0.223 | 0.383 |
| średnia ± sd | 0.23459 ± 0.04790 | 0.41808 ± 0.04422 |
| iloraz średnich | 0.237 | 0.420 |

### test@4

| warunek | RGB-only | model pełny | wartości RGB-only s0/s1/s2 | wartości pełne s0/s1/s2 |
|---|---|---|---|---|
| A | 0.35857 ± 0.00291 | 0.28422 ± 0.00403 | 0.35718 / 0.36192 / 0.35662 | 0.27956 / 0.28662 / 0.28647 |
| D | 0.35387 ± 0.00278 | 0.27657 ± 0.00204 | 0.35142 / 0.35330 / 0.35688 | 0.27459 / 0.27643 / 0.27867 |
| B | 0.29620 ± 0.00334 | 0.24739 ± 0.00043 | 0.30005 / 0.29413 / 0.29442 | 0.24690 / 0.24766 / 0.24761 |

Kontrasty (X − Y, ujemne = X lepsze):

| kontrast | RGB-only | model pełny |
|---|---|---|
| D − A | -0.00471 ± 0.00454 (p = 0.2143) | -0.00765 ± 0.00262 (p = 0.0368) |
| B − D | -0.05767 ± 0.00570 (p = 0.0032) | -0.02917 ± 0.00172 (p = 0.0012) |
| B − A | -0.06237 ± 0.00533 (p = 0.0024) | -0.03682 ± 0.00361 (p = 0.0032) |

Udział (D − A)/(B − A) per ziarno (para (D_s − A_s)/(B_s − A_s); ziarna sparowane indeksem):

| ziarno | RGB-only | model pełny |
|---|---|---|
| 0 | 0.101 | 0.152 |
| 1 | 0.127 | 0.262 |
| 2 | -0.004 | 0.201 |
| średnia ± sd | 0.07458 ± 0.06956 | 0.20478 ± 0.05487 |
| iloraz średnich | 0.075 | 0.208 |

### RGB-only B vs SE_seed0 (test@36)

- RGB-only B: ziarna 0.29069 / 0.28381 / 0.28680, średnia 0.28710.
- SE_seed0 (model pełny z potasowanym echem, ziarno 0): 0.26433; B_seed0 (model pełny): 0.24205.
- SE_seed0 − RGB-only B_seed0 = -0.02635; SE_seed0 − średnia RGB-only B = -0.02276 (sd RGB-only B = 0.00345, czyli -6.6 sd). Przy n = 1 dla SE nie ma testu istotności.
- Uwaga: SE_seed0 uczono na wszystkich 36 orientacjach (jak B), więc to właściwe porównanie 'echo bez informacji kątowej' vs 'brak echa'.

