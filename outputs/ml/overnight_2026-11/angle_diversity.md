# Różnorodność orientacji w zbiorze treningowym (1374 lokalizacji)

Źródło: `angles.select_angles` (hash `f"{angle_seed}|{scene}|{location_id}"`) i podział treningowy; `angle_diversity.py`.

| warunek | podzbiór | K siatki | próbek | różnych orientacji w zbiorze | ułamek lokalizacji z daną orientacją: średnia / min / maks. | oczekiwane 4/K |
|---|---|---|---|---|---|---|
| A | `cardinal` (seed 0) | 4 | 5496 | **4** | 1.0000 / 1.0000 / 1.0000 | 1.0000 |
| D | `random_4` (seed 0) | 36 | 5496 | **36** | 0.1111 / 0.0910 / 0.1346 | 0.1111 |
| EK6 | `random_4_of_6` (seed 0) | 6 | 5496 | **6** | 0.6667 / 0.6434 / 0.6841 | 0.6667 |
| EK9 | `random_4_of_9` (seed 0) | 9 | 5496 | **9** | 0.4444 / 0.4214 / 0.4665 | 0.4444 |
| EK12 | `random_4_of_12` (seed 0) | 12 | 5496 | **12** | 0.3333 / 0.3079 / 0.3552 | 0.3333 |
| EK18 | `random_4_of_18` (seed 0) | 18 | 5496 | **18** | 0.2222 / 0.2045 / 0.2409 | 0.2222 |
