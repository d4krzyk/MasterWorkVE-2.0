# Które metryki i na jakim zbiorze — fakty z plików (tylko odczyt)

Sprawdzono: `pretext/pretext_*/best.json` (12), `pretext_transfer/*/best.json` (43), `probing/probe_depth_*/best.json` (15), `probing/probe_aux_*/status.json` (12; brak `best.json`).

| grupa | metryka | zbiór wg pliku | liczba | wybór checkpointu |
|---|---|---|---|---|
| `pretext/pretext_*` (12) | MAAE_deg (+ top1, acc_within_*) | `split: "val"` we wszystkich 12 | `n = 2928` par (config: `n_val_pairs` 2928) | `best.json` = krok z najmniejszym MAAE na val (np. K4_seed0: krok 12000, MAAE 61,23°); `best_encoder.pth` stąd trafia do transferu |
| `pretext_transfer/*` (43) | RMSE głębi (`all.RMSE`, też edge/smooth/per_scene) | `split: "val"` we wszystkich 43 | `n_samples = 6588`, `n_pixels = 97 047 320` | najmniejszy val RMSE (`status.json: best_val_rmse`, np. K4_seed0_seed0: 0,29097 w kroku 36000); **to RMSE walidacyjne, nie test@36** |
| `probing/probe_depth_*` (15) | RMSE głębi (`all.RMSE`…) | pole `split` nieobecne; kod `probe.py` ewaluuje `DatasetConfig(mode="val", angle_subset="all")` | `n_samples = 6588`, `n_pixels = 97 047 320` (identycznie jak transfer) | najmniejszy val RMSE (`probe.py`: `if r["all"]["RMSE"] < best["rmse"]`), np. random_seed0: 0,55792 w kroku 22000 |
| `probing/probe_aux_*` (12) | MAAE_deg, top1 orientacji (36 klas) i sceny (15 klas) | w `status.json.wyniki`; podzbiór własny sondy | `n_eval = 9900`, `n_train = 39564` (razem 49 464 = rozmiar zbioru treningowego transferu), `aux_eval_fraction = 0.2`, `aux_split_seed = 20260817` | brak wyboru checkpointu (cechy zamrożone, klasyfikator dopasowany na cache cech) |

## Wnioski wynikające z plików
- Wszystkie 43 + 15 przebiegów głębi raportuje RMSE **walidacyjne** na tym samym zbiorze (6588 próbek = 183 lokalizacje × 36 orientacji, ta sama liczba pikseli), więc są porównywalne między sobą, ale **nie z test@36** pełnych modeli (inna część podziału; `eval.json`).
- Kryterium wyboru checkpointu w transferze i sondach jest to samo (min val RMSE, 36 orientacji, krok co 1000), natomiast w pretekście — MAAE (inna metryka, inne zadanie); enkoder do transferu wybrano więc po MAAE, nie po głębi.
- Pretekst ma osobny zbiór walidacyjny par (2928), nie próbek głębi.
- `probe_aux_*` mierzy się na 20 % **zbioru treningowego** podzielonym losowo (seed 20260817), a nie na val/test. Komentarz w `probe.py` (l. ~387) mówi o scenach held-out, ale liczby `n_eval + n_train = 49 464` wskazują na podział próbek treningowego zbioru; nie sprawdzałem, czy podział jest per lokalizacja (kod `_aux_split`, l. 383–).
- Test (`mode="test"`) nie występuje w `probe.py` ani w tych plikach.
- W `status.json` transferów `budget_ceiling_warning: true` (np. K4_seed0_seed0, najlepszy krok 36000) — pole z `_plateau_diagnostics`, które ma znany błąd klucza `overall`/`all`; dla porównania nowa tabela plateau jest w `plateau_recomputed.md`.
