# RESUME — kontynuacja nocnego zestawu (2026-11-13)

Zasady: nic nie nadpisujemy; nowe wyniki w `outputs/ml/overnight_2026-11/`; bez operacji gita zapisujacych (dozwolone tylko odczyty:
diff/log/status/reflog/stash list); istniejace .py tylko jako nowe opcje z domyslnym zachowaniem (diffy w PLAN.md); `beyond-image-to-depth/` nietkniety.
Srodowisko: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate habitat`, z korzenia repo. tmux NIE jest zainstalowany -> `nohup setsid`.

## Kroki

| krok | status | wyniki |
|---|---|---|
| 0. stan | **zrobione**: brak procesow ML/GPU, brak `ED_*_as*` i RGB-only w `runs/`; kroki 1-4 starego zestawu OK (`PROGRESS.md`, `SUMMARY.md`) | — |
| B1. ponowna ewaluacja 11 przebiegow (+3 ED z wczesniej = 14) | **zrobione**; max \|Δ RMSE test@36\| = 2,4e-7 | `reeval_14_runs.md`, `reeval_14_logs/`, `eval/<id>_reeval/`, `runs/<id>_reeval/` |
| przygotowanie kroku 6: opcja `--angle-seed` w `train_condition.py` (domyslnie None = jak dotad; nadpisanie zapisywane w `config.json` jako `spec.extra.angle_seed`); dry-run domyslny i z `--tag as1 --angle-seed 1` OK | **zrobione**; diff w PLAN.md | `my-operations/ml/depth_model/train_condition.py` |
| przygotowanie kroku 7: skrypt RGB-only `ml/depth_model/train_rgbonly.py` (agent; pary A/D/B zweryfikowane identyczne; ok. 15 min/przebieg) | **zrobione** | `rgbonly/check_pairs.json` |
| A1: roznorodnosc orientacji | **zrobione**: nota w `STAN_WYNIKOW.md` §1 i `USTALENIA_DO_PRACY.md` | `angle_diversity.py/.json/.md` |
| 6. ED z angle_seed 1,2,3 (tagi `_as1.._as3`) | **URUCHOMIONY 2026-11-13 07:19 w `overnight_b.sh` (nohup setsid)** — stan: `PROGRESS_B.md` | `runs/ED_seed0_as{1,2,3}`, `eval/ED_seed0_as{1,2,3}` |
| 7. RGB-only A, D, B x ziarna 0,1,2 (decyzja: trenujemy A, D i B — patrz PLAN.md) | **w kolejce w `overnight_b.sh`** (po kroku 6; ok. 9 x 15 min) | `rgbonly/RGBONLY_<warunek>_seed<N>/` |
| A2 (`OBSERWACJE_METODOLOGICZNE.md`, sekcja "Stan kodu treningu z 11–12.08") | **zrobione** (152 linie dopisane; zadna zmiana z 11.08 nie zmienia ED wzgledem e5eff1a1; stanu z 11.08 nie da sie odtworzyc z gita) |
| A3 (`ED_seed0_vs_nf.md`, `ed_vs_nf_analysis.py`) | **zrobione**: przewaga ED_seed0 skupiona glownie w scenie office_4 (86 % ΔSQ), w katach i lokalizacjach w szumie miedzy nf; rozbieznosc wyjasniona CZESCIOWO (przyczyna nieustalona) | `OBSERWACJE_METODOLOGICZNE.md`, `ED_seed0_vs_nf.md` |

Wazne ustalenie do decyzji o kroku 7: `scratch` w `transfer.py` ma 36 orientacji, 40 000 kroków, batch 32, n_train 49 464, walidacja na 36 — ALE jego `best.json` to RMSE WALIDACYJNE
(checkpoint wybrany na tym samym zbiorze), bez ewaluacji test@36 i test@4. Warunek reguly („walidacja na 36 i test@36") nie jest spelniony -> trenowane sa A, D **i B**.

## Wznowienie (nowa sesja)

1. Stan: `cat outputs/ml/overnight_2026-11/PROGRESS_B.md`; `ps -eo pid,etime,args | grep -E "overnight_b|train_condition|train_rgbonly"`; `nvidia-smi`.
2. Jesli `overnight_b.sh` NIE dziala, a `PROGRESS_B.md` nie ma wiersza `koniec:` — ktos go przerwal. NIE uruchamiaj ponownie calosci (skrypt odmowi: `PROGRESS_B.md` istnieje, a katalogi przebiegow tez). Zamiast tego dokoncz brakujace przebiegi recznie:
   - krok 6: `python my-operations/ml/depth_model/train_condition.py --condition ED --seed 0 --tag as<N> --angle-seed <N>` (+ `evaluate.py --run-dir outputs/ml/runs/ED_seed0_as<N> --test-set test@36`); czesciowy katalog `runs/ED_seed0_as<N>` usun tylko po potwierdzeniu, ze przebieg sie nie ukonczyl (`status.json` bez `finished`), albo uzyj `--resume`;
   - krok 7: `python my-operations/ml/depth_model/train_rgbonly.py --condition <A|D|B> --seed <0|1|2>` (skrypt odmawia przy istniejacym katalogu; `--resume` wznawia);
   - zestawienia: `python outputs/ml/overnight_2026-11/step6_summary.py`, `.../step7_summary.py` (odmawiaja, jesli plik .md juz istnieje), potem `make_summary_b.py` (pisze `SUMMARY_B.md`; stary `SUMMARY.md` to kroki 1-4 i nie jest nadpisywany).
3. Czesc A (dokumentacja/analiza) — patrz tabela wyzej; po zakonczeniu nowa sesja wpisuje wyniki krokow 6/7 do `STAN_WYNIKOW.md` (§11) i `LICZBY_DO_PRACY.md` przez `thesis_numbers.py`.
4. W repo lezy nieśledzony plik `my-operations/docs/Praca_Magisterska_VE-13-11-26.docx` (Word) — nie mój, nie dotykać.

## Dopisek (po części B)
- Część B zakończona 2026-11-13 09:32:50 (133 min), wszystkie kroki OK; wyniki: `step6_summary.md`, `step7_summary.md`, `SUMMARY_B.md`, `rgbonly/`.
- Zadanie tylko-odczyt: `plateau_recomputed.md` (43 przebiegi, ostatnie 40 wierszy `metrics.jsonl`) i `metryki_zbiory_fakty.md` (pretext/transfer/probing: zbiór i wybór checkpointu).
