# ED_seed0 (2026-08-11) a powtórzenia nf (2026-11-12): gdzie leży różnica 0,0055 RMSE test@36

Tylko CPU i tylko odczyt. Liczby pochodzą ze skryptu `outputs/ml/overnight_2026-11/ed_vs_nf_analysis.py`
(`CUDA_VISIBLE_DEVICES="" python -I ...`), który czyta `outputs/ml/eval/<id>/samples_test@36.npz` i `eval.json`
dla jawnie podanych runów: ED_seed0, ED_seed0_nf0..nf4, ED_seed1, ED_seed1_nf0, ED_seed2, ED_seed2_nf0. Kilka liczb
dodatkowych, opisanych w §5, pochodzi z `outputs/ml/runs/<id>/{config,status}.json`, `metrics.jsonl` i `train_loss.csv`.

Oznaczenia: **FAKT** to bezpośredni pomiar lub odczyt. **HIPOTEZA** to interpretacja, której te dane nie rozstrzygają.

## 0. Kontrola odtworzenia (FAKT)

- Wszystkie 10 tabel ma 6 588 wierszy (183 lokalizacje × 36 kątów). Klucze (scene_idx, location_id, angle_deg)
  i `all.n_px` są identyczne we wszystkich runach.
- RMSE liczone jako `sqrt(Σall.sq / Σall.n_px)` (tak samo jak w `SampleTable.aggregate`) zgadza się z
  `eval.json` z dokładnością **0,0** (max |Δ| = 0,00e+00) zarówno dla `overall`, jak i dla `per_scene` we
  wszystkich 10 runach. Dla ED_seed0 odtworzono 0,64695049, a eval.json podaje 0,64695049.
- Addytywną miarą jest wkład do sumy kwadratów: ΔSQ = Σ(sq_oryg − sq_ref), a udział grupy to
  ΔSQ_grupa / ΔSQ_razem. Udziały ujemne oznaczają, że w danej grupie oryginał jest **gorszy**. Udziały
  powyżej 100 % w innych grupach to kompensują.
- Szum: z = ΔSQ / (sd_nf · √1,2), gdzie sd_nf to odchylenie standardowe sumy grupowej między pięcioma nf
  (4 stopnie swobody), a √1,2 wynika z tego, że oryginał minus średnia z 5 ma pod H0 wariancję σ²(1+1/5).
  **Przy 4 stopniach swobody z ma rozkład zbliżony do t₄, a nie N(0,1).** Wartość |z| = 2,78 odpowiada
  p = 0,05 (dwustronnie), a E[z²] = 2.

## 1. Ziarno 0: ED_seed0 a średnia z nf0..4

### (i) Per scena (FAKT)

| scena | udział pikseli | RMSE oryg. | RMSE śr. nf (sd nf) | ΔRMSE | ΔSQ | **udział w ΔSQ** | z |
|---|---|---|---|---|---|---|---|
| apartment_2 | 38,3 % | 0,73412 | 0,73214 (0,00569) | +0,00198 | +106 832 | **−15,4 %** | +0,31 |
| frl_apartment_5 | 38,6 % | 0,63786 | 0,64206 (0,00446) | −0,00420 | −201 423 | **+29,1 %** | −0,86 |
| office_4 | 23,1 % | 0,48866 | 0,51518 (0,00872) | **−0,02652** | −598 048 | **+86,3 %** | **−2,72** |
| RAZEM | 100 % | 0,64695 | 0,65245 (0,00188) | −0,00550 | −692 640 | 100 % | −2,67 |

- FAKT: **86 % przewagi oryginału przypada na office_4**, który daje 23 % pikseli. Na apartment_2 oryginał jest
  nieco gorszy (−15 %), a frl_apartment_5 dokłada 29 %, w granicach szumu (z = −0,86).
- FAKT: ten obraz powtarza się względem każdego nf z osobna. Udział office_4 wynosi kolejno +148 %, +88 %,
  +67 %, +59 % i +94 % (dla nf0..nf4), a ΔRMSE oryg − nf_i to −0,00448, −0,00291, −0,00782, −0,00636 i −0,00593.
  Udziały apartment_2 i frl_apartment_5 zmieniają znak w zależności od tego, z którym nf porównujemy.
- Szum między nf: sd ΔSQ dla pojedynczego runu wynosi 309 902 dla apartment_2, 213 820 dla frl_apartment_5,
  200 985 dla office_4 i 236 946 łącznie. sd RMSE per scena: 0,00569, 0,00446, 0,00872. W porównaniach
  „nf_i − średnia pozostałych 4” sumy per scena sięgają ±578 000 (np. nf2 na apartment_2: +577 636),
  **ale po dwóch scenach o przeciwnych znakach się znoszą**. U oryginału dwie z trzech scen idą w tę samą
  stronę, a office_4 wypada na granicy szumu (z = −2,72, p ≈ 0,053 dla t₄).

### (ii) Per kąt (FAKT)

- Kąty kardynalne 0/90/180/270 dają **+6,8 %** różnicy przy 11,1 % próbek, a pozostałe 32 kąty +93,2 %.
  Kąty kardynalne nie są więc wyróżnione.
- Ćwiartki [0–80, 90–170, 180–260, 270–350]: +34,5 %, +26,0 %, **−8,0 %**, +47,5 %. Znak zgodny z różnicą
  całkowitą ma 21 z 36 kątów.
- Na pozór różnica jest skupiona: okno 60–80° niesie 47 % różnicy netto (przy oczekiwanych 8,3 %),
  40–90° 61 %, a 4 największe kąty dają 50 %. **Jest to jednak efekt małej różnicy netto przy dużym
  rozrzucie brutto.** Spójność kątowa Σd/Σ|d| wynosi −0,53. Liczone względem Σ|d|, najlepsze okno trzech
  kątów u oryginału daje 0,247, a w pięciu porównaniach nf_i − średnia pozostałych od 0,159 do 0,241
  (okna w różnych miejscach: 120°, 300°, 140°, 100°, 300°). **Koncentracja kątowa nie odbiega od szumu.**
  Σz² po 36 kątach wynosi 53,5, przy oczekiwanych ok. 72 pod H0 dla t₄, a max |z| = 3,87.
- Okno 40–90° (−422 030 z −692 640) rozkłada się na sceny: office_4 −275 780, frl_apartment_5 −83 671,
  apartment_2 −62 580. Pięć lokalizacji niesie 55,5 % tego okna.

### (iii) Per lokalizacja, 183 lokalizacje (FAKT)

- Oryginał jest lepszy w 101 ze 183 lokalizacji (55,2 %). W porównaniach nf_i − średnia pozostałych liczba
  lokalizacji ze znakiem zgodnym z różnicą wynosi 85–111.
- Udziały w różnicy netto: top 1 lokalizacja daje +20,9 %, top 5 +65,5 %, **top 10 % (18 lokalizacji)
  +164 %**, top 25 % +259 %. Wartości powyżej 100 % oznaczają, że reszta lokalizacji w sumie działa na
  korzyść nf. Różnica netto jest więc małą resztą po dużej wymianie zysków i strat, a nie efektem kilku
  lokalizacji.
- Skupienie brutto: top 10 % lokalizacji według |d| niesie 34,5 % Σ|d|, wobec 34,7–41,8 % w porównaniach
  nf–nf. **Rozkład per lokalizacja ma taką samą strukturę jak szum między nf.** Spójność Σd/Σ|d| wynosi −0,191,
  wobec od −0,112 do +0,096 dla nf–nf, czyli jest nieco wyższa, ale tego samego rzędu.
- Wśród top 10 % lokalizacji 3 pochodzą z apartment_2, 7 z frl_apartment_5 i 8 z office_4. Wewnątrz office_4
  (38 lokalizacji) oryginał jest lepszy w 24/38, a top 10 % (3 lokalizacje) niesie 47 % sumy sceny.
- z per lokalizacja: średnia −0,077 (sd 1,19), |z| > 3 w 5 lokalizacjach (pod t₄ oczekiwane ok. 7), z < −2
  w 9 i z > +2 w 6. **Nie ma garstki lokalizacji odstających.**
- Wzorzec per lokalizacja (oryg − śr. nf) koreluje ze wzorcami nf_i − średnia pozostałych na poziomie
  +0,18, +0,18, −0,11, −0,29 i +0,00. Można to traktować jako brak wspólnej struktury.

### (iv) Kąty treningowe random_4: punkt w praktyce bez treści (FAKT + uzasadnienie)

- Lokalizacje testowe (183) leżą w scenach wyłączonych z treningu (apartment_2, frl_apartment_5, office_4).
  Wspólne lokalizacje test/train: 0. Sceny testowe w train: brak (`outputs/ml/splits/replica_locations.json`).
  Test używa wszystkich 36 kątów każdej lokalizacji, więc dla żadnej próbki testowej „kąt widziany w
  treningu tej lokalizacji” nie istnieje.
- Sens ma tylko wariant globalny. angle_seed = 0 we wszystkich runach ED (z `config.json`), więc zbiór
  treningowy (5 496 próbek) jest identyczny dla oryginału i dla nf. Liczebność per kąt wynosi 125–185, a dla
  kątów kardynalnych 159/163/169/138. Korelacja Spearmana między liczebnością kąta w treningu a ΔSQ per kąt
  (oryg − śr. nf) wynosi **+0,06**, a dla nf_i − średnia pozostałych od −0,27 do +0,36. **Brak związku.**

## 2. Ziarna 1 i 2 (po jednej parze; szum = sd par nf z ziarna 0 · √2, przy założeniu podobnego szumu)

| para | ΔRMSE | udział ΔSQ: apartment_2 / frl_apt_5 / office_4 | z per scena | z razem | kardynalne | top 10 % lok. (netto) | lok. zgodne ze znakiem |
|---|---|---|---|---|---|---|---|
| ED_seed1 − nf0 | **+0,00069** | −391 % / +770 % / −279 % | −0,77 / **+2,19** / −0,84 | +0,26 | +36 % | +1321 % | 108/183 |
| ED_seed2 − nf0 | **−0,00494** | +61 % / −1 % / +40 % | −0,86 / +0,02 / −0,87 | −1,84 | +38 % | +261 % | 93/183 |

ΔRMSE per scena (FAKT): w ziarnie 1 apartment_2 −0,00626, frl_apartment_5 +0,01423, office_4 −0,01048.
W ziarnie 2 te same sceny dają −0,00700, +0,00013 i −0,01107.

- FAKT: w ziarnie 1 różnica netto jest bliska zera i powstaje z dużych zmian o przeciwnych znakach na scenach
  (frl_apartment_5 +2,19). W ziarnie 2 rozkłada się na apartment_2 (61 %) i office_4 (40 %), a każda scena
  osobno mieści się w szumie (|z| < 0,9).
- FAKT: w żadnym ziarnie nie powtarza się skupienie kątowe z ziarna 0. Najlepsze okna to 70–90° w ziarnie 1,
  a 100–120° i 310–0° w ziarnie 2. Spójność kątowa wynosi +0,04 i −0,31. Σz² to 66,1 i 51,2 (pod H0 ok. 72).
- FAKT: per lokalizacja obraz jest taki sam jak w ziarnie 0. Top 10 % według |d| niesie 36,5 % i 36,8 % Σ|d|,
  a spójność lokalizacyjna wynosi +0,017 i −0,122. |z| > 3: 9 i 12 lokalizacji. Rozrzut jest nieco szerszy niż
  w ziarnie 0 (sd z 1,38 i 1,55), czego należy się spodziewać, bo para ma wariancję 2σ², a nie 1,2σ².
- FAKT: office_4 to jedyna scena, w której oryginał jest lepszy we wszystkich trzech ziarnach (ΔRMSE
  −0,0265, −0,0105, −0,0111). Tylko w ziarnie 0 przekracza to jednak szum.
- HIPOTEZA: oryginał jest lepszy w 2 z 3 ziaren (−0,0055, +0,0007, −0,0049). Może to wskazywać na niewielkie
  systematyczne przesunięcie sierpień → listopad (wersje bibliotek lub sterownika, stan GPU). Przy jednej parze
  dla ziaren 1 i 2 i z ≈ −1,8 / +0,3 nie da się tego odróżnić od szumu, a ten materiał niczego nie dowodzi.

## 3. Ziarno 0, nf osobno: rozrzut między samymi powtórzeniami (FAKT)

| nf_i − śr. pozostałych 4 | ΔRMSE | ΔSQ apartment_2 | ΔSQ frl_apt_5 | ΔSQ office_4 |
|---|---|---|---|---|
| nf0 | −0,00128 | −175 753 | −280 685 | +294 801 |
| nf1 | −0,00324 | −345 508 | +280 967 | −344 253 |
| nf2 | +0,00290 | +577 636 | −283 494 | +72 767 |
| nf3 | +0,00108 | +211 321 | +80 194 | −155 679 |
| nf4 | +0,00054 | −267 696 | +203 018 | +132 364 |
| **oryg − śr. 5 nf** | **−0,00550** | +106 832 | −201 423 | **−598 048** |

Zakres per scena w porównaniach nf–nf sięga ±578 000. Wartość −598 048 dla office_4 to najbardziej skrajna
suma per scena w tej tabeli, choć nie odstaje rażąco (sd dla office_4: 200 985). Różnica całkowita −0,0055
jest 1,7× większa niż największa różnica nf–nf (−0,00324).

## 4. Kontrola zmian kodu i protokołu (ZADANIE 2)

### (a) Git (FAKT)

- `git stash list`: pusto.
- `git reflog` w oknie 2026-08-10..13 (czas lokalny CEST):
  - `e2fbf45e HEAD@{2026-08-12 15:03:38}: reset: moving to HEAD`
  - `e5eff1a1 HEAD@{2026-08-12 15:07:10}: commit: Measure angular density effect on echo2depth and reorganise ML package`
  - `8ede18ae HEAD@{2026-08-13 15:09:08}: commit: Measure angular density saturation and diagnose null transfer ...`
- `git log --all` 2026-08-04..14: `e2fbf45e 2026-08-05 18:20` (Add ML training layer), `e5eff1a1 2026-08-12
  15:07`, `8ede18ae 2026-08-13 15:09`. **Między 05.08 18:20 a 12.08 15:07 nie ma żadnego commitu.** Trening ED
  (11.08) szedł więc na niezacommitowanym drzewie roboczym, którego stanu git nie przechowuje. e5eff1a1
  przenosi `train_condition.py` do `depth_model/` (223 zmienione linie względem stanu z 05.08) i zawiera całą pracę z
  06–12.08 naraz.
- Zmiany kodu treningowego **po** e5eff1a1, czyli to, co odróżnia kod z 12.08 od kodu nf
  (`git log -p` dla `train_condition.py`, `echo_data.py`, `echo_h5_dataset.py`, `metrics.py`, `angles.py`):
  - 8ede18ae zmienia tylko komentarz przy statystykach ImageNet ("Wartości zostają niezmienione").
  - 9bb9327a dodaje `dataset_filter=None` w `build_dataloader` ("Gdy None (domyslnie) -- zachowanie bez zmian").
  - e8c567f6 (2026-11-12 16:58 CET) dodaje `--tag`.
  - Niezacommitowany diff `train_condition.py` dodaje `--angle-seed` (domyślnie None), ma mtime 13.11 07:15,
    czyli powstał po runach nf.
  - [W] Żadna z tych zmian nie zmienia ścieżki treningu ED.

### (b) mtime plików .py w `my-operations/ml` (czas lokalny CEST; FAKT)

- W oknie 11.08–13.08 mtime ma 17 plików: 16 z 12.08 14:19–14:23 (pairs, analysis/{__init__,geometry_check},
  checks/*, dataset/{angles,__init__,splits,verify_loader,echo_data}, depth_model/{fast_bilinear,__init__,metrics},
  matrix/__init__, train_pretext) oraz `depth_model/evaluate.py` z 13.08 14:15.
- **Żaden plik nie ma mtime między startem treningu ED (11.08 23:18 CEST = 21:18 UTC) a ewaluacją (12.08
  13:30 CEST = 11:30 UTC).** Paczka z 12.08 14:20 to reorganizacja pakietu z raportu 12.08 §1, wykonana
  **po** ewaluacji. `train_condition.py` ma mtime 13.11 07:15, a `echo_h5_dataset.py` 15.08. mtime nie niesie
  więc informacji o stanie kodu z 11.08.
- Dane wejściowe: żaden plik `outputs/echoes_36deg/*/*.h5` nie ma mtime nowszego niż 11.08 (najpóźniejszy
  to 31.07).

### (c) Wzmianki w raportach sesji (FAKT, cytaty skrócone)

`my-operations/docs/RAPORT_SESJI_2026-08-11.md`:
- l. 52–53: „Wszystkie warunki walidują teraz na `angle_subset="all"` (36 kątów) … `VAL_ANGLE_SUBSET = "all"`”.
- l. 63–68: val@4 liczone z tych samych predykcji, `status.json` z `best_val36_*` / `best_val4_*`, wagi `best_` i `best4_`.
- l. 152: „12 przebiegów (EA/EB/ED/ESE × ziarna 0/1/2), **wszystkie po wprowadzeniu §1**”.
- l. 165: ED 0,64695 / 0,64232 / 0,64369.
- l. 279 (oraz RAPORT_SESJI_2026-08-10 l. 818): ED według starego protokołu, ziarno 0: 0,66308.
- l. 405–407: „`evaluate.py --compare` nie waliduje, że oba przebiegi użyły tego samego protokołu walidacji”.

`my-operations/docs/RAPORT_SESJI_2026-08-12.md`:
- l. 11–60: reorganizacja (`git mv` / `mv`), smoke run `train_condition.py --steps 20`, l. 64–67: poprawka
  `%`→`%%` w helpie `echo_data.py`.
- l. 123–134: asercja protokołu w `evaluate.py --compare` (dotyczy tylko `--compare`).
- l. 211: „Po zmianie `evaluate()` z 2026-08-11 §1 rekordy mają klucz `all`, nie `overall`”.
- **l. 216–227: „`train_condition.py` otwiera `metrics.jsonl` i `train_loss.csv` w trybie dopisywania, a
  `--force` nie kasuje katalogu … `EA_seed0`, `EB_seed0`, `ED_seed0`, `ESE_seed0`: 80 rekordów, 40 stary
  format, 40 nowy … ponowiono z `--force`”.** l. 232: poprawka (kasowanie logów przy świeżym starcie).
- l. 286–291: naprawa ścieżki `TRAIN_SCRIPT` w `exp_ctl` (narzędzie, nie trening).

### Ocena: czy ED_seed0/1/2 mogły być ewaluowane innym kodem niż trenowane?

- FAKT: tak, formalnie. Ewaluacja (12.08 11:30 UTC) odbyła się po treningu (11.08 21:18–22:52 UTC) na
  niezacommitowanym kodzie, a raport 12.08 opisuje zmiany w `evaluate`/`exp_ctl`. Zwrócone RMSE zależy jednak
  wyłącznie od wag: ponowna ewaluacja oryginalnych wag obecnym kodem daje to samo RMSE (|Δ| ≤ 1,1e-7, wg
  briefu), a tabele per próbka zgadzają się co do bitu na kluczach i `n_px`. **Różnica w kodzie ewaluacji nie
  wyjaśnia rozbieżności.**
- FAKT: ED_seed0 to run ponowiony z `--force` w tym samym katalogu. `outputs/ml/runs/ED_seed0/train_loss.csv`
  ma 800 wierszy, z restartem na kroku 100 w linii 402. `metrics.jsonl` ma 40 rekordów starego formatu (val na
  732 próbkach) i 40 nowego (6 588). Wagi `best_audiodepth.pth` (mtime 11.08 23:21 CEST) i `status.json`
  (best_step 9000, val_angle_subset "all", started 21:18:54, ended 21:29:35 UTC) należą do drugiego,
  nowoprotokołowego przebiegu. ED_seed1/2 mają po 40/400 rekordów (pojedynczy przebieg).
- FAKT: oryginał i nf zaczynają z tego samego stanu. Strata na kroku 100 wynosi 1,656106 (oryg) wobec
  1,656112–1,656194 (nf0..4). W ziarnie 1: 1,758748 wobec 1,758539, w ziarnie 2: 1,666864 wobec 1,666995. To
  ta sama inicjalizacja i ta sama kolejność danych, z rozjazdem rzędu niedeterminizmu. Konfiguracja
  (`config.json`) różni się tylko polem `tag` i znacznikiem czasu.
- HIPOTEZA (niesprawdzalna z gita): kod treningu z 11.08 różnił się od e5eff1a1 tylko kosmetycznie (logi,
  ścieżki, help), jak opisuje raport 12.08. Wskazują na to identyczny schemat rekordów walidacji (klucze `all`
  i `val4`, jak w nf) oraz zgodność strat na starcie. Bez snapshotu drzewa z 11.08 nie da się tego wykazać.

## 5. Dodatkowa obserwacja: walidacja (FAKT, `runs/<id>/metrics.jsonl`)

- Najlepszy val@36 (checkpoint wybrany według tej wartości) wynosi dla oryginału **0,63236** (krok 9000), a
  dla nf0..4 0,63112 / 0,62859 / 0,63508 / 0,63420 / 0,62911 (średnia 0,63162, sd ≈ 0,0029). **Na walidacji
  oryginał nie jest lepszy, a wręcz o 0,0007 gorszy.** Średni val@36 w krokach 11k–40k wynosi 0,65757 wobec
  0,64950–0,65554 dla nf. Na office_4 w walidacji (best step) oryginał ma 0,49723 przy średniej nf ≈ 0,49869
  (sd ≈ 0,0076), więc office_4 nie wyróżnia się na walidacji.
- HIPOTEZA: przewaga oryginału na teście dotyczy konkretnego checkpointu (krok 9000) i lokalizacji testowych
  office_4, a nie lepiej wytrenowanej sieci. Taki obraz pasuje do losowego rozjazdu walidacja–test przy
  wyborze checkpointu. Przyczyna nie jest tym wykazana.

## Czy rozbieżność jest wyjaśniona? — CZĘŚCIOWO

Ustalone: przewaga oryginału ED_seed0 (−0,0055 RMSE; ΔSQ = −692 640) nie wynika z kodu ewaluacji (RMSE jest
odtwarzalne co do bitu) ani ze zmiany danych, splitu czy kątów treningowych (identyczne). **Jest skupiona per
scena:** 86 % różnicy przypada na office_4 (z = −2,72, na granicy p ≈ 0,05 dla t₄). Pozostałe sceny mieszczą
się w szumie, a apartment_2 działa przeciwnie (−15 %). Wewnątrz tej skali różnica jest rozłożona szeroko:
kątowo (kardynalne 6,8 % przy 11 % próbek, skupienie 60–80° nie większe niż w porównaniach nf–nf) i per
lokalizacja (oryginał lepszy w 101/183, top 10 % lokalizacji według |d| niesie 34,5 % Σ|d| wobec 35–42 % dla
nf–nf, brak odstających). Różnica całkowita jest 1,7× większa od największej różnicy nf–nf, ale z = −2,67 przy
4 stopniach swobody estymaty szumu daje p ≈ 0,056. **Nie przekracza więc szumu między nf w sposób
rozstrzygający.** Na walidacji oryginał nie jest lepszy niż nf, a ziarna 1 i 2 dają +0,0007 i −0,0049 (w
szumie). Nie znaleziono dowodu na inną przyczynę niż niedeterminizm treningu połączony z wyborem checkpointu.
Systematycznego przesunięcia sierpień → listopad (2 z 3 ziaren na korzyść oryginału) nie można jednak
wykluczyć, bo kod treningu z 11.08 nie istnieje w gicie. Stąd ocena CZĘŚCIOWO, a nie TAK.
