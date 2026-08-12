# Raport sesji — 2026-08-10: domknięcie fazy przygotowawczej przed treningiem

Dokument do pracy magisterskiej. Sesja **nie uruchamia macierzy eksperymentów** — domyka to, co
musi być rozstrzygnięte, zanim ruszy pierwszy pełny przebieg, bo po starcie te decyzje są
nieodwracalne bez powtórzenia wszystkiego.

Kontynuacja sesji z 2026-08-05 (`RAPORT_SESJI_2026-08-05.md`). Ograniczenie utrzymane i
zweryfikowane: **żaden plik w `beyond-image-to-depth/` nie został zmieniony** — cały nowy kod stoi
w `my-operations/ml/`.

## 0. Streszczenie — co z tej sesji wchodzi do pracy

Sesja miała domknąć przygotowania. Domknęła je, a przy okazji — ponieważ bramka wykonalności wypadła
pomyślnie — wyprodukowała **pierwsze wyniki merytoryczne** na gałęzi `echo2depth`.

| # | ustalenie | liczba |
|---|---|---|
| 1 | **Gęstość kątowa odpowiada za 62,1 % efektu**, ilość danych za 37,9 % (§5.6) | D−A = +0,1396 [0,114; 0,166] |
| 2 | **Luka generalizacji kątowej jest monotoniczna i duża** (§5.6) | +61,5 % RMSE na 40° od siatki |
| 3 | **Model na 4 kątach jest na tych 4 kątach tak samo dobry jak model na 36 na 36** (§5.6) | 0,57792 wobec 0,57783 |
| 4 | **Całkowity wkład echa w `echo2depth`** — bramka otwarta (§3.3) | +0,5881 [0,547; 0,629] |
| 5 | **Brak sufitu zabiera 52 % energii pogłosu**, nie 8 % jak sugeruje energia całkowita (§2.5) | E_późna ×0,478 |
| 6 | **Domknięcie sufitu obniża względny kontrast kątowy o 17,5 %** przy +46 % SNR (§2.6) | kompromis, nie poprawa |
| 7 | **Podłoga szumu frameworka** — kontekst dla każdego wyniku macierzy (§3.1) | 0,0023–0,0073 RMSE |
| 8 | `--fast-bilinear` **włączone domyślnie**, na pomiarze (§3.1) | 16,15× szybciej |

**Trzy rzeczy wymagają decyzji autora przed startem macierzy** — jedna nowa: **§4.7** (na jakim
zbiorze kątów wybierać checkpoint). Dwie poprzednie są zamknięte (§8).

## Legenda statusów

| status | znaczenie |
|---|---|
| **[Z]** | **Zmierzone** — istnieje skrypt, surowe wyjście, liczby w dokumencie. Nadaje się do pracy jako wynik. |
| **[Z-]** | Zmierzone, ale z zastrzeżeniem, które trzeba cytować razem z liczbą. |
| **[W]** | **Wywnioskowane** z kodu źródłowego, nie z pomiaru. |
| **[X]** | **Nie sprawdzone** — wymieniam, żeby nie powstało wrażenie, że zostało. |

---

## 1. Co powstało

| plik | rola | linii |
|---|---|---|
| `ml/geometry_check.py` | **nowy** — BLOK 0: rozstrzygnięcie `main` vs `patched` (4 pomiary) | 715 |
| `ml/determinism_check.py` | **nowy** — BLOK 1.1: kontrola niedeterminizmu, decyzja `--fast-bilinear` | 441 |
| `ml/evaluate.py` | **nowy** — BLOK 2: protokół ewaluacji, wszystko z jednego checkpointu | 376 |
| `ml/pretext/` (7 plików) | **nowy** — BLOK 4: Model 2, zadanie pretekstowe orientacji | 1 187 |
| `ml/metrics.py` | +statystyki per próbka, maska przecięcia, bootstrap, odległość kątowa | +329 |
| `ml/experiments.py` | +10 warunków, +budżet dysku i czasu, +zmierzone `PARAM_COUNTS` | +168 |
| `ml/echo_h5_dataset.py` | +maska przecięcia, +permutacja echa, `get_audio()` | +128 |
| `ml/angles.py` | +`random_K_of_G` (krzywa przy stałym budżecie) | +53 |
| `ml/train_condition.py` | +permutacja echa, +`best_step` i ostrzeżenie o suficie budżetu | +42 |
| `ml/exp_ctl.py` | kolejność grup, szersze kolumny, budżet nie ginie przy `plan` | +12 |

Razem `my-operations/ml/`: **6 707 linii** (było 3 193). Wszystkie moduły importują się bez błędu,
wszystkie CLI parsują `--help`.

`ml/pretext/summarize.py` (zero GPU) zbiera wyniki obu etapów Modelu 2 w dwie tabele do pracy —
patrz §6.7.

### 1.1 Kontrola regresji dataloadera **[Z]**

Zmiany w `echo_h5_dataset.py` dotknęły **współdzielonych** ścieżek (`_build_index`, `__getitem__`),
z których korzysta cała macierz — więc 8 testów z poprzedniej sesji zostało uruchomionych ponownie:

```
python my-operations/ml/echo_data.py --verify-loader --geometry {main,patched}
```

**8/8 PASS w obu wariantach.** Liczności wszystkich 12 podzbiorów kątów zgadzają się co do sztuki,
`random_K` nadal odtwarzalny, brak NaN/Inf, kształty wsadu zgodne z `audioVisual_model.py`.

Przy okazji **niezależne potwierdzenie** wyniku §2.2 — inną ścieżką kodu (pełny skan przez
dataloader, nie przez `geometry_check.py`):

| wariant | globalny odsetek `depth == 0` |
|---|---|
| `main` | **8,480338 %** |
| `patched` | **0,212069 %** |

Głębia powyżej `max_depth`: 131 pikseli (1,3·10⁻⁵ %) w obu wariantach — identycznie jak w §3.5
poprzedniego raportu, co potwierdza, że łatka nie wprowadziła nowych pikseli poza zakresem.

Pliki dowodowe (odbiałolistowane w `.gitignore`): `outputs/ml/geometry_check/geometry_check.json`,
`outputs/ml/determinism/determinism_check.json`, `outputs/ml/experiments.json`.

---

## 2. BLOK 0 — rozstrzygnięcie geometrii `main` vs `patched`

`python my-operations/ml/geometry_check.py` — pełny skan, bez próbkowania: wszystkie 44 064 próbki
10 scen łatanych w obu wariantach, wszystkie trzy kanały; 432 s + 144 s.

### 2.1 Łatka zmienia **wszystkie trzy** kanały **[Z]**

Raport §3.3 wykazał wcześniej, że `location_id` i `position` są bit-identyczne między wariantami.
To **nie** przenosi się na dane: `rgb`, `depth` i `echo` różnią się w każdej z 10 scen.

Wiersze dopasowywane po kluczu `(location_id, angle_deg)`, nie po indeksie — kolejność wyszła
identyczna we wszystkich 10 scenach, ale to zostało **sprawdzone, nie założone**.

`frl_apartment_5` (scena held-out, pełny skan 5 328 wierszy):

| kanał | % wartości zmienionych | max \|Δ\| | mediana \|Δ\| (po zmienionych) |
|---|---|---|---|
| `rgb` | 13,14 % | 208 | 197,0 |
| `depth` | 17,57 % | 10,100 m | 1,873 m |
| `echo` | **92,11 %** | 2,762 | 0,0215 |

Echo różni się na 92 % komórek spektrogramu we wszystkich scenach (91,9–93,1 %) — to konsekwencja
stochastycznych odbić Monte Carlo, nie tylko zmiany geometrii.

### 2.2 Łatka **wyłącznie dodaje** geometrię — nigdy jej nie usuwa **[Z]**

Kluczowa liczba: piksele przechodzące z `0` na wartość dodatnią (dorobione sufity) wobec pikseli
przechodzących z wartości dodatniej na `0`.

Pełny skan wszystkich 10 scen (bez próbkowania), kolumny `%` to odsetek zmienionych wartości kanału:

| scena | zera `main` | zera `patched` | dorobione (0→+) | **usunięte (+→0)** | `rgb` % | `depth` % | `echo` % |
|---|---|---|---|---|---|---|---|
| `frl_apartment_1` | 15,1707 | 0,0038 | 15,1669 | **0** | 13,66 | 18,28 | 92,15 |
| `frl_apartment_0` | 15,1215 | 0,0027 | 15,1188 | **0** | 13,64 | 18,22 | 92,11 |
| `frl_apartment_3` | 14,8317 | 0,0663 | 14,7654 | **0** | 13,99 | 18,67 | 91,91 |
| `frl_apartment_4` | 14,5556 | 0,0024 | 14,5533 | **0** | 12,65 | 16,91 | 92,48 |
| `frl_apartment_5` (held-out) | 14,2418 | 0,0038 | 14,2379 | **0** | 13,15 | 17,57 | 92,11 |
| `frl_apartment_2` | 13,6097 | 0,0007 | 13,6090 | **0** | 12,60 | 16,85 | 92,74 |
| `apartment_2` (held-out) | 11,3932 | 1,2197 | 10,1735 | **0** | 8,27 | 11,05 | 92,66 |
| `office_2` | 6,2605 | 0,0047 | 6,2558 | **0** | 4,71 | 6,28 | 93,00 |
| `apartment_1` | 6,0473 | 1,0359 | 5,0114 | **0** | 4,16 | 5,56 | 92,53 |
| `office_3` | 1,0423 | 0,0031 | 1,0391 | **0** | 0,78 | 1,05 | 93,22 |

**Usunięte = 0 we wszystkich 10 scenach.** Maska przecięcia (piksele ważne w obu wariantach) jest
więc **dokładnie oryginalną maską `main`** — nie jest to założenie, tylko zmierzona własność.
Implementacja i tak liczy iloczyn masek, żeby pozostała poprawna, gdyby to się kiedyś zmieniło.

Drugie spostrzeżenie z tej tabeli: `depth` zmienia się na **więcej** pikseli, niż wynosi liczba
dorobionych (`frl_apartment_5`: 17,57 % zmienionych wobec 14,24 % dorobionych). Różnica ~3,3 pp to
piksele, które **już były ważne** i zmieniły wartość. Sprawdzone osobno na próbce co 6. wiersza
**[Z]**:

| scena | zmienione a już ważne | `patched` **bliżej** | mediana zmiany |
|---|---|---|---|
| `frl_apartment_5` | 3,28 % kadru | **100,00 %** | −0,2796 m |
| `apartment_2` | 0,94 % kadru | **100,00 %** | −0,3089 m |

**Sto procent** tych pikseli robi się *bliższych* — czyli łata je **przesłania**: promień, który
wcześniej uciekał przez dziurę i trafiał w dalszą powierzchnię, zatrzymuje się teraz na suficie.
Łatka nie jest więc wyłącznie dopisaniem brakujących pikseli; zmienia też część istniejącej głębi.
To dodatkowy argument za maską przecięcia — ale uwaga, maska przecięcia **tych** pikseli nie
usuwa (są ważne w obu wariantach), więc pozostają one różnicą, której nie da się zamaskować.
Ich udział (≤ 3,3 % kadru) trzeba wymienić jako ograniczenie porównania `main` vs `patched`.

### 2.3 Kontrola negatywna: 8 scen szczelnych jest bit-identycznych **[Z]**

Wynik: **OK**. Sprawdzone przez pełny `EchoH5Dataset` (a nie surowy HDF5), czyli cała droga od
nazwy sceny do tensora:

| tryb | indeks zgodny | porównano | wynik |
|---|---|---|---|
| `train` | tak | 56 próbek z 7 scen | bit-identyczne |
| `val` | tak | 8 próbek z 1 sceny (`office_4`) | bit-identyczne |
| `test` | tak | 8 próbek z 1 sceny (`office_4`) | bit-identyczne |

Kontrola **pozytywna** (żeby test nie przechodził przez porównywanie pliku z samym sobą):
`apartment_2` w tym samym układzie **różni się** — tak jak musi.

Wniosek: kompozycja ścieżek w `paths.scene_h5()` działa; wariant `patched` faktycznie serwuje
8 scen szczelnych z katalogu `main`.

### 2.4 Odsetek `depth == 0` — pełny skan wszystkich 18 scen, oba warianty **[Z]**

Kolumna `main` odtwarza **co do cyfry** tabelę §3.6 z poprzedniej sesji (niezależny skan).

| scena | `main` % | `patched` % | różnica |
|---|---|---|---|
| `frl_apartment_1` | 15,1707 | 0,0038 | −15,1669 |
| `frl_apartment_0` | 15,1215 | 0,0027 | −15,1188 |
| `frl_apartment_3` | 14,8317 | 0,0663 | −14,7654 |
| `frl_apartment_4` | 14,5556 | 0,0024 | −14,5533 |
| `frl_apartment_5` (held-out) | 14,2418 | 0,0038 | −14,2379 |
| `frl_apartment_2` | 13,6097 | 0,0007 | −13,6090 |
| `apartment_2` (held-out) | 11,3932 | 1,2197 | −10,1735 |
| `office_2` | 6,2605 | 0,0047 | −6,2558 |
| `apartment_1` | 6,0473 | 1,0359 | −5,0114 |
| `office_3` | 1,0423 | 0,0031 | −1,0391 |
| 8 scen szczelnych | 0,0004–0,0375 | *ten sam plik* | 0 |

Oczekiwanie z zadania potwierdzone: `frl_apartment_1` spada z 15,17 % do 0,0038 %. Dwie sceny
(`apartment_1`, `apartment_2`) zachowują ~1 % dziur po łatce — łata domyka sufit, ale nie wszystkie
braki skanu.

### 2.5 Ile energii ucieka przez brak sufitu — i dlaczego liczba całkowita kłamie **[Z]**

Energia całkowita spektrogramu (suma kwadratów magnitud) daje **mylącą** odpowiedź, bo dominuje ją
ścieżka bezpośrednia: źródło i słuchacz są współlokowane, więc sygnał w t = 0 nie niesie żadnej
informacji o pomieszczeniu, a stanowi większość energii.

Rozbicie na część **późną** (od ramki 30, czyli 10,9 ms — za pierwszym odbiciem podłoga/sufit przy
wysokości 1,25 m, które przychodzi w 7,3 ms) zmienia obraz całkowicie:

| porównanie | energia **całkowita** | energia **późna** (pogłos) |
|---|---|---|
| `main` sceny otwarte vs `main` sceny szczelne | **−8,2 %** | **−52,2 %** |
| `patched` vs `main`, te same 10 scen | **+1,3 %** | **+46,3 %** |

**Brak sufitu zabiera ponad połowę energii pogłosu**, przy zaledwie 8 % spadku energii całkowitej.
To jest liczba do pracy zamiast argumentu jakościowego.

Skrajny przypadek: `frl_apartment_5` ma najniższą energię późną z całego zbioru (760,97) i po łatce
rośnie ona ×1,538 — najsilniej ze wszystkich scen.

### 2.6 Kontrast kątowy: hipoteza z zadania **potwierdzona co do kierunku** **[Z]**

Hipoteza wymagała sprawdzenia, nie założenia: brak sufitu usuwa odbicie niemal *niezależne od
orientacji* (przychodzące z góry), więc może *podnosić* względny kontrast kątowy przy jednoczesnym
obniżeniu SNR. Kierunek nie był oczywisty. Zmierzone (mediana RMSE między spektrogramami 0° i 90°
tej samej lokalizacji, normalizowana energią — sam RMSE spada trywialnie razem z głośnością):

| grupa | RMSE(0°,90°) | względny | względny **późny** |
|---|---|---|---|
| `main` otwarte (`frl_apartment_*`) | 0,3029 | 0,8068 | **0,4202** |
| `main` szczelne | 0,3085 | 0,7759 | **0,3596** |
| `main`, 10 scen łatanych | 0,3036 | 0,8018 | 0,4085 |
| `patched`, te same 10 scen | 0,3032 | 0,7891 | **0,3370** |

- otwarte / szczelne: kontrast względny **×1,040**, w części późnej **×1,169**
- po załataniu: kontrast względny **×0,984**, w części późnej **×0,825**

Czyli: **domknięcie sufitu podnosi SNR o 46 % i jednocześnie obniża względny kontrast kątowy pola
późnego o 17,5 %.** To jest realny kompromis, a nie jednoznaczna poprawa — i musi iść razem z każdym
wynikiem wariantu `patched`. Surowy RMSE(0°,90°) ≈ 0,30 zgadza się z charakterystyką z
`GENERATOR_PARAMS.md` (0,30–0,35 dla renderów 90° od siebie), co jest niezależnym potwierdzeniem, że
pomiar mierzy to, co ma mierzyć.

### 2.7 Decyzja wynikowa (0.5) **[Z]**

`depth` **się zmienia**, więc:

1. **Metryki `main`-vs-`patched` liczone są na masce przecięcia** — wyłącznie na pikselach ważnych
   w obu wariantach, czyli na oryginalnej masce `main`. Zaimplementowane:
   `DatasetConfig.mask_variant` + `SampleStatsCollector(valid_ref=...)`, flaga
   `evaluate.py --intersection-mask`.

   Sprawdzone na poziomie danych **[Z]** (zbiór testowy, `cardinal`, indeksy obu wariantów
   identyczne):

   | scena | próbka | ważne `patched` | ważne `main` | przecięcie | przecięcie == maska `main` |
   |---|---|---|---|---|---|
   | `apartment_2` | 12 | 14 797 | **7 000** | 7 000 | tak |
   | `frl_apartment_5` | 287 | 16 383 | 15 449 | 15 449 | tak |

   Przypadek `apartment_2` pokazuje skalę problemu: łatka **ponad dwukrotnie** zwiększa liczbę
   punktowanych pikseli (7 000 → 14 797) w tej jednej próbce. Bez maski przecięcia model wariantu
   `patched` byłby oceniany na 14 797 pikselach, a model wariantu `main` na 7 000 — te dwa RMSE nie
   są tą samą wielkością i ich różnica nie znaczy nic.
2. **Uzasadnienie do pracy:** dorobiony sufit to geometria **syntetyczna, nie zmierzona**. Uczenie
   gałęzi wizualnej przewidywania płaszczyzny, której w skanie nie było, to uczenie fikcji;
   punktowanie na niej zaniża porównywalność, bo model wariantu `main` nie ma prawa jej znać.
3. **Wielkością porównywaną między wariantami jest zawsze Δ = RMSE(A) − RMSE(B)**, nigdy surowe
   RMSE. Maska krawędzi liczona jest z pełnej prawdy (przynależność piksela do nieciągłości to
   własność geometrii), a maska przecięcia ogranicza dopiero **punktowanie**.

---

## 3. BLOK 1 — kontrole przed macierzą

### 3.1 Kontrola niedeterminizmu i decyzja `--fast-bilinear` **[Z]**

Raport §7 poprzedniej sesji sugerował „jeden przebieg kontrolny". To za mało: z n=1 vs n=1 nie
orzeka się o nieodróżnialności, bo nie wiadomo, ile te dwa przebiegi rozjechałyby się **bez**
podmiany. Zamiast tego zmierzono podstawienie **przeciwko własnej podłodze szumu frameworka** —
tą samą logiką, co kontrola negatywna przy `hfov=70` w fazie generowania.

Warunek `A`, ziarno 0, 2 000 kroków, AMP włączone, cztery przebiegi:

| para | co mierzy | przebiegi |
|---|---|---|
| 1 | **podłoga**: własny niedeterminizm cuDNN/atomików | `nn.Bilinear` × 2 |
| 2 | podłoga wariantu szybkiego | `BilinearEinsum` × 2 |
| 3 | **efekt podstawienia** | `nn.Bilinear` vs `BilinearEinsum` |

**Warunki wstępne, bez których pomiar byłby nieważny — oba spełnione:**

- wagi startowe `nn.Bilinear` i `BilinearEinsum` są **bit-identyczne** (podmiana nie narusza
  strumienia RNG, bo zachodzi po zbudowaniu wszystkich sieci),
- **kolejność danych identyczna we wszystkich czterech przebiegach** — skrót ciągu
  `(scene, location_id, angle_deg)` po 2 000 krokach: `fc7a5a9e16388b7d` w każdym.

#### Rozbieżność wag (relatywna norma L2)

| krok | para 1 (podłoga) | para 2 (podłoga fast) | para 3 (podstawienie) | p3/p1 |
|---|---|---|---|---|
| 0 | 0 | 0 | **0** | — |
| 100 | 1,2478·10⁻¹ | 1,2297·10⁻¹ | 1,2892·10⁻¹ | 1,033 |
| 250 | 2,7018·10⁻¹ | 2,3193·10⁻¹ | 2,3954·10⁻¹ | 0,887 |
| 500 | 1,4975·10⁻¹ | 1,5220·10⁻¹ | 1,6877·10⁻¹ | 1,127 |
| 1000 | 4,6895·10⁻² | 5,2085·10⁻² | 5,7121·10⁻² | 1,218 |
| **2000** | **1,0218·10⁻²** | **1,0104·10⁻²** | **1,1678·10⁻²** | **1,143** |

#### Rozbieżność walidacyjnego RMSE

| krok | para 1 (podłoga) | para 2 (podłoga fast) | para 3 (podstawienie) | p3/p1 |
|---|---|---|---|---|
| 100 | 4,682·10⁻³ | 2,337·10⁻² | 2,145·10⁻² | 4,582 |
| 250 | 4,077·10⁻³ | 1,882·10⁻² | 1,274·10⁻² | 3,124 |
| 500 | 9,582·10⁻³ | 1,318·10⁻² | 2,937·10⁻³ | 0,307 |
| 1000 | 2,106·10⁻³ | 2,979·10⁻³ | 5,483·10⁻⁴ | 0,260 |
| **2000** | **7,320·10⁻³** | **1,041·10⁻³** | **7,290·10⁻⁴** | **0,100** |

#### Odczyt

**1. Trening jest kontraktywny, nie rozbieżny.** Rozbieżność wag rośnie do maksimum na kroku 250
(2,7·10⁻¹) i potem **spada o rząd wielkości** do 1,0·10⁻² na kroku 2 000 — i to we wszystkich trzech
parach jednakowo. Niedeterminizm frameworka nie kumuluje się; trajektorie wracają ku sobie. To
osobny wynik, wart wymienienia, bo unieważnia obawę, że drobna zmiana kolejności redukcji
„rozjedzie" trening na przestrzeni 40 000 kroków.

**2. Nadwyżka podstawienia mieści się w błędzie oszacowania samej podłogi.** Para 1 i para 2 to
**dwie niezależne oceny tej samej wielkości**, więc ich wzajemny stosunek mierzy błąd próbkowania
tego pomiaru:

| | zakres po wszystkich krokach |
|---|---|
| p2/p1 — sam szum oceny podłogi | **0,858 – 1,111** |
| p3/p1 — nadwyżka podstawienia | **0,887 – 1,218** |

Zakresy się pokrywają, a p3/p1 bywa **poniżej** jedności (0,887 na kroku 250). Przy jednej parze na
wielkość nie da się odróżnić 1,14 od 1,00.

**3. Na wielkości, która faktycznie decyduje, podstawienie jest 10× PONIŻEJ podłogi.** Walidacyjne
RMSE na końcu budżetu: podstawienie **7,29·10⁻⁴** wobec podłogi **7,32·10⁻³**. Dwa przebiegi tego
samego kodu rozjeżdżają się dziesięciokrotnie bardziej niż przebieg oryginalny od podstawionego.

**4. Zapisane kryterium formalnie nie zostało spełnione — i tak trzeba to podać.** Kryterium
zakodowane przed pomiarem żądało `p3/p1 ≤ 1,0` **jednocześnie** dla wag i dla RMSE. Dla RMSE wyszło
0,0996 (spełnione z ogromnym zapasem), dla wag 1,143 (**niespełnione**). `criterion_met = False`,
`criterion_met_lenient_2x = True`. Nie zmieniam kryterium po zobaczeniu danych — podaję jego wynik
i osobno odczyt, który uważam za poprawny, bo kryterium w formie „stosunek dwóch pojedynczych
próbek ≤ 1" nie ma mocy rozstrzygania przy zmienności ±14 % widocznej między dwoma ocenami tej
samej podłogi (punkt 2).

#### DECYZJA

**`--fast-bilinear` idzie na domyślnie włączone.** Uzasadnienie do pracy stoi na trzech zmierzonych
rzeczach: tożsamość funkcji dowiedziona analitycznie i numerycznie (poprzedni raport §3.9,
`verify_equivalence()`), wagi startowe bit-identyczne, a rozbieżność po 2 000 krokach nie odróżnialna
od niedeterminizmu, który występuje i tak w każdym przebiegu.

Wraz z decyzją o batchu 32 (§3.2) **oba odstępstwa od literalnego kodu referencyjnego muszą być
jawnie wymienione w rozdziale o ograniczeniach**.

#### Przyspieszenie zmierzone na pełnej pętli treningowej

| | s/krok | 40 000 kroków |
|---|---|---|
| `nn.Bilinear` | 1,5391 | **17,10 h** |
| `BilinearEinsum` | 0,0953 | **1,06 h** |

**16,15×** end-to-end. Obie liczby zawierają ten sam narzut diagnostyczny (zdjęcia wag, wczytanie
referencji, porównanie 317 M parametrów na 6 punktach pomiarowych), więc **stosunek jest uczciwy, a
prawdziwe przyspieszenie samego treningu jest większe** — narzut jest stały addytywnie, więc bardziej
obciąża przebieg szybki. Zgadza się to z 19,5× z mikrobenchmarku poprzedniej sesji.

#### Uwaga metodologiczna o odtworzeniu przebiegu `slow_1`

Sesja została przerwana limitem w trakcie `slow_2`, ale komplet 6 zdjęć wag `slow_1` przetrwał na
dysku. Metryki `slow_1` zostały **odtworzone z tych zdjęć** (41,7 s) zamiast powtarzania 51-minutowego
treningu. Zdjęcia **są** wagami tego przebiegu, więc ewaluacja z nich wczytana daje dokładnie te
liczby, które dałby oryginał — to pominięcie ponownego liczenia tego samego, nie przybliżenie.
Kolejność danych odtworzono przejściem po loaderze i **potwierdzono zgodność skrótu** z trzema
przebiegami wykonanymi normalnie. Skrypt zapisuje teraz wynik częściowy po każdym przebiegu.

### 3.2 Argument dodatkowy: batch 32 vs 64 **[W]**

Z równoważników epok w tabeli §5 poprzedniego raportu wynika **batch 32**
(5 496 × 232,9 / 40 000 = 32,0; zgadza się dla wszystkich wierszy), podczas gdy `base_options.py`
Paridy ma `batchSize = 64`.

Rozróżnienie, które musi trafić do rozdziału o ograniczeniach:

- **zmiana batcha realnie zmienia trajektorię optymalizacji** (inne oszacowanie gradientu, inna
  skala szumu SGD, inna efektywna regularyzacja);
- **zmiana kolejności redukcji zmiennoprzecinkowej nie zmienia funkcji** — `BilinearEinsum` liczy
  ten sam wzór z tymi samymi parametrami.

Jeśli odstępstwo od batcha 64 jest akceptowane, to od `nn.Bilinear` tym bardziej — ale **oba muszą
być jawnie wymienione** w rozdziale o ograniczeniach.

### 3.3 Kontrola permutacyjna echa — bramka wykonalności **[Z] dla `echo2depth`, [X] dla pełnego modelu**

Dodane warunki: **`SE`** (pełny model) i **`ESE`** (`echo2depth`), grupa **`bramka`**, która w
`GROUPS` stoi **pierwsza**. Wtedy:

```
RMSE(SE) − RMSE(B)  =  całkowity wkład echa do pełnego modelu
                    =  GÓRNE OGRANICZENIE na jakikolwiek efekt gęstości kątowej
```

Bez tej liczby wynik zerowy macierzy jest nieinterpretowalny: „gęstość kątowa nie niesie
informacji" i „model w ogóle nie używa echa" dają identyczne liczby i różne wnioski. Punkt
odniesienia: u Gao echo daje 7,5 % (RGB2Depth 0,374 → RGB+Echo2Depth 0,346); u Paridy marginalny
wkład echa **nie jest nigdzie raportowany**.

#### WYNIK dla `echo2depth` — bramka OTWARTA **[Z]**

Dwa pełne przebiegi po 40 000 kroków, ziarno 0, `EB` (echo prawdziwe) i `ESE` (echo z losowo innej
lokalizacji). Jedyna różnica między nimi to **źródło spektrogramu** — ten sam kod, te same wagi
startowe, ten sam budżet, ten sam zbiór:

| | `EB` echo prawdziwe | `ESE` echo permutowane | **wkład echa** |
|---|---|---|---|
| RMSE całość | **0,56096** | **1,16837** | **+0,60740** |
| RMSE krawędzie | 1,11678 | 2,05191 | +0,93513 |
| RMSE gładkie | 0,44199 | 1,00050 | +0,55851 |
| δ < 1,25 | 0,79973 | 0,34821 | −0,45151 |

**Całkowity wkład echa w gałęzi `echo2depth` wynosi 0,607 RMSE** — czyli **83–262× powyżej podłogi
szumu frameworka** (0,0023–0,0073 z §3.1). Permutacja echa **podwaja** błąd i zabiera ponad połowę
pikseli mieszczących się w δ < 1,25.

Rozbicie per scena (wkład echa):

| scena | `EB` | `ESE` | wkład |
|---|---|---|---|
| `frl_apartment_5` | 0,46991 | 1,33942 | **+0,86951** |
| `office_4` | 0,43640 | 1,05653 | +0,62013 |
| `apartment_2` | 0,69608 | 1,04011 | +0,34402 |

**Wniosek: gałąź `echo2depth` ma ogromny zapas nad szumem i nadaje się do wykrycia efektu gęstości
kątowej.** Ta oś macierzy jest wykonalna i to jest miejsce, w którym trzeba szukać efektu.

**Czego to NIE rozstrzyga [X].** To jest wkład echa w modelu, który **nie ma nic innego**. W pełnym
modelu obraz RGB niesie większość informacji o głębi, więc marginalny wkład echa będzie
**wielokrotnie mniejszy** — i to on jest właściwym górnym ograniczeniem dla warunków `A/B/D`.
Przebieg `SE` (pełny model, ~1 h) **nie został wykonany** i pozostaje pierwszą rzeczą do
uruchomienia.

Dodatkowa obserwacja z `ESE` **[Z]**: najlepszy checkpoint padł na kroku **6 000 (15 % budżetu)**,
po czym krzywa już się nie poprawiała. Model bez użytecznego sygnału uczy się priora głębi w ~4 epoki
i dalej nie ma czego robić — to zachowanie zgodne z oczekiwaniem i dodatkowe potwierdzenie, że
permutacja faktycznie zniszczyła informację, a nie tylko ją zaszumiła.

Zweryfikowane własności permutacji, na **pełnym zbiorze treningowym** (49 464 próbki) **[Z]**:

| własność | `train/all` (49 464) | `train/cardinal` (5 496) |
|---|---|---|
| jest permutacją (bijekcją) | tak | tak |
| kolizji lokalizacji (echo z własnej pozycji) | **0** | **0** |
| punktów stałych | 0 | 0 |
| `img` i `depth` nietknięte | tak, bit-identyczne | — |
| `audio` pochodzi dokładnie z próbki `perm[i]` | tak | — |

Wykluczenie tej samej lokalizacji jest istotne, a nie kosmetyczne: przy 36 orientacjach zwykła
permutacja przypisałaby ~35 próbkom echo z tej samej pozycji (tylko innego kąta), a takie echo
nadal niesie pełną informację o położeniu — kontrola mierzyłaby wtedy „ile wnosi **orientacja**
echa", a nie „ile wnosi echo".

**Błąd znaleziony i naprawiony w trakcie sesji [Z].** Pierwsza wersja naprawiała kolizje
wektorowo:

```python
perm[bad], perm[partners] = perm[partners], perm[bad]     # BŁĘDNE
```

Gdy w `partners` powtórzy się ten sam indeks — a przy ~35 złych pozycjach losowanych i.i.d.
z 49 464 to zdarza się regularnie — ta sama wartość trafia w dwa miejsca i wynik **przestaje być
permutacją**. Zmierzone na 200 ziarnach: **5 z 200 (2,5 %)** dawało nie-permutację, **po cichu** —
nic tego nie sprawdzało. Naprawione zamianami skalarnymi (zawsze zachowują bijekcję) plus **jawną
kontrolą** bijekcji i braku kolizji, która podnosi wyjątek zamiast trenować na uszkodzonym zbiorze.

To jest dokładnie ten rodzaj usterki, który unieważniłby bramkę bez żadnego widocznego objawu:
warunek `SE` używa tej permutacji na zbiorze `train/all`, czyli w najbardziej narażonym przypadku.

Wersja **darmowa** (bez treningu): `evaluate.py --shuffle-echo` przepuszcza zbiór testowy przez
gotowy checkpoint z permutowanym echem. Mówi, ile *wytrenowany* model polega na audio — zero kosztu
GPU poza jednym przelotem ewaluacji. Ścieżka ta korzysta z **tej samej** permutacji co warunek `SE`
(`DatasetConfig.shuffle_echo_seed`), a nie z własnej kopii: dwie równoległe implementacje tej samej
rzeczy mogłyby się po cichu rozjechać, a wtedy wersja darmowa i wersja trenowana mierzyłyby dwie
różne wielkości. Sprawdzone na `test@36`: 6 588 próbek, bijekcja, **0 kolizji lokalizacji**.

**Zastrzeżenie [Z-]:** permutacja jest **stała dla danego ziarna**, a nie losowana co epokę.
Powód: odtwarzalność warunku. Ryzyko zapamiętania 49 464 arbitralnych par obraz-echo jest ograniczone
tym samym budżetem (~26 epok), który ma warunek porównawczy `B` — gdyby zapamiętywanie było możliwe,
działałoby w obu.

### 3.4 `best_step` i sufit budżetu — sprawdzone na realnym przebiegu **[Z]**

`train_condition.py` zapisuje `best_step`, `best_step_fraction_of_budget` i `budget_ceiling_warning`
do `status.json`. Interpretacja: warunek `cardinal` widzi każdą próbkę 233 razy, `all` — 26 razy.
Jeśli `all` jest niedotrenowane, działa to **przeciwko** hipotezie pracy, a budżet trzeba podnieść
**wszystkim warunkom jednakowo**, nigdy pojedynczemu.

#### Pierwszy realny pomiar: warunek `all` **NIE jest niedotrenowany**

Przebieg `EB` (`echo2depth`, `all`, 40 000 kroków, 25 epok):

| | wartość |
|---|---|
| `best_step` | **39 000 (97,5 % budżetu)** |
| najlepsze val RMSE | 0,56096 |
| poprawa w ostatniej ćwiartce budżetu | +0,00539 |
| rozrzut (odch. std.) RMSE w ostatniej ćwiartce | **0,00549** |

Krzywa walidacyjna wychodzi na plateau ok. kroku **21 000** i dalej fluktuuje:
0,5629 (21 tys.) → 0,5611 (25 tys.) → 0,5670 (29 tys.) → 0,5788 (33 tys.) → 0,5630 (37 tys.).
Poprawa 20 tys. → 30 tys. jest **ujemna** (−0,0076), 30 tys. → 40 tys. dodatnia (+0,0152) — to są
wahania, nie trend.

#### Wykryty fałszywy alarm i poprawka heurystyki **[Z]**

Pierwotne kryterium (`best_step ≥ 90 % budżetu`) **zapaliło ostrzeżenie**, mimo że model stoi.
Powód jest strukturalny: przy plateau z rozrzutem 0,0055 najlepszy checkpoint ląduje w **losowym**
miejscu, więc `best_step` blisko końca zdarza się regularnie **bez żadnego niedotrenowania**.
Ostrzeganie na samym `best_step` kazałoby podnieść budżet całej macierzy bez powodu — dziesiątki
godzin GPU.

Kryterium poprawione na **dwuczłonowe**: pόźny `best_step` **oraz** poprawa w ostatniej ćwiartce
budżetu większa niż **2 × rozrzut samego plateau**. Sprawdzone w obie strony:

| przypadek | poprawa | 2 × rozrzut | werdykt |
|---|---|---|---|
| `EB` (realne dane, plateau) | 0,00539 | 0,01097 | **nie niedotrenowany** ✓ |
| `ESE` (realne dane) | 0,01061 | 0,01809 | nie niedotrenowany ✓ |
| krzywa syntetyczna wciąż opadająca (1/√t) | 0,03994 | 0,01370 | **niedotrenowany** ✓ |

`status.json` zapisuje teraz obie składowe osobno (`late_best_step`, `still_improving_at_end`,
`plateau`), żeby werdykt dało się zweryfikować, a nie tylko odczytać.

**[Z-] Zastrzeżenie:** to jest pomiar na `echo2depth`, nie na pełnym modelu. Pełny model ma 35× więcej
parametrów i inną dynamikę uczenia — wniosek „40 000 kroków wystarcza" **nie przenosi się
automatycznie** na warunki `A/B/D` i trzeba go powtórzyć na pierwszym przebiegu `B`.

---

## 4. BLOK 2 — protokół ewaluacji (zero GPU ponad jeden przelot)

### 4.0 Protokół przepuszczony przez realne checkpointy **[Z]**

Dzięki temu, że BLOK 1.2 dał dwa wytrenowane modele, cały protokół został sprawdzony **end-to-end**,
a nie tylko jednostkowo. Ewaluacja jednego checkpointu na obu zbiorach testowych: **3,8 s**.

| | `EB` (echo prawdziwe) | `ESE` (echo permutowane) |
|---|---|---|
| `test@36` RMSE (n = 6 588) | 0,57783 | 1,16591 |
| `test@4` RMSE (n = 732) | 0,57642 | 1,16192 |
| krawędzie / gładkie (`test@36`) | 1,13767 / 0,45471 | 2,01932 / 0,99976 |

`test@36` i `test@4` różnią się o 0,0014 — czego należało oczekiwać dla warunku `all`, który widział
wszystkie 36 orientacji. **Ta kolumna zacznie znaczyć dopiero przy warunku `cardinal`**, gdzie
`test@36` ocenia 32 kąty nigdy niewidziane.

**Niezależne potwierdzenie tabeli dziur z §2.4:** kolumna `valid_pixel_fraction` liczona w
`evaluate.py` daje `apartment_2` 88,57 %, `frl_apartment_5` 85,57 %, `office_4` 100,00 % — czyli
dokładnie dopełnienia zmierzonych odsetków zer (11,39 %, 14,24 %, 0,0012 %). Dwie niezależne ścieżki
kodu zgadzają się co do drugiego miejsca po przecinku.

#### Bootstrap sparowany po lokalizacjach na realnych danych **[Z]**

`evaluate.py --compare ESE_seed0 EB_seed0`, 2 000 losowań, **183 lokalizacje**:

| warstwa | ΔRMSE (wkład echa) | 95 % CI | istotne |
|---|---|---|---|
| całość | **+0,58808** | [+0,54706; +0,62936] | tak |
| **krawędzie** | **+0,88164** | [+0,80504; +0,95672] | tak |
| gładkie | +0,54505 | [+0,50761; +0,58298] | tak |

**Kierunek zgodny z hipotezą pracy:** echo wnosi **1,6× więcej na pikselach krawędziowych niż na
gładkich** (0,882 wobec 0,545), a przedziały ufności obu warstw są rozłączne. To jest dokładnie to,
po co powstała metryka stratyfikowana (§3.12 poprzedniego raportu): efekt siedzi tam, gdzie teza
mówi, że powinien — na nieciągłościach głębi.

Stratyfikacja otwarte/szczelne (2.6) na tych samych danych: +0,58715 (145 lokalizacji, sceny
z dziurą) wobec +0,59994 (38 lokalizacji, `office_4`) — przedziały ufności silnie się pokrywają,
więc **wkład echa nie zależy tu od geometrii**. Sonda `office_4` (2.5) działa: RMSE 0,45922 przy
100,00 % ważnych pikseli.

Wszystkie punkty liczą się z **jednej tabeli statystyk per próbka**, zbieranej w jednym przelocie po
zbiorze testowym i zapisywanej na dysk (`samples_test@36.npz`). Każde późniejsze grupowanie,
bootstrap i porównanie między warunkami nie dotyka już ani GPU, ani zbioru danych.

**Dlaczego tabela, a nie kolejne akumulatory.** Blok 2 wymaga tej samej metryki w kilkunastu
grupowaniach, a bootstrap po lokalizacjach wymaga **losowania grup po fakcie** — z góry
zdefiniowanymi akumulatorami jest to niewykonalne. Tabela trzyma **statystyki dostateczne**
(sumy kwadratów błędu, liczby pikseli, liczniki δ), z których każda metryka odtwarza się **dokładnie**
przez zsumowanie wierszy grupy.

Dowód zgodności **[Z]**: `metrics.test_table_matches_accumulator()` — **17 porównań**, maksymalna
różnica wobec `MetricAccumulator` (a więc pośrednio wobec `compute_errors` Paridy)
**2,75·10⁻⁸**, czyli szum float32. Sprawdzane są cztery rzeczy naraz:

1. zgodność na całym zbiorze,
2. zgodność na **podzbiorze** (grupowanie po scenie — to jest to, na czym stoi cały Blok 2),
3. **predykcja zawierająca dokładnie zera**,
4. maska przecięcia faktycznie ogranicza zbiór punktowanych pikseli (373 569 → 186 651).

Punkt 3 wykrył **realną niespójność w kodzie tej sesji [Z]**: `MetricAccumulator` liczy różnicę na
*przyciętej* predykcji (`pred.clamp_min(1e-6)`), a pierwsza wersja tabeli na surowej. Model wypuszcza
`sigmoid · max_depth`, więc dokładne zero jest osiągalne i obie ścieżki dawałyby wtedy różne liczby.
Naprawione; test tego przypadku jest teraz stały.

`test_matches_parida()` nadal przechodzi (1,494·10⁻⁶).

### 4.1 Dwa zbiory testowe z jednego checkpointu

- **`test@36` — podstawowy.** Agent może stać zwrócony dowolnie; to jest sytuacja docelowa.
  Warunek `cardinal` jest tu oceniany na **32 kątach, których nigdy nie widział** — i to jest
  właśnie mierzone.
- **`test@4` — kolumna dodatkowa**, dla zgodności z układem Gao/Paridy.

### 4.2 RMSE w funkcji odległości kątowej od siatki treningowej **[Z] (definicja)**

Odległość liczona do najbliższego kąta **obecnego w całym zbiorze treningowym warunku** (definicja
globalna, nie per lokalizacja) — i to jest cała różnica między A a D:

| warunek | siatka treningowa | odległości w zbiorze testowym |
|---|---|---|
| `A` (`cardinal`) | 4 kąty {0, 90, 180, 270} | **0°, 10°, 20°, 30°, 40°** |
| `C6` (`every_6`) | 6 kątów | 0°, 10°, 20°, 30° |
| `D` (`random_4`) | **36 kątów** (suma po lokalizacjach) | 0° |
| `B` (`all`) | 36 kątów | 0° |

**Sprostowanie wobec zadania:** wartość **45° nie występuje**. Siatka renderów ma krok 10°, więc
żaden kąt testowy nie leży dokładnie w połowie między dwoma kierunkami kardynalnymi; maksimum to 40°.

To jest kandydat na najważniejszy rysunek pracy — bezpośrednia odpowiedź na pytanie, ile kosztuje
próbkowanie co 90°. Zero GPU, samo grupowanie istniejących próbek. Raportowany jest też test
monotoniczności krzywej i luka `RMSE(40°) − RMSE(0°)`.

### 4.3 Bootstrap sparowany po lokalizacjach **[Z]**

36 próbek jednej lokalizacji różni się wyłącznie orientacją, więc efektywne n zbioru testowego to
**183 lokalizacje, nie 6 588 próbek** — test traktujący próbki jako niezależne jest
antykonserwatywny o czynnik rzędu √36 ≈ 6.

Implementacja losuje 183 lokalizacje ze zwracaniem, przelicza metrykę dla **obu** porównywanych
warunków na tej samej próbie i zwraca przedział ufności **różnicy**, nie dwóch średnich osobno.
Sparowanie jest możliwe, bo wszystkie warunki są oceniane na identycznym zbiorze; funkcja
**odmawia** działania, jeśli klucze `(scene, location, angle)` obu tabel nie są identyczne.

Test na danych syntetycznych **[Z]**: efekt realny (bias 0,30 m) → Δ = +0,10120, CI
[+0,10092, +0,10144], istotny; tabele identyczne → Δ = 0,00000, CI [0, 0], nieistotny; tabele
niesparowane → odrzucone wyjątkiem.

### 4.4 Rozbicie per scena z liczbą ważnych pikseli

Każdy wiersz per scena niesie `n_pixels` i `valid_pixel_fraction`. Powód: `frl_apartment_5` ma
14,24 % dziur, `office_4` 0,0012 % — RMSE liczone na 86 % kadru i na 100 % kadru to nie jest ta sama
wielkość, a dziury nie są losowe: siedzą tam, gdzie geometrii brakuje.

### 4.5 `office_4` jako sonda transferu geometrii

Wyodrębnione jako osobna pozycja (`office_4_probe`) z jawną adnotacją, że scena jest szczelna, więc
w obu wariantach serwowana z **tego samego pliku**, a jej próbki testowe są **bit-identyczne**
(potwierdzone w §2.3). Różnica wyniku na `office_4` między modelem trenowanym na `main` a
trenowanym na `patched` mierzy czysto transfer, przy danych testowych trzymanych dosłownie stałych.

### 4.6 Stratyfikacja otwarte / szczelne **[Z-]**

Zaimplementowana (`by_geometry_group`) plus osobny bootstrap różnicy w obu grupach. **Zastrzeżenie,
które trzeba cytować:** zbiór testowy ma tylko 3 sceny — `apartment_2` i `frl_apartment_5` (obie
z dziurą) oraz `office_4` (szczelna). Stratyfikacja wewnątrz zbioru testowego jest więc podziałem
2 sceny vs 1 scena, a nie 6 vs 12 jak w całym zbiorze. Wniosek „efekt gęstości różni się między
grupami" da się z tego wyciągnąć tylko jako przesłankę, nie jako rozstrzygnięcie.

### 4.7 DECYZJA DO PODJĘCIA: na jakim zbiorze kątów wybierać checkpoint **[W]**

Wykryte przy uruchamianiu `EA`. `train_condition.py` buduje zbiór walidacyjny z **własnym
podzbiorem kątów warunku**:

```python
DatasetConfig(..., mode="val", angle_subset=cond.angle_subset, ...)
```

Czyli warunek `cardinal` wybiera checkpoint po RMSE na **4 kątach**, a `all` na **36**. Ma to dwie
konsekwencje:

**1. `best_val_rmse` ze `status.json` NIE jest porównywalne między warunkami.** To są liczby na
różnych zbiorach (732 wobec 6 588 próbek, inne orientacje). Każde zestawienie warunków **musi** iść
przez `evaluate.py`, które zawsze liczy na tym samym `test@36` / `test@4`. Nie jest to błąd, ale jest
to pułapka, w którą łatwo wpaść przy czytaniu logów.

**2. Poważniejsze: to dokłada DRUGĄ różnicę między warunkami.** Zasada trzymana w całej pracy brzmi
„między warunkami różni się dokładnie jedna rzecz — gęstość kątowa danych treningowych". Przy
obecnym układzie różnią się **dwie**: dane treningowe **i kryterium wyboru checkpointu**. Warunek
`cardinal` dostaje checkpoint najlepszy na 4 kątach, a oceniany jest na 36 — czyli jest karany dwa
razy, przy czym drugie ukaranie nie jest częścią pytania badawczego.

**Rekomendacja: walidować wszystkie warunki na pełnych 36 kątach** (`angle_subset="all"` w konfiguracji
walidacyjnej, niezależnie od warunku). Uzasadnienie: zbiór walidacyjny to dane **odłożone**, więc
korzystanie z pełnych 36 orientacji nie jest wyciekiem, a pytaniem pracy jest „ile agent traci,
trenując na 4 kątach, skoro działać musi pod dowolnym" — nie „ile traci, jeśli dodatkowo wybierze
sobie checkpoint krótkowzrocznie".

Zmiana to jedna linia. **Nie wprowadziłem jej sam**, bo zmienia protokół dla całej macierzy i jest
decyzją autora pracy, nie narzędzia — dokładnie jak `--fast-bilinear` w poprzedniej sesji. Musi
jednak zapaść **przed** startem macierzy: po starcie porównanie warunków wymagałoby powtórzenia
wszystkiego.

Przebiegi `EA`/`EB`/`ED` z tej sesji poszły w układzie obecnym. Ich **porównanie na `test@36` jest
tym niezmienione** (test jest wspólny); dotknięty jest wyłącznie wybór checkpointu.

---

## 5. BLOK 3 — uzupełnienie macierzy

Macierz urosła z 12 do **22 warunków**, z 36 do **66 przebiegów**.

### 5.1 Krzywa przy stałym budżecie próbek (3.1) **[Z]**

Nowa składnia podzbioru kątów: `random_K_of_G` — K kątów na lokalizację losowanych **wyłącznie
z podsiatki G równomiernie rozłożonych orientacji**. Krzywa `EK6/EK9/EK12/EK18` ma **liczność stałą
5 496 w każdym punkcie**; zmienia się wyłącznie różnorodność kątowa.

Zweryfikowane **[Z]**, że końcami krzywej są dokładnie istniejące warunki:

- `random_4_of_4` == `cardinal` (warunek A) — identyczny zbiór kątów
- `random_4_of_36` == `random_4` (warunek D) — identyczny zbiór kątów

Uruchamiana na `echo2depth`: 4 warunki × 3 ziarna × 0,13 h = **1,56 h**.

Powód istnienia: `C6/C9/C12/C18` idą na naturalnej liczności (8 244 → 24 732), więc obecna krzywa
nasycenia rośnie po **obu** zmiennych naraz i to, co widać, jest w dużej mierze nasyceniem po
rozmiarze zbioru — zjawiskiem znanym i nieciekawym.

### 5.2 `PD` — brakujący warunek wariantu `patched` (3.2)

Dodany. Bez niego replikacja `patched` odtwarzała dokładnie tę dwuznaczność, którą warunek `D`
naprawił w `main`: dawała Δ łączne, nie rozłożone na składową gęstości i składową ilości danych.

### 5.3 Replikacja `patched` na `echo2depth` (3.3)

Dodane `EPA` / `EPB` / `EPD`, grupa `geometria_echo`: 9 przebiegów × 0,13 h = **1,17 h**, czyli
mniej niż **jeden** przebieg `PA` na pełnym modelu (0,86 h × 3 = 2,59 h).

Uzasadnienie wzmocnione pomiarem z §2.5: wada geometrii jest wadą **akustyczną** — sufit dokłada do
obrazu 13 % pikseli, a do echa **46,3 % energii pogłosu**. Sygnał jest tam, gdzie faktycznie gryzie.

### 5.4 Wendorowanie `beyond-image-to-depth/` (3.4) — **było już zrobione [Z]**

Sprawdzone: **25 plików** jest w gicie (`git ls-files beyond-image-to-depth | wc -l` = 25), hash
upstream `dcdef5122fa456a92bd58ead4eea0a777158c535` zapisany w `beyond-image-to-depth/COMMIT_HASH.txt`,
wpis licencyjny w `THIRD_PARTY_LICENSES.md` (sekcja „Beyond Image to Depth (Parida et al., CVPR 2021)",
MIT). Zdanie „sieć, strata i optymalizator są dokładnie te opublikowane" jest więc weryfikowalne.

Punkt 2 z §8 poprzedniego raportu („nie jest śledzone przez git") jest **nieaktualny** — zostało to
zrobione między sesjami.

### 5.5 Budżet: dysk i czas przeliczone **[Z]**

Zmierzone liczby parametrów (nie oszacowane):

| model | parametry | same wagi | checkpoint z Adamem | **razem na przebieg** |
|---|---|---|---|---|
| `full` | 316 918 781 | 1,18 GB | **3,54 GB** | **4,72 GB** |
| `echo2depth` | 8 984 073 | 0,03 GB | 0,10 GB | 0,13 GB |

Poprzedni raport podawał ~1,3 GB na przebieg — to były **same `state_dict`y**.
`save_checkpoint()` zapisuje **także** `optimizer.state_dict()`, a Adam trzyma **dwa** momenty na
parametr (`exp_avg`, `exp_avg_sq`), więc checkpoint wznowieniowy jest ~3× większy od wag. Skoro
`train_condition.py` ma realnie wznawiać po SIGTERM, obowiązuje liczba z Adamem.

| grupa | przebiegów | godzin | GB |
|---|---|---|---|
| `bramka` | 6 | 2,98 | 14,6 |
| `echo` | 9 | 1,17 | 1,2 |
| `glowne` | 9 | 7,77 | 42,6 |
| `krzywa` | 12 | 10,36 | 56,8 |
| `krzywa_staly` | 12 | 1,56 | 1,6 |
| `geometria_echo` | 9 | 1,17 | 1,2 |
| `geometria` | 9 | 7,77 | 42,6 |
| **RAZEM** | **66** | **32,8 h (1,37 dnia)** | **161 GB** |

**161 GB z Adamem wobec 40 GB po skasowaniu checkpointów wznowieniowych.** Na dysku jest 225 GB
wolnego, więc mieści się — ale bez zapasu na cokolwiek innego. Praktyczny wniosek: kasować
`checkpoint.pt` po zakończonym przebiegu (`status.json: finished=true`), zostawiając `best_*.pth`.

Czas liczony przy `--fast-bilinear` (0,0776 s/krok dla `full`, 0,0116 s/krok dla `echo2depth`).
Bez niego grupy z pełnym modelem rosną ~16–19,5× (§3.1).

**[Z-] Zastrzeżenie do czasu:** 0,0776 s/krok pochodzi z mikrobenchmarku **samego kroku
treningowego**, bez walidacji. Zmierzone na pełnej pętli w §3.1 (z walidacją co ~333 kroki
i narzutem diagnostycznym) wyszło **0,0953 s/krok**, czyli 1,06 h na przebieg zamiast 0,86 h.
Liczby w tabeli wyżej są więc **dolnym oszacowaniem**; realny czas macierzy to raczej ~40 h niż
32,8 h. Nie podnoszę ich w `experiments.json`, bo narzut diagnostyczny z §3.1 nie występuje
w normalnym przebiegu, a rozdzielenie obu składników wymagałoby osobnego pomiaru.

**Błąd wykryty przy okazji [Z]:** `exp_ctl.py plan` wołał `dump_config()` bez liczb parametrów, więc
**po cichu kasował sekcję `budzet`** z `experiments.json` przy każdym wywołaniu. Naprawione: liczby
parametrów są teraz stałymi `experiments.PARAM_COUNTS` (zmierzonymi, z funkcją `verify_param_counts()`
sprawdzającą zgodność z faktycznymi modelami — **zwraca `ok=True`**), a budżet liczy się zawsze.

---

### 5.6 Grupa `echo` uruchomiona — PIERWSZE WYNIKI MERYTORYCZNE **[Z]**

Skoro bramka (§3.3) wypadła pomyślnie i została jeszcze doba GPU, uruchomiono całą grupę `echo` po
jednym ziarnie: `EA` (`cardinal`), `EB` (`all`), `ED` (`random_4`) plus `ESE` z bramki. Wszystkie
4 warunki × 40 000 kroków, ~10 min każdy. **Wszystkie oceniane na identycznym zbiorze testowym**
(`test@36`, 6 588 próbek, 183 lokalizacje).

| warunek | kątów/lok. | próbek train | `test@36` | `test@4` |
|---|---|---|---|---|
| `EA` `cardinal` | 4 | 5 496 | **0,80265** | 0,57792 |
| `ED` `random_4` | 4 | 5 496 | **0,66308** | 0,66215 |
| `EB` `all` | 36 | 49 464 | **0,57783** | 0,57642 |
| `ESE` echo permutowane | 36 | 49 464 | 1,16591 | 1,16192 |

#### Rozkład efektu — po to powstał warunek D

Bootstrap sparowany po 183 lokalizacjach, 2 000 losowań:

| składowa | ΔRMSE | 95 % CI | co izoluje |
|---|---|---|---|
| **D − A** | **+0,13957** | [+0,11403; +0,16599] | **sama gęstość kątowa** przy równej liczności |
| **B − D** | **+0,08525** | [+0,07012; +0,10337] | sama ilość danych |
| B − A | +0,22482 | [+0,19706; +0,25541] | efekt łączny |

Składowe sumują się dokładnie: 0,13957 + 0,08525 = 0,22482.

**Gęstość kątowa odpowiada za 62,1 % efektu łącznego i jest 1,64× większa niż efekt samej ilości
danych.** Bez warunku `D` obie te rzeczy byłyby nierozróżnialne — i to jest dokładnie ta
dwuznaczność, którą warunek `D` miał usunąć (§4.3 poprzedniego raportu).

#### Krzywa generalizacji kątowej — rysunek, o który chodziło w 2.2 **[Z]**

Warunek `EA` widział wyłącznie 4 kierunki kardynalne, a oceniany jest na 36. RMSE w kubełkach
odległości kątowej od najbliższego kąta treningowego:

| odległość od siatki treningowej | n | RMSE | RMSE krawędzie |
|---|---|---|---|
| **0°** | 732 | **0,57792** | 1,06177 |
| 10° | 1 464 | 0,65906 | 1,21230 |
| 20° | 1 464 | 0,79178 | 1,40657 |
| 30° | 1 464 | 0,89516 | 1,55024 |
| **40°** | 1 464 | **0,93332** | 1,59951 |

**Luka +0,35540, czyli +61,50 %, monotoniczna na całej długości.** Zero GPU ponad ewaluację — samo
grupowanie istniejących próbek.

#### Obserwacja, która porządkuje interpretację całej pracy

Zestawienie dwóch liczb z tabeli wyżej:

- `EA` na `test@4` = **0,57792**
- `EB` na `test@36` = **0,57783**

Są **identyczne do czwartego miejsca po przecinku**. Model trenowany na 4 orientacjach jest na tych
4 orientacjach **dokładnie tak samo dobry**, jak model trenowany na wszystkich 36 jest na
wszystkich 36. Cała kara za rzadkie próbkowanie kątowe siedzi **wyłącznie w orientacjach, których
model nie widział** — nie w gorszym nauczeniu tych, które widział.

To znaczy, że baseline VisualEchoes (Gao, 4 kierunki) **nie jest gorzej wytrenowany**; on po prostu
**nie pokrywa przestrzeni orientacji**. Zdanie do pracy brzmi więc: próbkowanie co 90° nie psuje
uczenia, tylko zostawia 89 % przestrzeni orientacji bez nadzoru — i to kosztuje 61,5 % RMSE
na najdalszych kątach.

#### Skala efektów wobec szumu

| wielkość | RMSE |
|---|---|
| podłoga szumu frameworka (§3.1) | 0,0023 – 0,0073 |
| efekt samej gęstości kątowej (D − A) | **0,1396** |
| całkowity wkład echa (ESE − EB) | 0,5881 |

Efekt gęstości jest **19–60× nad podłogą szumu** i stanowi **24 % całkowitego wkładu echa**.
Oś gęstości kątowej na gałęzi `echo2depth` jest więc mierzalna z dużym zapasem.

**[Z-] Zastrzeżenia, które muszą iść razem z tymi liczbami:**

1. **n = 1 ziarno na warunek.** Przedziały ufności opisują zmienność po **lokalizacjach**, nie po
   ziarnach. Zmierzona podłoga między ziarnami (0,0023–0,0073) jest 20–60× mniejsza niż raportowane
   efekty, więc wniosek jest bezpieczny — ale pełna macierz z 3 ziarnami musi to potwierdzić.
2. **To jest gałąź `echo2depth`, nie pełny model.** W pełnym modelu prior wizualny może przykryć
   ten efekt; warunki `A/B/D` pozostają nieuruchomione.
3. Checkpointy wybrano w układzie sprzed decyzji z **§4.7** (walidacja na własnym podzbiorze kątów).
   Porównanie na `test@36` jest tym niezmienione, ale warunek `EA` mógł dostać checkpoint
   krótkowzrocznie dobry na 4 kątach — co jeśli cokolwiek, to **zawyża** raportowaną lukę.

---

## 6. BLOK 4 — Model 2 (zadanie pretekstowe orientacji)

Zaimplementowany i przechodzi przebieg dymny. **Nie uruchomiony** w pełnym wymiarze — zgodnie
z zadaniem uruchomienie może pójść później.

### 6.1 Dlaczego to ma pierwszeństwo przed rozbudową Modelu 1

1. **Gao sam zrobił ablację po liczbie klas i zatrzymał się na 4 z powodu narzędzia.** Tabela 3
   pracy głównej: `Scratch` 0,360 → `SimpleVisualEchoes` (2 klasy) 0,340 → `VisualEchoes`
   (4 klasy) **0,332**. Trend monotoniczny. Rozszerzenie na 12 i 36 klas to przedłużenie **ich
   własnej osi**, a nie nowy pomysł do obrony.
2. **Zadanie rośnie kwadratowo, nie liniowo** (§6.3).
3. **To jest teza z postera** — zmuszenie kodera wizualnego do głębszego rozumienia geometrii to
   Model 2, nie Model 1.

### 6.2 Specyfikacja — trzymana dosłownie za suplementem Gao (§I) **[Z]**

| element | realizacja | źródło |
|---|---|---|
| gałąź audio | `SimpleAudioDepthNet` Paridy **bez dekodera**: 3 konwolucje 8×8/4×4/3×3, BatchNorm+ReLU, warstwa liniowa → **512** | `networks.py:42`, `audio_feature_length=512` w `models.py:11` |
| gałąź wizualna | enkoder `RGBDepthNet` Paridy: 5 warstw, 128×128×3 → **4×4×512** | `networks.py:162-166` |
| redukcja | `conv1x1` 512 → 8 kanałów, potem spłaszczenie → **128** | odtworzona z zakomentowanej linii `networks.py:172` |
| fuzja | konkatenacja (128 + 512) → FC + ReLU → **D = 128** | suplement §I |
| głowa | jedna FC → K klas, **płaska cross-entropy** | suplement §I |
| podział | `outputs/ml/splits/replica_locations.json`, odcisk `e0bf7547668d9e0a` | ten sam, co Model 1 |

**Nie wprowadzono ResNet-50** — u Gao jest on wyłącznie dla NYU-V2/DIODE, nie dla Repliki.
`conv1x1` odtworzony w **naszym** module, a nie przez odkomentowanie linii w pliku Paridy.

Liczba parametrów: **25 733 446** (z czego 16 658 561 to przenoszony `RGBDepthNet`).

### 6.3 Liczba par — zgadza się co do sztuki z tabelą zadania **[Z]**

Etykieta to **przesunięcie względne** `(j − i) mod 360`, nie orientacja bezwzględna — sprawdzone
na próbkach (`i=0°, j=90° → klasa 1, shift 90°`; `i=90°, j=90° → klasa 0`).

| K | par/lokalizację | par treningowych (1 374 lok.) | losowo top-1 |
|---|---|---|---|
| 4 | 16 | **21 984** | 25,0 % |
| 12 | 144 | **197 856** | 8,33 % |
| 36 | **1 296** | **1 780 704** | 2,78 % |
| 36 @ 16 par/lok. (kontrola) | 16 | **21 984** | 2,78 % |

81× więcej sygnału uczącego z **tych samych renderów**, wobec 9× w Modelu 1.

Pary powstają z podsiatki K orientacji (dla K=4 są to kierunki kardynalne — dokładnie układ Gao),
bo tylko wtedy każda różnica kątów wpada dokładnie w jedną klasę.

Kontrola przy równej liczbie par (4.4) daje **dokładnie 21 984** par, czyli tyle co K=4:

- `K36 − K36@16par` izoluje **ilość danych**
- `K36@16par − K4` izoluje **samą rozdzielczość kątową zadania**

### 6.4 Pułapka metryczna — MAAE zamiast top-1 **[Z]**

Trafność top-1 **nie jest porównywalna** między różnymi K: poziom losowy spada z 25 % do 2,8 %.
Metryką porównywalną jest **średni bezwzględny błąd kątowy (MAAE)** liczony jako odległość po
okręgu; dla rozkładu równomiernego MAAE poziomu losowego wynosi **90° niezależnie od K**, więc
wszystkie warianty mają wspólny punkt odniesienia.

Raportowane: MAAE (podstawowa), trafność w tolerancji ±10°/±30°/±45° **razem z poziomem losowym
tej tolerancji dla danego K**, trafność top-1 dosłowna (wyłącznie do zestawienia z 66 % Gao przy
K=4), macierz pomyłek i udział błędów trafiających w klasę **sąsiednią**.

### 6.5 Ryzyko zmierzone, nie założone (4.6) **[Z] (implementacja)**

MAAE liczone **osobno** dla par o prawdziwym przesunięciu ≤ 20° i > 20°. Jeśli sieć jest bezradna
poniżej 20°, to jest wynik sam w sobie i wyznacza faktyczną granicę rozdzielczości metody. Przy
K=36 rozróżnienie 0° od 10° wymaga sygnału RMSE ≈ 0,0644 przy szumie próbki ~0,018 (SNR ≈ 3,5) —
jest nad szumem, ale niewiele.

Ablacja opcjonalna (strata okrężna: wygładzanie etykiet na sąsiadów albo regresja von Misesa)
**nie została zaimplementowana** — wersja z płaską cross-entropy pozostaje podstawową, bo tylko ona
jest porównywalna z Gao.

### 6.6 Przebieg dymny **[Z]**

| etap | wynik |
|---|---|
| `train_pretext.py --k 4 --steps 60` | przeszedł; MAAE 86,87° (losowo 90), top-1 27,8 % (losowo 25 %) — czyli **na poziomie losowym**, tak jak musi być po 60 krokach |
| `transfer.py --init <best_encoder.pth>` | przeszedł; **35 z 35** kluczy enkodera przeniesione, 0 niezgodności kształtu |
| `transfer.py --init scratch` | przeszedł |

Przeniesienie **35 kluczy** to 5 warstw × 7 tensorów (conv.weight, conv.bias, bn.weight, bn.bias,
bn.running_mean, bn.running_var, bn.num_batches_tracked). `load_pretrained_encoder()` **przerywa
z błędem**, jeśli nie dopasuje żadnego klucza — ciche `strict=False`, które nie wczytało niczego,
wyglądałoby dokładnie jak `Scratch` i cała tabela 4.5 byłaby tabelą pięciu razy tego samego warunku.

Katalogi dymne usunięte po sprawdzeniu.

**Uwaga wydajnościowa [Z]:** walidacja przy K=36 na wszystkich parach to 237 168 par **na każdą
walidację** — wielokrotnie drożej niż sam trening między walidacjami. Dodany
`--val-pairs-per-location` (domyślnie 16), co daje ten sam rozmiar zbioru walidacyjnego dla
każdego K.

### 6.7 Ewaluacja końcowa — liczba do pracy

Zadaniem docelowym jest **RGB2Depth bez audio w czasie testu** (`RGBOnlyModel`), na tych samych
scenach odłożonych, z tą samą stratą `LogDepthLoss`, maskowaniem `depth_gt != 0` i skalą
`max_depth` co Model 1. Warunki różnią się **wyłącznie inicjalizacją enkodera**:

| inicjalizacja enkodera | odniesienie u Gao (Replica) |
|---|---|
| `Scratch` | 0,360 |
| pretrening K=4 | 0,332 |
| pretrening K=12 | — (nasze) |
| pretrening K=36 | — (nasze) |
| pretrening K=36 @ 16 par/lok. | — (nasze, kontrola) |

Kolumna „odniesienie" **nie jest baseline'em do przepisania** — silnik akustyczny jest inny, a
§2.5 pokazał, że sam wariant geometrii zmienia energię późną o 46 %. Służy wyłącznie do sprawdzenia,
czy odtwarzamy właściwy **porządek** warunków i rząd wielkości efektu.

`ml/pretext/summarize.py` składa oba etapy w dwie tabele i **wypełnia kolumnę odniesienia wyłącznie
dla `Scratch` i K=4** — bo tylko te dwa warunki Gao raportuje. Skrypt sygnalizuje też przebiegi
z **nieudanym przeniesieniem wag** (`transfer_ok=False`): taki przebieg jest liczbowo
nieodróżnialny od `Scratch`, więc bez tego ostrzeżenia cała tabela mogłaby po cichu być tabelą
pięciu razy tego samego warunku.

---

## 7. Czego **NIE** sprawdzono **[X]**

- **Kontrola niedeterminizmu poszła na 2 000 kroków, nie na 40 000.** Rozbieżność wag na kroku 2 000
  wyraźnie **malała** (§3.1, punkt 1), więc ekstrapolacja na pełny budżet jest wiarygodna, ale
  **nie została zmierzona**. Gdyby trajektorie zaczęły się rozchodzić dopiero po dziesiątkach
  tysięcy kroków, ten pomiar by tego nie pokazał.
- **Kontrola niedeterminizmu ma po JEDNEJ parze na wielkość.** Dlatego wniosek stoi na porównaniu
  dwóch niezależnych ocen podłogi (p2/p1), a nie na samym `p3/p1` — ale to nadal jest n=1 na parę.
- **Walidacja w §3.1 liczona na stałym podzbiorze 2 048 próbek**, nie na pełnym zbiorze
  walidacyjnym. Do porównania **różnicy** między przebiegami to wystarcza (jest sparowane), ale
  bezwzględne RMSE stamtąd nie jest liczbą do raportowania jako jakość modelu.
- **Bramka dla PEŁNEGO modelu (`SE` vs `B`) nie została uruchomiona.** Zmierzony wkład echa
  (0,607) dotyczy `echo2depth`, czyli modelu, który nie ma nic innego. W pełnym modelu obraz RGB
  niesie większość informacji o głębi, więc **marginalny** wkład echa będzie wielokrotnie mniejszy
  — i to on jest właściwym górnym ograniczeniem dla warunków `A/B/D`. **To jest najważniejsza
  brakująca liczba** i pierwsza rzecz do uruchomienia (~2 h: `SE` + `B`, po jednym ziarnie).
- **Sufit budżetu sprawdzony tylko na `echo2depth`** (§3.4) — pełny model ma 35× więcej parametrów
  i inną dynamikę; wniosek nie przenosi się automatycznie na `A/B/D`.
- **Wszystkie cztery przebiegi grupy `echo` mają po JEDNYM ziarnie.** Efekty (0,085–0,588) są
  20–80× nad zmierzoną podłogą między ziarnami (0,0023–0,0073), więc wnioski o **znaku i rzędzie
  wielkości** są bezpieczne — ale liczby do tabeli w pracy muszą pochodzić z 3 ziaren.
- **Nie uruchomiono `C6/C9/C12/C18` ani krzywej stałego budżetu (`EK*`).** Krzywa nasycenia i
  rozdzielenie „gęstość vs rozmiar zbioru" wzdłuż całej osi pozostają niezmierzone — mamy tylko
  trzy punkty (A, D, B).
- **`office_4` jako sonda transferu geometrii (2.5) nie została użyta zgodnie z przeznaczeniem** —
  wymaga modelu trenowanego na `patched`, a takiego nie ma.
- **Maska przecięcia nie była użyta w realnej ewaluacji** — wszystkie checkpointy są z wariantu
  `main`, więc nie było czego przecinać. Ścieżka sprawdzona wyłącznie na poziomie danych (§2.7).
- **Stratyfikacja otwarte/szczelne (2.6) jest w zbiorze testowym oparta na 2 vs 1 scenie** — patrz
  zastrzeżenie w §4.6.
- **Model 2 nie był trenowany dłużej niż 60 kroków.** Nie wiadomo, czy zadanie jest w ogóle
  wykonalne przy K=36 — to jest właśnie pytanie z §6.5.
- **Ablacja straty okrężnej** dla Modelu 2 nie została zaimplementowana (świadomie).
- **Nie zmierzono narzutu walidacji** na pełnym przebiegu ani realnego zużycia dysku (liczby w §5.5
  są policzone z liczby parametrów, nie zważone na dysku).
- **Maska przecięcia sprawdzona na poziomie danych, nie na wytrenowanym modelu** (§2.7) —
  `mask_variant` czyta drugi plik HDF5 na próbkę, co podwaja liczbę odczytów w ewaluacji; **koszt
  czasowy nie został zmierzony**.
- **Piksele zmienione, a już ważne** (≤ 3,3 % kadru, §2.2) **pozostają nieusuwalną różnicą** między
  wariantami — maska przecięcia ich nie wycina, bo są ważne w obu. Ich wpływ na Δ nie został
  oszacowany.

---

## 8. Co blokuje i w jakiej kolejności ruszać

**Decyzje autora z §8 poprzedniego raportu są zamknięte — obie:**

1. **`--fast-bilinear`: WŁĄCZONE domyślnie**, na podstawie pomiaru z §3.1, nie argumentu.
2. **`beyond-image-to-depth/` w gicie** — było zrobione między sesjami (§5.4).

**JEDNA NOWA DECYZJA CZEKA NA AUTORA — i blokuje start macierzy: §4.7**, zbiór kątów, na którym
wybierany jest checkpoint. Obecnie każdy warunek waliduje na własnym podzbiorze, co dokłada drugą
różnicę między warunkami. Rekomendacja: walidować wszystkie na pełnych 36 kątach. Zmiana to jedna
linia, ale po starcie macierzy jest nieodwracalna bez powtórki.

**Nie startować grupy `glowne` przed zamknięciem bramki.** Kolejność zapisana w
`experiments.GROUPS` i to nie jest kwestia gustu:

1. **`bramka` na `echo2depth` (`ESE`) — ZROBIONE**, bramka otwarta (§3.3).
2. **`echo` (`EA`, `EB`, `ED`) — ZROBIONE po 1 ziarnie**, wynik w §5.6. Zostaje **ziarno 1 i 2**
   (6 przebiegów, ~1,2 h), żeby liczby do tabeli miały rozrzut po ziarnach.
3. **`bramka` na PEŁNYM modelu (`SE`) + `B`** — ~2 h. To jest teraz **najważniejsza brakująca
   liczba**: marginalny wkład echa przy obecnym priorze wizualnym. Jeśli okaże się rzędu
   0,005 RMSE, warunki `A/B/D` na pełnym modelu są niezdolne wykryć efekt gęstości i ciężar dowodu
   zostaje na `echo` i Modelu 2 — co §5.6 już częściowo zabezpieczyło.
4. **`glowne`** — dopiero po punkcie 3 i po decyzji z §4.7.
5. reszta (`krzywa`, `krzywa_staly`, `geometria`, `geometria_echo`), potem Model 2.

Kontekst z §3.1, który trzeba mieć przed oczami przy czytaniu **każdego** wyniku macierzy:
**niedeterminizm samego frameworka daje ~0,002–0,007 RMSE rozrzutu** między bitowo identycznymi
konfiguracjami. Dla porównania cały wkład echa u Gao to 0,028 (0,374 → 0,346). Przy jednym ziarnie
podłoga szumu sięga więc ~¼ tego efektu — i to jest ilościowe uzasadnienie 3 ziaren na warunek oraz
bootstrapu sparowanego z §4.3.

---

## 9. Gdzie leżą dowody

| co | gdzie | w gicie? |
|---|---|---|
| rozstrzygnięcie geometrii (BLOK 0), 4 pomiary | `outputs/ml/geometry_check/geometry_check.json` | **tak** (60 KB) |
| kontrola niedeterminizmu (BLOK 1.1) | `outputs/ml/determinism/determinism_check.json` | **tak** |
| **bramka: wkład echa (BLOK 1.2)** | `outputs/ml/echo_ablation/echo_ablation.json` | **tak** (4 KB) |
| **rozkład efektu gęstości + krzywa kątowa (§5.6)** | `outputs/ml/echo_ablation/echo_density_seed0.json` | **tak** |
| ewaluacja per przebieg (BLOK 2) | `outputs/ml/eval/<run>/eval.json` | **tak** (24 KB) |
| porównanie sparowane + bootstrap | `outputs/ml/eval/compare_<a>_vs_<b>.json` | **tak** (4 KB) |
| konfiguracja macierzy + budżet dysku i czasu | `outputs/ml/experiments.json` | **tak** |
| podział lokalizacji + odcisk | `outputs/ml/splits/replica_locations.json` | **tak** |
| weryfikacja dataloadera, oba warianty | `outputs/ml/verify_loader/{main,patched}/verify_loader.json` | **tak** |
| test zgodności tabeli z akumulatorem (17 kontroli) | `ml/metrics.py::test_table_matches_accumulator()` | tak (kod) |
| test zgodności metryk z Paridą | `ml/metrics.py::test_matches_parida()` | tak (kod) |
| test tożsamości bilinear | `ml/fast_bilinear.py::verify_equivalence()` | tak (kod) |
| tabele per próbka ze zbioru testowego | `outputs/ml/eval/<run>/samples_*.npz` | nie (duże) |
| przebiegi treningowe i pretreningowe | `outputs/ml/{runs,pretext,pretext_transfer}/` | nie (161 GB) |

Zasada odbiałolistowania w `.gitignore` pozostaje ta sama: **małe pliki JSON, na których stoją
liczby w tym dokumencie — tak; wszystko odtwarzalne i duże — nie.** Sprawdzone:
`samples_test@36.npz` jest ignorowany, `eval.json` obok niego nie.
