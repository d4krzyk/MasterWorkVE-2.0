# Raport prac 2026-07-26 → 2026-07-29

Dokument do pracy magisterskiej. Każde twierdzenie ma przypisany **status wiarygodności** — co
zostało zmierzone, jak, i czego pomiar **nie** obejmuje. Sekcja 6 wymienia błędy popełnione
i skorygowane po drodze; sekcja 7 mówi, gdzie leżą dowody i czego brakuje w repozytorium.

## Legenda statusów

| status | znaczenie |
|---|---|
| **[Z]** | **Zmierzone** — istnieje skrypt, surowe wyjście, liczby w dokumencie. Nadaje się do pracy jako wynik. |
| **[Z-]** | Zmierzone, ale na **małej próbce** albo z zastrzeżeniem, które trzeba zacytować razem z liczbą. |
| **[W]** | **Wywnioskowane** z kodu źródłowego / dokumentacji, nie z pomiaru. Wymaga sprawdzenia, jeśli ma trafić do pracy jako twierdzenie. |
| **[P]** | **Przejęte** z wcześniejszej fazy projektu (przed 2026-07-28). Nie weryfikowane w tych sesjach. |
| **[X]** | **Nie sprawdzone** — wymieniam, żeby nie powstało wrażenie, że zostało. |

---

## 1. Co było przed tymi sesjami (żeby nie przypisać sobie cudzej pracy)

Stan repozytorium na commit `bd5f53bf` (2026-07-26): generator jeszcze nie istniał, ale faza
charakteryzacji szumu była zamknięta. `diagnostics_report.json` zawierał **21 kluczy**:
`p0`, `e1`, `e1_extended`, `e1_checkpoint_boundary*` (4), `e2_rays_vs_renders`, `e2_ray_bias`,
`e2_bias_orientation`, `e2_thread_*` (3), `e3_averaging_domain`, `e4_ir_length`,
`listener_height`, `materials_verify`, `signal_noise_recheck`, `gpu_memory_scale`,
`noise_floor_scenes`, `noise_floor_orientation`.

**[P]** Wszystkie liczby pochodzące z tych eksperymentów — determinizm RNG, bezpieczeństwo
checkpointu (Wilcoxon p=0.949), bias kątowy 2.1 %, budżet wątków, domena uśredniania `mag`,
uzasadnienie `float16`, wysokość słuchacza 1.25 m, sygnał 10° = 0.0644 — **nie były
weryfikowane w sesjach 07-28/29**. Cytuję je, ale ich nie sprawdzałem. Jeśli mają trafić do
pracy jako wyniki, ich wiarygodność opiera się na wcześniejszej pracy, nie na tych sesjach.

W tych sesjach dodałem do raportu **dokładnie jeden klucz**: `noise_floor_remaining`
(zweryfikowane programowo: 0 kluczy usuniętych, 0 zmodyfikowanych, 301 linii dodanych).

---

## 2. Ustalenia twarde — nadają się do pracy jako wyniki

### 2.1 Wysokość agenta `y` pochodzi z `graph.pkl`, nie z `pathfinder.snap_point()` **[Z]**

Najważniejsza zmiana merytoryczna całego okresu. Specyfikacja podawała `snap_point()`, co było
kalką z kodu diagnostyk, a nie świadomą decyzją.

**Pomiar rozstrzygający:** 16 lokalizacji × 4 kąty = 64 porównania piksel-po-pikselu renderów
z `scene_observations_128.pkl` (scena `office_1`).

| wariant `y` | RGB RMSE śr./max | % pikseli bit-identycznych | depth RMSE śr. |
|---|---|---|---|
| `graph.pkl` | 0.0125 / 0.0214 | **99.982 %** | 9·10⁻⁶ m |
| `snap_point` | 50.05 / 75.28 | 36.02 % | 0.150 m |

**Dlaczego test ma moc rozstrzygającą:** rendering wizualny w habitat-sim jest deterministyczny
(w odróżnieniu od audio), więc każda niezerowa różnica oznacza rozbieżność konfiguracji.
Wynik replikuje wcześniejszy test z `PKL_FORMAT.md` (0.0077 / 99.99 % na `room_0`) na scenie,
której tam nie testowano.

**Przyczyna:** `snap_point()` zwraca wysokość *powierzchni navmesha*, nie podłogi. Navmesh
Repliki nie ma zapisanych `NavMeshSettings`, więc recast odtwarza go z domyślną kwantyzacją.
Zmierzone na **wszystkich 1740 lokalizacjach**: mediana różnicy 0.2125 m, maksimum 0.4901 m,
dotyczy 1738/1740. Nie zależy od punktu startowego — podanie `y` z grafu jako `y_guess` daje
wynik co do bitu identyczny.

**Waga błędu:** eksperyment `listener_height` (praca wcześniejsza, **[P]**) zmierzył, że
przesunięcie słuchacza o 0.25 m zmienia echo 1.02× tak mocno jak pełny obrót o 10° — czyli
mediana rozbieżności była rzędu całego mierzonego efektu.

**Sformułowanie do pracy:** to jest dobry przykład błędu, który przeszedłby niezauważony,
gdyby nie test spójności z istniejącym zbiorem referencyjnym. Warto to opisać jako element
metodologii, nie jako wpadkę.

### 2.2 Symulacja akustyczna wykonywała się dwa razy na render **[Z]**

`phase3_echolocation()` wywoływało `runSimulation()` jawnie, a `sim.get_sensor_observations()`
dla sensora typu AUDIO wchodzi w `Sensor._get_audio_observation()`
(`habitat-sim/src_python/habitat_sim/simulator.py:763-777`), które wywołuje ją **ponownie** —
i to jej wynik trafia do obserwacji. Pierwsza symulacja była liczona i wyrzucana.

**Pomiar** (mediana z 25 renderów, `office_1`, 500 promieni, 1 wątek, materiały włączone):

| wariant | czas / render |
|---|---|
| 2 symulacje + RGB + depth (stan zastany) | 274.1 ms |
| 2 symulacje, bez wizualnych | 277.4 ms |
| **1 symulacja, bez wizualnych** | **139.3 ms** |
| same RGB + depth | 0.2 ms |

Dwa wnioski: **RGB i depth są w tym potoku darmowe** (0.2 ms wobec 139 ms — 700×), więc
rozdzielanie obserwacji wizualnych od audio nic nie daje; oraz usunięcie zdublowanej symulacji
skraca generację **dokładnie dwukrotnie**.

**Walidacja równoważności** (bo zmiana przesuwa sekwencję RNG): 2 pozycje skrajne
(`office_1/33` najgłośniejsza, `frl_apartment_5/186` najcichsza), kąty 0° i 10°, M = 40
renderów na kąt, ścieżki nieprzeplatane.

| pozycja | metryka | podwójna → pojedyncza | wynik |
|---|---|---|---|
| `office_1/33` | energia | 0.220816 → 0.221185 (+0.167 %) | **0.72 SE**, Mann-Whitney p = 0.462 |
| | `sigma_1` | 0.10048 → 0.10181 (+1.3 %) | **0.51 SE** |
| | sygnał 10° | 0.06673 → 0.06612 (−0.9 %) | w zakresie 0.0639–0.0662 |
| `frl_apartment_5/186` | energia | 0.091763 → 0.091768 (+0.005 %) | **0.02 SE**, p = 0.877 |
| | `sigma_1` | 0.03375 → 0.03476 (+3.0 %) | **0.75 SE** |
| | sygnał 10° | 0.06431 → 0.06451 (+0.3 %) | w zakresie 0.0639–0.0662 |

Wszystkie porównania poniżej 2 SE. Werdykt: **równoważne**.

*Uwaga o odtwarzalności:* wartości sygnału 10° zależą od tego, który estymator wchodzi do
dekompozycji odszumiającej. Powyższe pochodzą z `measurements/audio_path_analyse.py`
(estymator wariancyjny). Wcześniejsza wersja analizy używała połówkowego i dawała
0.06650 → 0.06538 oraz 0.06424 → 0.06426 — różnica rzędu 0.3 %, wniosek identyczny.

**Potwierdzenie na pełnej scenie:** `office_1` przegenerowana nową ścieżką. Rozkłady `N`
(Mann-Whitney p=0.850, Kołmogorow-Smirnow p=1.000), `sigma_1` (p=0.955) i `snr_final`
(p=0.263) nieodróżnialne, przy czasie 55.8 → 28.3 min (**1.97×**).

**Zastrzeżenie [Z-]:** czułość tego pomiaru to ~5–8 % dla `sigma_1`. Różnicy mniejszej niż
5 % nie wykrywa. Nawet gdyby istniała, byłaby kompensowana automatycznie, bo reguła
adaptacyjna mierzy `sigma_1` na miejscu.

### 2.3 Pełny census sondy — rozkład `N` dla wszystkich 1740 lokalizacji **[Z]**

Zamiast dalej ekstrapolować z próbki, policzono rzeczywisty rozkład. Sonda 8-renderowa jest
i tak pierwszym krokiem generatora, więc dała się wykonać osobno: **14 300 renderów, 35 min**.

`sigma_1`: mediana 0.05830, zakres **0.02530–0.13451**. `N_raw`: mediana 11, średnia 11.80,
zakres **2–54**.

| `N_raw` | lokalizacji | udział |
|---|---|---|
| 1–5 | 319 | 18.33 % |
| 6–8 | 333 | 19.14 % |
| 9–12 | 436 | 25.06 % |
| 13–16 | 314 | 18.05 % |
| 17–20 | 149 | 8.56 % |
| 21–24 | 85 | 4.89 % |
| 25–30 | 69 | 3.97 % |
| 31–40 | 28 | 1.61 % |
| 41–48 | 6 | 0.34 % |
| 49–64 | 1 | 0.06 % |
| > 64 | **0** | 0.00 % |

**To jest najmocniejszy wynik tych sesji** — nie próbka, nie model, tylko pomiar całej
populacji, na której będzie generowany zbiór.

### 2.4 Ekstrapolacja z próbki 52 pozycji była błędna — i wiadomo dlaczego **[Z]**

Przed census zmierzono 52 pozycje (2–3 na scenę, dobierane po ustalonych ułamkach 0.20 i 0.75
listy lokalizacji). Ekstrapolacja przewidywała **zero** przekroczeń `N_MAX = 40`.
Census znalazł **7 lokalizacji (0.402 %)**.

Powód jest pouczający i wart opisania w pracy: gorące miejsce akustyczne w `apartment_0` leży
przy `loc_id` 285–310, czyli w okolicy ułamka 0.9 listy — poza oboma próbkowanymi punktami.
To dokładnie ten rodzaj błędu, przed którym zastrzegano w dokumencie („próbka mała i **nie
dobrana losowo**"), tylko że tym razem się zmaterializował.

**Wniosek metodologiczny:** przy zbiorze, w którym pojedyncza lokalizacja generuje 36 próbek,
próbkowanie systematyczne (co k-ty punkt) potrafi ominąć skupisko. Jeśli koszt pełnego pomiaru
jest rzędu 1 % kosztu generacji — warto go ponieść zamiast ekstrapolować.

### 2.5 Weryfikacja outlierów: cztery potwierdzone, trzy odrzucone **[Z]**

Wszystkie 7 przekroczeń domierzono estymatorem wariancyjnym (M = 40), **głęboko w stanie
ustalonym** (po 120 renderach w tej samej instancji):

| scena | lok | sonda n=8 | domiar M=40 | `N` | wynik |
|---|---|---|---|---|---|
| `frl_apartment_2` | 0 | 0.13451 | 0.10703 | 54 → 34 | artefakt sondy |
| `apartment_0` | 285 | 0.12432 | **0.12799** | 46 → **49** | **potwierdzony, maksimum** |
| `apartment_0` | 307 | 0.12652 | 0.12632 | 48 → **48** | potwierdzony |
| `apartment_0` | 308 | 0.12390 | 0.12449 | 46 → **46** | potwierdzony |
| `apartment_0` | 310 | 0.12030 | 0.11799 | 43 → **42** | potwierdzony |
| `hotel_0` | 101 | 0.12136 | 0.11305 | 44 → 38 | poniżej progu |
| `hotel_0` | 95 | 0.11693 | 0.10959 | 41 → 36 | poniżej progu |

Cztery potwierdzone leżą w sąsiadujących `loc_id` jednej sceny — to jedno realne skupisko,
nie rozproszony szum. **Prawdziwe maksimum: `sigma_1` = 0.12799 → `N_raw` = 49.**

### 2.6 Estymator połówkowy ma sufit dokładności niezależny od liczby renderów **[Z]**

Bootstrap 3000×, rendery ze stanu ustalonego, referencja z estymatora wariancyjnego (80 renderów):

| pozycja | SD przy n=8 | n=20 | n=40 |
|---|---|---|---|
| `office_1/33` | 5.5 % | 5.5 % | 5.5 % |
| `frl_apartment_5/186` | 4.2 % | 4.2 % | 4.2 % |
| `room_0/43` | 4.5 % | 4.6 % | 4.5 % |

**Wyjaśnienie strukturalne:** estymator liczy RMSE po ~85 000 komórkach spektrogramu, a
`RMSE²·h/2` jest estymatorem `σ²` o **jednym stopniu swobody na komórkę, niezależnie od `h`**.
Zwiększanie liczby renderów zmniejsza amplitudę różnicy `A−B`, ale nie jej względną precyzję;
tę ogranicza efektywna liczba niezależnych komórek (rzędu 600–1200 z 85 324 — silna korelacja
przestrzenna). Estymator wariancyjny ma `n−1` stopni swobody na komórkę i **poprawia się z `n`**
(niepewność 0.1–1.1 % przy 80 renderach).

To jest realny wynik metodologiczny, nie ciekawostka: znaczy, że dokumentacyjne „~10 % błędu
`sigma_1` z 8 renderów" było oszacowaniem, a prawdziwa wartość to 4–6 % i **nie da się jej
poprawić dokładając rendery do tego estymatora**.

### 2.7 Rozgrzewka Simulatora jest własnością konstrukcji, nie pozycji **[Z]**

Pierwsze ~10 renderów w świeżej instancji ma wyższy szum. Zmierzone estymatorem wariancyjnym,
bloki po 10 renderów, 100 renderów na pozycję, 3 sceny:

| pozycja | blok 1 (r0–9) | stan ustalony (r40–99) | nadwyżka |
|---|---|---|---|
| `office_1/33` | 0.11286 | 0.10130 ± 0.00266 | +11.4 % (+4.4 SD) |
| `frl_apartment_5/186` | 0.03945 | 0.03289 ± 0.00041 | +19.9 % (+16.1 SD) |
| `room_0/43` | 0.06984 | 0.06329 ± 0.00090 | +10.4 % (+7.3 SD) |

**Test rozstrzygający:** po przeniesieniu agenta na drugą pozycję **w tej samej instancji**
(po 100 renderach) pierwszy blok **nie** jest podwyższony — stosunek „pierwszy/pozostałe"
1.004, 0.999, 0.993 wobec 1.114, 1.199, 1.104 na pozycji pierwszej. Efekt dotyczy zatem
18 lokalizacji (po jednej na scenę), a nie 1740.

Bezpośredni skutek na produkcyjnym parametrze: `office_1/5` daje `sigma_1` = 0.0918 (N=25)
przy `WARMUP_DISCARD = 20` i **0.0817 (N=20)** przy 500.

### 2.8 Refaktoryzacja nie zmieniła zachowania **[Z]**

| test | wynik |
|---|---|
| `--verify office_1` przed vs po | wyjście **bit-identyczne**, 94 linie |
| `--status` przed vs po | identyczne |
| `--dry-run apartment_2` przed vs po | identyczne |
| **sonda na GPU przed vs po** | **identyczne do 6 cyfr**: 0.081716 / 0.070838 / 0.067673 |
| rejestr eksperymentów | 22/22, identyczny zbiór nazw |
| import 22 modułów | 22/22 |
| pyflakes — nierozwiązane nazwy | 0 (3 znalezione i naprawione w trakcie) |

Test na GPU jest rozstrzygający, bo świeży Simulator odtwarza tę samą sekwencję RNG
(ustalenie `e1`, **[P]**) — identyczne `sigma_1` do szóstej cyfry oznacza, że kolejność wywołań
audio nie drgnęła.

### 2.9 Test fizyczny: RT60 z symulacji vs Sabine/Eyring **[Z]**

Pierwszy test w projekcie sprawdzający **fizykę**, a nie spójność wewnętrzną. Obie strony
równania z niezależnych źródeł: powierzchnie per kategoria z `mesh_semantic.ply`, absorpcja
z `replica_material_config.json`, RT60 z całkowania wstecznego Schroedera po uśrednionej
energii 30 renderów, w pasmach oktawowych 125–4000 Hz.

**Wynik dla scen zamkniętych** (stosunek zmierzone/Eyring w pasmach 500 Hz – 2 kHz):

| scena | V [m³] | 500 Hz | 1 kHz | 2 kHz | pomiar 1 kHz | Eyring |
|---|---|---|---|---|---|---|
| `room_0` | 83 | 1.21 | 1.53 | 1.38 | 0.463 s | 0.302 s |
| `office_1` | 23 | 1.15 | 1.41 | 1.27 | 0.401 s | 0.284 s |
| `hotel_0` | 75 | 1.35 | 1.74 | 1.75 | 0.550 s | 0.317 s |
| `apartment_0` | 379 | 1.13 | 1.16 | 1.29 | 0.771 s | 0.664 s |

**Mediana 1.32×, zakres 1.13–1.75× (n = 12).** Wartości bezwzględne RT60 (0.40–0.77 s przy
1 kHz) są typowe dla umeblowanych wnętrz mieszkalnych. Kierunek odchylenia też jest
oczekiwany: Eyring zakłada **równomiernie rozłożoną** absorpcję, a w tych scenach jest ona
skupiona na kilku powierzchniach (`blinds` i `sofa` mają α = 0.75, ściany 0.04) — absorpcja
skupiona daje pogłos **dłuższy** niż przewiduje model równomierny.

**Czego test nie dowodzi:** Sabine i Eyring zakładają pole dyfuzyjne; zgodność co do czynnika
~2 jest tu normalna. Test wyklucza błędy rzędu wielkości — materiały w ogóle nieprzypisane,
złą jednostkę, urwany RIR — i **nie kalibruje dokładności**.

**Dwa błędy metodyczne znalezione i naprawione w trakcie** (oba zmieniały wynik jakościowo):

1. *Filtrowanie w złej domenie.* Pierwsza wersja filtrowała `sqrt(energii)`, czyli `|h|`
   po uśrednieniu — co niszczy fazę, więc widmo takiego sygnału nie ma związku z widmem
   odpowiedzi impulsowej. Dawało to stosunki 0.13–1.15× i bezsens w niskich pasmach.
   Poprawnie: filtrować `h(t)` **każdego renderu osobno**, potem kwadrat, potem uśredniać.
   Po poprawce: 1.11–1.30×, spójnie we wszystkich pasmach.
2. *Okno dopasowania.* Klasyczne T20 (−5…−25 dB) jest tu niewłaściwe, bo w echolokacji
   **źródło jest współlokowane z odbiornikiem** — dźwięk bezpośredni ma energię
   nieporównywalnie większą niż pole pogłosowe i sam tworzy stromy spadek na starcie.
   Punkt −5 dB wypada po **3 ms**, zanim pogłos się rozwinie. Krzywa Schroedera ma dwa
   nachylenia; na `apartment_0` przy 1 kHz: −5…−15 dB → RT60 = 0.441 s, −25…−35 dB →
   0.755 s. Fit przesunięto w późną część (−25…−45 dB).

### 2.10 Sześć scen Replica nie ma sufitu — 46 % zbioru **[Z]**

Odkryte jako niewyjaśniony outlier w teście RT60 (`frl_apartment_0` dawał 0.28–0.34× zamiast
~1.3×), rozstrzygnięte pomiarem geometrii wszystkich 18 scen:

| grupa | sceny | pokrycie sufitem | lokalizacji |
|---|---|---|---|
| **otwarte** | `frl_apartment_0..5` | **5–7 %** | **807 (46 %)** |
| zamknięte | pozostałe 12 | 87–100 % | 933 |

Rodzina `frl_apartment_*` jest skanowana **bez sufitu**. Skoro sufitu nie ma w siatce, to nie
ma go też w symulacji akustycznej — energia ucieka górą, a pogłos jest krótszy niż w
zamkniętym pomieszczeniu o tej samej geometrii. Sabine i Eyring **nie są dla tych scen
właściwym odniesieniem** (zakładają zamkniętą objętość), więc wyłączono je z agregatu — to
nie jest „niezgodność modelu", tylko niestosowalność.

**To wyjaśnia wzorzec zaobserwowany wcześniej w census, ale wtedy nieskomentowany:** sceny
`frl_apartment_*` mają najniższą podłogę szumu w całym zbiorze. Rozdzielenie jest **zupełne**:

| grupa | mediana `sigma_1` | zakres | Mann-Whitney |
|---|---|---|---|
| otwarte (6 scen) | 0.04570 | 0.04136–0.04892 | **p = 0.00005** |
| zamknięte (12 scen) | 0.06322 | 0.05648–0.08830 | |

`max(otwarte) = 0.04892 < min(zamknięte) = 0.05648` — żadna scena otwarta nie jest głośniejsza
od żadnej zamkniętej. Mechanizm jest spójny: krótszy ogon pogłosowy → mniej stochastycznych
odbić w oknie 60 ms → mniejsza wariancja Monte Carlo → niższe `sigma_1` → niższe `N`.

**Czy to własność dzielona z literaturą?** Sprawdzone: Replica dostarcza **jedną**
geometrię na scenę — `mesh.ply` i `habitat/mesh_semantic.ply` mają bit-identyczne tablice
wierzchołków (1 757 500, ten sam zakres z). Nie istnieje zamknięty wariant siatki, którego
SoundSpaces 1.0 mógłby użyć, a ich metadane (`points.txt`, `graph.pkl`) to te same pliki,
których używa ten projekt. Bezpośredniego dowodu jednak brak — kod renderujący SS 1.0 nie
jest opublikowany. Test rozstrzygający (pomiar RT60 na ich RIR-ach, ~7 GB) opisany
w `OBSERWACJE_METODOLOGICZNE.md` §1.

**Konsekwencje do wypunktowania w pracy:**
- 46 % lokalizacji pochodzi ze scen akustycznie otwartych; ich echa mają systematycznie
  krótszy pogłos. To **własność zbioru Replica**, nie generatora.
- Jedna z trzech scen held-out (`frl_apartment_5`) jest otwarta, a dwie (`apartment_2`,
  `office_4`) zamknięte — zbiór testowy miesza oba typy.
- Niższe `N` w tych scenach nie jest artefaktem reguły adaptacyjnej, tylko poprawną reakcją
  na realnie niższy szum.

---

## 3. Ustalenia z zastrzeżeniami — cytować razem z zastrzeżeniem

### 3.1 Odsetek próbek wymagających dorenderowania **[Z-]**

Na `office_1` (576 próbek) dorenderowania wymagało **41.0 %**. Symulacja Monte Carlo reguły
odtworzona na prawdziwych renderach przewiduje 46.2 % dla najgłośniejszej pozycji tej sceny,
37.5 % dla `room_0/43` i 0.0 % dla `frl_apartment_5/186` (tam `N` obcina się do `N_MIN`).

**Mechanizm** (to jest wartościowe do pracy): `snr_probe < próg` zachodzi w przybliżeniu wtedy,
gdy `sigma_1` zmierzone z `N` renderów wypadnie wyżej niż `sigma_1` z sondy. Oba estymatory są
nieobciążone i mają podobny rozrzut, więc **z konstrukcji** zdarza się to w około połowie
przypadków. To nie jest usterka reguły — to konsekwencja celowania dokładnie w próg.

**Zastrzeżenie:** liczba 41 % jest zmierzona tylko na jednej scenie i będzie się różnić między
scenami (0 % tam, gdzie `N` obcina `N_MIN`).

### 3.2 Model przewidujący 0.066 % próbek przy limicie — **odwołuję sformułowanie „skalibrowany"** **[Z-]**

W trakcie sesji napisałem, że model „przewiduje 0.17 %, zaobserwowano 0.17 % — model jest
skalibrowany". **To było przesadzone i nie należy tego cytować w tej formie.** Zgodność
dotyczyła **jednego zdarzenia na 576 próbek**; przedział ufności Poissona dla 1 obserwacji to
w przybliżeniu 0.03–5.6 zdarzenia. Trafienie co do drugiego miejsca po przecinku jest przy
takiej liczności zbieżnością, nie potwierdzeniem modelu.

Co można powiedzieć uczciwie: model dał wynik tego samego rzędu wielkości co obserwacja i nie
został przez nią obalony. Do wniosków o `N_MAX` i tak użyto census, nie modelu.

### 3.3 Sygnał 10° na geometrii produkcyjnej **[Z-]**

`SIGNAL_10DEG = 0.0644` pochodzi z wcześniejszej fazy **[P]**, mierzonej na pozycjach
`snap_point` (czyli 0.21 m wyżej). Sprawdzenie na geometrii produkcyjnej dało 0.06441–0.06491
(3 pozycje, `office_1`), czyli w udokumentowanym zakresie 0.0639–0.0662. Stosunek `sigma_1`
produkcja/charakteryzacja = 0.962×.

**Zastrzeżenie:** 3 pozycje w jednej scenie. To potwierdza zgodność rzędu wielkości, nie
przemierza stałej.

### 3.4 Bias pierwszej lokalizacji sceny **[Z-]**

Census wykazał, że pierwsza sondowana lokalizacja każdej sceny jest systematycznie głośniejsza:
mediana percentyla w scenie **92 %**, 13/18 powyżej 75. percentyla, Wilcoxon **p = 0.001**.

Domiar 5 takich lokalizacji w stanie ustalonym pokazał, że to **w większości efekt
przestrzenny**, nie rozgrzewkowy: `loc_id` rośnie wzdłuż siatki punktów, więc id 0 to róg
sceny, często przy ścianach (`frl_apartment_3/0`: 0.09444 w stanie ustalonym przy medianie
sceny 0.04550 — wciąż 2× głośniej). Rozgrzewka dokładała kilka procent.

**Zastrzeżenie:** rozdzielenie obu składników opiera się na 5 domiarach przy szumie sondy ~5 %.
Różnica median (−3.9 % dla pierwszych vs −1.0 % dla pozostałych) to około 1.3 SE — kierunek
zgodny, ale nie rozstrzygnięty ilościowo.

---

## 4. Decyzje projektowe podjęte w tych sesjach

| decyzja | wartość | podstawa |
|---|---|---|
| źródło `y` | `graph.pkl` | §2.1 **[Z]** |
| `audio_sims_per_render` | 2 → **1** | §2.2 **[Z]** |
| `N_MAX` | 40 → **64** | §2.3, §2.5 **[Z]** |
| `WARMUP_DISCARD` | 20 → **500** | §2.7 **[Z]** + margines |
| kolejność scen | held-out najpierw | harmonogram, nie pomiar |
| układ katalogów | podkatalog na scenę | organizacja |

**`N_MAX = 64`:** pokrywa `sigma_1` do 0.14720, powyżej najwyższej wartości census (0.13451)
i potwierdzonego maksimum (0.12799). Żadna z 1740 lokalizacji nie jest obcinana. `N_MAX = 48`
odrzucono, bo pokrywa do 0.12748 — **poniżej** potwierdzonego maksimum. Margines jest potrzebny
także dlatego, że produkcyjna sonda zmierzy każdą lokalizację na nowo, raz, z SD ~5 %.
Koszt: +3.6 min na 31.9 h.

**`WARMUP_DISCARD = 500`:** zmierzony punkt osiadania jest dużo niższy (skumulowane odchylenie
<1 % od renderu 20), ale 500 to wartość, którą projekt udokumentował jako koniec rozgrzewki
w `gpu_memory_scale` **[P]**. To 25× zmierzony punkt osiadania. Koszt: 21 min na cały zbiór
(+1.1 %). **Wybór z marginesem, nie z konieczności** — tak należy to opisać.

**Budżet końcowy [Z]:** 812 273 renderów × 0.1412 s = **31.9 h**, dysk **14.0 GiB**.
Najdłuższa scena: `apartment_0` 6.42 h (211 lokalizacji, średnie `N` 20.32). Trzy sceny
held-out łącznie 6.16 h.
Wszystkie składniki zmierzone: tempo i narzut pętli weryfikacyjnej (1.0578×) z pełnej sceny
`office_1`, `N` z census wszystkich lokalizacji, rozgrzewka z parametru.

---

## 5. Czego NIE sprawdzono **[X]**

Wymieniam, żeby nie powstało wrażenie, że zostało:

1. **Poprawność akustyczna ech.** Weryfikacja obejmuje kształty, dtype, brak NaN/Inf, brak
   próbek zerowych, zakres mieszczący się w `float16`, sensowny zakres głębi — czyli
   **integralność danych, nie ich fizyczną poprawność**. Nie porównano ech z żadnym pomiarem
   rzeczywistym ani z innym symulatorem.
2. **Dokładność przypisania materiałów akustycznych Repliki.** `REPLICA_MATERIALS.md` odnotowuje,
   że 2.14 % powierzchni dostaje materiał domyślny i że przypisania są częściowo niskiej pewności.
   Nie weryfikowano tego w tych sesjach.
3. **Zbiór testowy habitat-sim** (`pytest habitat-sim/tests`) nie był uruchamiany.
4. **Wszystkie liczby z fazy przed 2026-07-28** (sekcja 1) — przejęte, nie sprawdzane.
5. **Wpływ zmiany `y` o 0.21 m na sam rozkład szumu** — sprawdzono zgodność na 3 pozycjach
   (§3.3), nie przemierzono charakterystyki od nowa.
6. **Czy 36 orientacji faktycznie daje przewagę nad 4** — to jest teza pracy, a nie coś, co
   zostało tu ustalone. Wygenerowany zbiór dopiero pozwoli to zbadać.

---

## 6. Błędy popełnione i skorygowane

Odnotowuję je, bo (a) metodologicznie uczciwy opis powinien je zawierać, (b) niektóre trafiły
przejściowo do dokumentacji i mogły zostać zacytowane.

| błąd | jak wykryty | status |
|---|---|---|
| **„Estymator połówkowy ma SD 11–17 % przy 40 renderach"** — nieprawda. Replikaty liczyłem w kolejności naturalnej, więc replikat 1 to były rendery 0–19 (rozgrzewka), a 2 — stan ustalony. Mierzyłem przesunięcie rozgrzewkowe, nie rozrzut estymatora. | Bootstrap na tych samych renderach w losowej kolejności dał 4.5–6.0 % | **Skorygowany**, prawdziwa wartość w §2.6 |
| **Ekstrapolacja przewidywała 0 przekroczeń `N_MAX = 40`** | Pełny census znalazł 7 | **Skorygowany**, opisany jako wynik w §2.4 |
| **`WARMUP_DISCARD = 20` uznane za wystarczające** na podstawie Bloku A, gdzie „plateau" szacowałem z tego samego przebiegu — to zaniża wykrywalność wolnego ogona | Census pokazał resztkowy bias pierwszych lokalizacji | **Skorygowany** (500), ale patrz §3.4 — składnik przestrzenny okazał się większy niż rozgrzewkowy |
| **„Model skalibrowany"** przy zgodności na 1 zdarzeniu | Analiza własna przy pisaniu tego raportu | **Odwołane**, patrz §3.2 |
| Skrypt kontrolny `.gitignore` błędnie czytał kod wyjścia `git check-ignore -v` (zwraca 0 także dla wzorca negacji) | Sprzeczność z `git status --ignored` | Naprawiony, kontrola powtórzona metodą rozstrzygającą |
| Ekstrakcja `params.py` po cichu pominęła 2 stałe (`CAMERA_RESOLUTION`, `CAMERA_HFOV`) | Jawne porównanie zbioru stałych z oryginałem | Naprawione przed użyciem |
| Wygenerowany bubel składniowy w `echo_ctl.py` (obejście zagnieżdżonych cudzysłowów w f-stringu) | Przegląd własny | Przepisane |
| `id` neutralnej lokalizacji wzięte z indeksów `points.txt` zamiast ze zbioru pkl | `KeyError` w trakcie | Naprawione |

**Wzorzec, który się powtarza:** wszystkie te błędy wynikały z **pomiaru na danych skażonych
przez inny efekt** albo z **przyjęcia założenia zamiast sprawdzenia**. Wykryte zostały wtedy,
gdy pomiar powtórzono inną metodą albo na pełnej populacji. To jest argument za tym, żeby
w pracy opisywać nie tylko wyniki, ale i kontrole krzyżowe.

---

## 7. Dowody: gdzie są i czy przetrwają

### 7.1 W repozytorium (przetrwa)

| dowód | lokalizacja |
|---|---|
| Eksperyment `noise_floor_remaining` (kod) | `my-operations/diagnostics/exp_noise_floor.py` |
| Wyniki `noise_floor_remaining` (22 pozycje) | `outputs/diagnose_rlr_noise_out/diagnostics_report.json` |
| Tryb census (kod) | `my-operations/generate_echo_dataset.py --probe-only` |
| Wykres rozgrzewki | `outputs/diagnose_rlr_noise_out/warmup_simulator.png` |
| Wykres rozkładu census | `outputs/diagnose_rlr_noise_out/probe_census.png` |
| Wszystkie decyzje i liczby | `my-operations/docs/GENERATOR_PARAMS.md` |
| Metadane każdej wygenerowanej sceny | atrybuty pliku `.h5` (parametry, hashe, commity, host, tempo) |

### 7.2 Skrypty pomiarowe — przeniesione do repozytorium (2026-07-29)

Wszystkie pomiary z sekcji 2 i 3 mają teraz skrypt w `my-operations/measurements/`
z docstringiem zawierającym pytanie, metodę, wynik i odsyłacz do sekcji tego raportu.
Mapa w `measurements/README.md`.

| skrypt | dowodzi | GPU |
|---|---|---|
| `agent_height_vs_pkl.py` | §2.1 — źródło `y` | tak |
| `navmesh_offset_survey.py` | §2.1 — skala offsetu na 1740 lokalizacjach | nie |
| `audio_duplication_bench.py` | §2.2 — 274 vs 139 ms, zachowanie sekwencji RNG | tak |
| `audio_path_render.py` → `audio_path_analyse.py` | §2.2, §2.6 — równoważność ścieżek | tak → nie |
| `simulator_warmup.py` | §2.7 — rozgrzewka per konstrukcja | tak |
| `probe_estimator_accuracy.py` | §2.6 — dokładność sondy, sufit estymatora | nie |
| `signal_10deg_production.py` | §3.3 — `SIGNAL_10DEG` po poprawce `y` | tak |
| `census_analysis.py` | §2.3, §2.4 — rozkład `N_raw`, werdykt o `N_MAX` | nie |
| `census_outlier_recheck.py` | §2.5, §3.4 — weryfikacja outlierów | tak |
| `probe_discard_unittest.py` | odrzucanie nadmiaru sondy przy `N < 8` | nie |

**Świadomie NIE przeniesiono** (wersje pośrednie, zastąpione):
`budget.py`, `budget2.py`, `compare_and_budget.py`, `nmax_verdict.py` — trzy kolejne wersje
rachunku budżetu i analiza na próbce 52 pozycji, wszystkie zastąpione przez
`census_analysis.py` liczący budżet z pełnego rozkładu; `validate_single_sim.py` (M=20)
zastąpiony przez parę `audio_path_render/analyse` (M=40); `split_diagnostics.py`, `do_split.py`
— narzędzia jednorazowe do refaktoryzacji, nie materiał dowodowy (refaktor jest udowodniony
testami regresji z §2.8).

### 7.3 Stan wersjonowania dowodów

| artefakt | rozmiar | w gicie | uzasadnienie |
|---|---|---|---|
| `outputs/probe_census/*.csv` | ~120 KB | **tak** (wyjątek dopisany 2026-07-29) | podstawa rozstrzygnięcia o `N_MAX` |
| `outputs/diagnose_rlr_noise_out/*.json`, `*.png` | ~2 MB | **tak** | wyniki eksperymentów i wykresy |
| `my-operations/measurements/*.py` | ~100 KB | **tak** | odtwarzalność każdej liczby |
| `outputs/measurements/*.npz` | 210 MB | **nie** | pośrednie spektrogramy; przy archiwizacji dołączyć `sha256sum` |
| `outputs/echoes_36deg/**` | docelowo 14 GB | **nie** | właściwy dataset |

Zweryfikowane metodą rozstrzygającą (`git status --porcelain --ignored`, bo
`git check-ignore -v` zwraca kod 0 także dla wzorca negacji).

---

## 8. Co się nadaje do pracy i jak to sformułować

**Jako wyniki metodologiczne (mocne):**

- Test spójności z istniejącym zbiorem referencyjnym jako sposób wykrycia błędu konfiguracji,
  którego nie widać w kodzie (§2.1). Wykorzystanie determinizmu renderingu wizualnego jako
  narzędzia diagnostycznego.
- Adaptacyjne próbkowanie sterowane wariancją z weryfikacją po fakcie — i uczciwy opis, że
  dorenderowanie dotyczy ~40 % próbek **z konstrukcji reguły**, a nie z jej wady (§3.1).
- Ograniczenie dokładności estymatora połówkowego przez korelację przestrzenną spektrogramu,
  nie przez liczbę renderów (§2.6). To nietrywialny wynik.
- Pomiar pełnej populacji zamiast ekstrapolacji, gdy koszt pomiaru jest rzędu 1 % kosztu
  właściwego zadania (§2.4) — wraz z przykładem, jak systematyczne próbkowanie ominęło skupisko.

**Jako decyzje inżynierskie (do opisania, nie do obrony jako wyniki naukowe):**

- Podwójna symulacja audio (§2.2) — to jest znalezisko implementacyjne o dużym wpływie na
  koszt, ale nie wynik naukowy.
- `N_MAX`, `WARMUP_DISCARD`, układ katalogów, struktura kodu.

**Czego NIE wpisywać jako wynik:**

- Zgodności modelu na jednym zdarzeniu (§3.2).
- Liczb przejętych z wcześniejszej fazy bez zaznaczenia, skąd pochodzą (sekcja 1).
- Czegokolwiek o jakości akustycznej ech — to nie było badane (§5).

**Pytania, na które trzeba umieć odpowiedzieć na obronie:**

1. Dlaczego `sigma_1` liczone jest dwoma różnymi estymatorami w różnych miejscach?
   (Odpowiedź: generator używa połówkowego, bo musi być spójny z regułą wyznaczającą `N`;
   pomiary kontrolne używają wariancyjnego, bo ma `n−1` stopni swobody zamiast 1 — §2.6.)
2. Dlaczego `N_MAX = 64`, skoro maksimum to 49? (Margines na to, że produkcyjna sonda mierzy
   raz z SD ~5 %; koszt marginesu 0.2 % czasu — §4.)
3. Dlaczego 41 % próbek wymaga dorenderowania? (Reguła celuje dokładnie w próg, oba estymatory
   są nieobciążone o podobnym rozrzucie — §3.1.)
4. Skąd wiadomo, że usunięcie zdublowanej symulacji nic nie zepsuło? (Sześć porównań poniżej
   2 SE plus pełna scena z testami rozkładów — §2.2.)
