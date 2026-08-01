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

**Czy to własność dzielona z literaturą? NIE — sprawdzone pomiarem.** Replica dostarcza
jedną geometrię na scenę (`mesh.ply` i `habitat/mesh_semantic.ply` mają bit-identyczne
tablice wierzchołków), więc SoundSpaces 1.0 nie miał alternatywnej, zamkniętej siatki.
Mimo to ich prekomputowane RIR-y **nie wykazują sygnatury braku sufitu**. Po pobraniu ich
danych (7 GB) i dwóch kontrolach — dopasowaniu odległości źródło–odbiornik oraz odtworzeniu
ich geometrii źródła w naszym silniku:

| scena | sufit | V [m³] | nasze SS 2.0 | SoundSpaces 1.0 |
|---|---|---|---|---|
| `office_1` | zamknięta | 23 | 0.358 s | 0.396 s |
| `frl_apartment_2` | brak | 191 | **0.186 s** | **0.463 s** |

Silniki zgadzają się na scenie zamkniętej (~10 %) i rozjeżdżają 2.5× wyłącznie na otwartej.
Mechanizm nieustalony (kod SS 1.0 nieopublikowany); wiodąca hipoteza to domykanie objętości
w ich potoku. **Konsekwencja: różnica wobec baseline'ów jest realna i dotyczy 46 %
lokalizacji.** Szczegóły: `OBSERWACJE_METODOLOGICZNE.md` §1.

**Konsekwencje do wypunktowania w pracy:**
- 46 % lokalizacji pochodzi ze scen akustycznie otwartych; ich echa mają systematycznie
  krótszy pogłos. To **własność zbioru Replica**, nie generatora.
- Jedna z trzech scen held-out (`frl_apartment_5`) jest otwarta, a dwie (`apartment_2`,
  `office_4`) zamknięte — zbiór testowy miesza oba typy. **Sprostowanie, patrz §2.11:**
  klasyfikacja `apartment_2` jako „zamkniętej" pochodzi z pokrycia sufitem i jest myląca —
  pomiar ucieczki promieni pokazuje, że 36 % jej lokalizacji traci ponad 10 % kąta bryłowego
  przez otwory boczne.
- Niższe `N` w tych scenach nie jest artefaktem reguły adaptacyjnej, tylko poprawną reakcją
  na realnie niższy szum.

---

### 2.11 Pomiar ucieczki promieni — sceny dzielą się na trzy grupy, nie dwie **[Z]**

Klasyfikacja z §2.10 opiera się na **pokryciu sufitem** — heurystyce geometrycznej. SoundSpaces
2.0 (arXiv 2206.08312) wskazuje właściwą wielkość wprost: *„the scene meshes need to have high
quality, i.e., no large open holes on the mesh, otherwise the rays will leak from the holes"* —
i przypisuje jej konkretny artefakt we własnych danych (*„lots of broken meshes (…) results in
ray leaking from holes and smaller reverberation in general"*, o Matterport3D). Zmierzono ją
dla wszystkich 1740 lokalizacji.

**Metoda.** Równoprostokątny sensor głębi w punkcie słuchacza — jeden render = pełna sfera
256 × 512 kierunków; piksel o głębi 0 to kierunek bez geometrii. Udział ważony kątem bryłowym
= ułamek uciekających promieni izotropowych (pierwsze odbicie; miara RLR liczy odbicia
wielokrotne, więc to **dolne ograniczenie** otwartości). Skrypt: `ray_escape_survey.py`.

Dedykowane API biblioteki (`RLRA_GetIndirectRayEfficiency()`, `RLRAudioPropagation.h:503`)
istnieje w naszej kopii, ale **habitat-sim go nie eksponuje** — brak w publicznym API
`AudioSensor` i w bindingach. `sim.cast_ray()` odpada niezależnie: na Replice Bullet odrzuca
siatkę (czworokąty — `isMeshPrimitiveValid : Invalid primitive 0`) i konstrukcja Simulatora
kończy się `AssertionError`.

| grupa | scen | lokalizacji | mediana | max | lok. > 10 % | rozkład kątowy |
|---|---|---|---|---|---|---|
| szczelne | 8 | 516 (29.7 %) | 0.00 % | 0.09 % | 0 % | — |
| nieszczelne bokiem | 4 | 417 (24.0 %) | 0.53–3.60 % | **48.3 %** | 0–36 % | horyzont i poniżej |
| bez sufitu | 6 | 807 (46.4 %) | 21.8–23.6 % | 29.0 % | 89–93 % | **tylko** powyżej horyzontu |

Rozdzielenie mechanizmów jest bezbłędne: w `frl_apartment_*` pasmo > 60° elewacji ucieka
w 99–100 %, a przy horyzoncie w 0.0 %; w czterech pozostałych dokładnie odwrotnie.
**Kontrola negatywna wbudowana:** 8 scen daje ≤ 0.09 % w każdej lokalizacji, więc „głębia = 0"
nie generuje fałszywych trafień.

**Co to koryguje:** `apartment_2` (held-out), `apartment_1`, `office_2` i `office_3` były w
§2.10 opisane jako „zamknięte" — nie są akustycznie szczelne. `sigma_1` tego nie wykryło, bo
przeciek boczny dotyczy mniejszości lokalizacji w scenie i nie przesuwa mediany sceny.
Rozdzielenie `sigma_1` z §2.10 pozostaje prawdziwe, ale jest **grubsze**, niż zakładano.

Omówienie i konsekwencje dla podziału train/held-out: `OBSERWACJE_METODOLOGICZNE.md` §1
(„Uściślenie 2026-07-29"). Dane: `outputs/measurements/ray_escape/` (w gicie).

---

### 2.12 Eksperyment: domknięcie sceny sufitem wyjaśnia większość rozbieżności z SS 1.0, ale przestrzeliwuje **[Z]**

Hipoteza z §2.10 — że potok SoundSpaces 1.0 domykał objętość — była dotąd nieprzetestowana.
Test falsyfikowalny w obie strony: dokleić sufit do siatki `frl_apartment_2` i sprawdzić,
czy RT60 rośnie w kierunku 0.463 s.

**Wykonanie.** `patch_scene_ceiling.py` dokleja do `mesh_semantic.ply` płaszczyznę na
wysokości rzeczywistego sufitu (mediana `z` istniejącego fragmentu sufitu = 1.294;
`gravity_dir` potwierdza, że pionem w surowym PLY jest **z**), pokrywającą cały rzut
poziomy, z `object_id` **istniejącego** obiektu klasy `ceiling` — dzięki temu
`info_semantic.json` nie wymaga zmiany, a materiał przypisuje się poprawnie
(0 ostrzeżeń „Material for category 'ceiling' was not found"). Oryginalne rekordy PLY
są przepisywane bajt w bajt, nowe doklejane; navmesh skopiowany, więc **zbiór lokalizacji
jest identyczny**. Scena zapisana obok, w `outputs/patched_scenes/` — dataset nietknięty.

**Kontrola domknięcia:** ucieczka promieni spadła z **21.79 % → 0.00 %** mediany
(max 28.36 % → 0.01 %), czyli scena stała się nieodróżnialna od w pełni szczelnych.

**Kluczowa kontrola interpretacyjna:** ten sam pomiar na `office_1`, scenie **naturalnie
zamkniętej**, bez żadnego łatania. Bez niej nie da się odróżnić „łata przestrzeliła" od
„nasz silnik generalnie gra dłużej niż SS 1.0".

| pomiar | RIR | RT60 @ 1 kHz | mediana 500 Hz–2 kHz vs SS 1.0 |
|---|---|---|---|
| `office_1` — naturalnie zamknięta *(kontrola)* | 1.08 s | 0.356 s | **0.94×** (0.90–1.09 po pasmach) |
| `frl_apartment_2` — oryginalna | 0.48 s | 0.222 s | **0.48×** |
| `frl_apartment_2` — **załatana** | 1.55 s | 0.628 s | **1.36×** |

60 par źródło–odbiornik, odległość 1.0–3.0 m (średnia **1.99 m** — dokładnie tyle, co
w parach SS 1.0), 5 renderów na parę uśrednionych w domenie energii po przefiltrowaniu
każdego renderu osobno w domenie ciśnienia, estymator identyczny z zastosowanym do danych
SS 1.0 (Schroeder → T20 w oknie −5…−25 dB).

**Wynik: hipoteza potwierdzona co do kierunku i rzędu wielkości, ale nie co do wartości.**
Sam RIR wydłużył się 3.2× (0.48 → 1.55 s), RT60 wzrósł 2.5–3.4× zależnie od pasma, a błąd
względem SS 1.0 spadł z 2.1× za mało do 1.36× za dużo — w skali logarytmicznej z 0.73 do
0.31, czyli **2.4× bliżej**. Brak sufitu odpowiada więc za większość rozbieżności. Ale
domknięcie **przestrzeliwuje**: gdyby łata odtwarzała to, co miał SS 1.0, wynik powinien
wypaść ok. 0.94× jak na scenie naturalnie zamkniętej, a wypada 1.36× — reszta to czynnik
**1.45×**.

**Czego eksperyment NIE rozstrzyga.** Przyczyna przestrzelenia nie została zbadana.
Kandydaci: (a) materiał `ceiling` w `replica_material_config.json` to „Gypsum Board"
o α = 0.04–0.05 w środku pasma, czyli bardzo odbijający — 70 m² takiej powierzchni mocno
podnosi RT60; (b) płaska płaszczyzna nie ma geometrii rozpraszającej, jaką ma prawdziwy
sufit z oprawami i listwami; (c) SS 1.0 mógł domykać objętość inaczej albo mieć inne
materiały. **Materiału celowo nie dobierano pod wynik** — użyto wpisu, który konfiguracja
Repliki przypisuje kategorii `ceiling`. Dostrajanie go do 0.463 s byłoby dopasowywaniem
odpowiedzi, nie pomiarem.

**Wniosek dla pracy:** różnica wobec baseline'ów SS 1.0 na scenach `frl_apartment_*` ma
**zidentyfikowaną, w większości usuwalną przyczynę geometryczną**. To mocniejsze zdanie
niż „nie wiadomo, skąd rozbieżność". Nie oznacza jednak, że łatanie należy włączyć do
generacji — patrz `OBSERWACJE_METODOLOGICZNE.md` §1.

Dane: `outputs/measurements/ceiling_patch/*.json` (w gicie).

---

### 2.13 Zalatanie WSZYSTKICH dziur + dobór materiału — RT60 trafia w SS 1.0 z dokładnością 0.96× **[Z / Z-]**

Rozwinięcie §2.12 z eksperymentu na jednej scenie i jednym typie dziury do pełnego zestawu.
Skrypty: `patch_scene_holes.py`, `patch_material_sweep.py`.

**Gdzie są dziury i czym są** (wykryte przez krawędzie brzegowe → pętle, metodą, którą opisuje
paper Repliki). Każda z 10 nieszczelnych scen ma **1–2 dziury o polu > 1 m²** i kilkaset
poniżej 0.5 m² — te małe to brzegi obiektów (spod krzesła, blat stołu) i **nie wolno ich
łatać**. Próg 1 m² rozdziela je czysto: największa „obiektowa" ma 0.47 m², najmniejsza
„pomieszczeniowa" 1.12 m².

| sceny | dziura | typ | materiał łaty |
|---|---|---|---|
| `frl_apartment_0..5` | 89.5–99.8 m², pozioma na szczycie ścian | **brakujący sufit** | `wall` → Gypsum Board |
| `apartment_1`, `apartment_2` | 24.2 / 34.1 m², pionowa, brzeg `floor`+`ceiling`+`wall` | **urwana krawędź skanu** | `wall` → Gypsum Board |
| `office_2`, `office_3` | 2.14 / 1.74 m² i 1.12 / 1.40 m², pionowe | **OKNO i DRZWI** | `window` → **Glass**, `door` → **wood, Thick** |

Okna i drzwi to nie przypadek: szkło nie odbija światła strukturalnego IR, więc skaner ich
nie złapał. Materiał nie jest dobierany ręcznie — łata dziedziczy `object_id` z otoczenia
dziury, więc materiał wynika z tej samej konfiguracji, co reszta sceny. Weryfikacja
end-to-end potwierdziła przypisanie (`window`→Glass α=0.12, `door`→wood α=0.06,
`wall`→Gypsum α=0.04).

**Domknięcie — zmierzone, wszystkie sceny:**

| scena | ucieczka przed | po | scena | przed | po |
|---|---|---|---|---|---|
| `frl_apartment_0` | 23.60 % | **0.00 %** | `frl_apartment_4` | 22.94 % | **0.00 %** |
| `frl_apartment_1` | 23.59 % | **0.01 %** | `frl_apartment_5` | 23.01 % | **0.00 %** |
| `frl_apartment_2` | 21.79 % | **0.00 %** | `apartment_1` | 0.78 % | **0.04 %** |
| `frl_apartment_3` | 23.36 % | **0.12 %** | `apartment_2` | 3.60 % | **0.24 %** |
| `office_2` | 2.27 % | **0.00 %** | `office_3` | 0.53 % | **0.00 %** |

Wszystkie 18 scen jest więc obecnie domykalnych. Reszta w `apartment_1` (maks. 20.5 % w jednej
lokalizacji) to prawdopodobnie druga dziura poniżej progu 1 m².

**Dwa błędy, które kosztowały dwa nieudane podejścia — warto je znać:**

1. **Trójkąt zapisany jako zdegenerowany czworokąt (c, u, v, v) nie daje powierzchni**
   w tym potoku. Format Repliki ma stałą długość rekordu ściany 19 B tylko dlatego, że
   wszystkie ściany są czworokątami.
2. **Znak normalnej z SVD jest dowolny.** Dla `frl_apartment_2` wypadł `[0, 0.01, −1]`, przez
   co baza (e1, e2) była obrócona i nawinięcie dawało ściany zwrócone **w górę**, odwrotnie od
   pomieszczenia — renderer odrzucał je jako tylne. Objaw jest zdradliwy: geometria jest
   w pliku, we właściwym miejscu, o poprawnym polu 71 m², a ucieczka promieni **nie zmienia
   się ani o jotę** (21.79 % przed i po). Rozwiązanie: łata dwustronna, bez zgadywania
   konwencji odrzucania w dwóch niezależnych potokach.

Dodatkowo wachlarz trójkątów z centroidu nie pokrywa **niewypukłego** obrysu — zastąpiony
siatką pełnych czworokątów przyciętą testem należności do wielokąta.

**Korekta wysokości łaty sufitowej (2026-07-30) — łata siedziała 0.37 m za wysoko.**
Dziura sufitowa jest ograniczona **szczytem ścian**, więc pierwsze wypełnienie szło w jej
własnej płaszczyźnie. Pomiar na wszystkich 14 scenach pokazał, że to błąd:

| grupa | szczyt siatki − sufit | wysokość pomieszczenia |
|---|---|---|
| sceny szczelne (6 z 8) | **0.02–0.10 m** | 2.64–2.87 m |
| `frl_apartment_0..5` | **0.32–0.42 m** | 2.69–2.74 m |

W scenach szczelnych sufit leży praktycznie **na szczycie siatki**. W rodzinie FRL ściany
wystają ~0.35 m nad ocalały fragment klasy `ceiling`, a wysokość liczona od tego fragmentu
(2.69–2.74 m) zgadza się z normą scen szczelnych — fragment jest więc na **prawdziwej**
wysokości sufitu, a nadwyżka ścian to materiał powyżej linii sufitu. Wypełnianie dziury w jej
płaszczyźnie zawyżało pokój z 2.71 m do ~3.10 m (**+14 % objętości**) i wydłużało pogłos.
Łata jest teraz opuszczana do wysokości ocalałego sufitu (nigdy podnoszona); domknięcie po
korekcie utrzymane (0.00–0.05 % we wszystkich sześciu scenach FRL).
*(Dwie sceny szczelne są tu odstające i słusznie: `apartment_0` ma 104 % pokrycia sufitem
i szczyt 1.69 m nad sufitem — to scena dwupoziomowa; `room_2` 0.46 m.)*

**RT60 wobec SoundSpaces 1.0** (60 par, odległość źródło–odbiornik średnio 1.99 m — tyle samo
co w parach SS 1.0, 5 renderów na parę, estymator identyczny z zastosowanym do ich danych).
Liczby poniżej pochodzą z łaty **przed** korektą wysokości i z materiałem semantycznym:

| pasmo | oryginalna | załatana | SS 1.0 | orig/SS1 | załat/SS1 |
|---|---|---|---|---|---|
| 250 Hz | 0.198 | 0.661 | 0.493 | 0.40× | 1.34× |
| 500 Hz | 0.241 | 0.578 | 0.520 | 0.46× | 1.11× |
| 1 kHz | 0.222 | 0.446 | 0.463 | 0.48× | **0.96×** |
| 2 kHz | 0.226 | 0.391 | 0.429 | 0.53× | 0.91× |
| **mediana 500 Hz–2 kHz** | | | | **0.48×** | **0.96×** |

Długość RIR: 0.48 s → 1.55 s. Kontrola z §2.12 (naturalnie zamknięty `office_1`, bez łatania)
daje 0.94×, więc 0.96× jest **na poziomie zgodności silników na scenie, która sufitu nigdy
nie potrzebowała**.

**Zastrzeżenie [Z-], bez którego tej liczby nie wolno cytować: materiał sufitu został
DOPASOWANY, nie zmierzony.** Sweep po drabince materiałów na tej samej scenie:

| materiał łaty | α @ 1 kHz | /SS 1.0 |
|---|---|---|
| Concrete | 0.02 | 1.60× |
| **Gypsum Board** (wybór semantyczny) | 0.04 | **1.48×** |
| wood, Thick | 0.06 | 1.47× |
| Steel | 0.10 | 1.39× |
| Glass | 0.12 | 1.39× |
| Foliage | 0.17 | 0.86× |
| Carpet | 0.20 | 1.06× |
| **Carpet, Heavy** (wybrany) | **0.37** | **0.97×** |
| Curtain | 0.75 | 0.71× |

Trzy rzeczy, które trzeba powiedzieć razem z tą tabelą:

- **To dopasowanie do jednej sceny, nie walidacja.** RIR-y SS 1.0 mamy dla `office_1`
  (zamkniętej, łata jej nie dotyczy) i `frl_apartment_2`. Zgodność 0.96× jest tautologiczna.
  Żeby to była walidacja, trzeba dociągnąć RIR-y drugiej sceny otwartej i sprawdzić na niej
  materiał dobrany tutaj.
- **Dopasowanie poprawia skalę, nie kształt widmowy.** Po dobraniu zostaje 1.34× przy 250 Hz
  i 0.91× przy 2 kHz — RT60 w SS 1.0 jest **płaskie w częstotliwości**, nasze opada. Jedna
  liczba (α) nie może tego naprawić.
- **Zależność nie jest monotoniczna**: Foliage (α = 0.17) daje 0.86×, a Carpet (α = 0.20)
  1.06×. Wynik zależy więc także od rozpraszania i kształtu widmowego pochłaniania, nie
  tylko od α @ 1 kHz. „α = 0.37" nie jest zmierzoną stałą fizyczną, tylko wierszem tabeli,
  który wypadł najbliżej.

Fizyczna interpretacja dopasowania, na tyle, na ile jest uprawniona: aby trafić w SS 1.0,
sufit musi być **akustycznie wytłumiony** (α ≈ 0.37), a nie surowym gipsem (α ≈ 0.04).
To nie jest absurd — sufity podwieszane z płyt akustycznych mają α 0.6–0.7, a konfiguracja
Repliki zawiera „Acoustic Tile" (0.60/0.70/0.70), tylko żadna klasa semantyczna go nie używa.

Dane: `outputs/measurements/ceiling_patch/*.json`, `outputs/measurements/ray_escape/*__patched.csv`
(w gicie). Załatane siatki: `outputs/patched_scenes/` — 742 MB, poza gitem, odtwarzalne
jednym wywołaniem `patch_scene_holes.py --all --ceiling-class rug`.

---

### 2.14 SS 1.0 nie może rozstrzygnąć poprawności — rozrzut międzysilnikowy to ±44 % na scenę **[Z]**

Pomiar odwrotny do §2.13: **ani jednego dopasowanego parametru**. Na scenach o **całej**
geometrii nie ma czego łatać ani dobierać, więc zgodność z SS 1.0 jest tam czystą własnością
dwóch silników. Pobrano RIR-y SS 1.0 dla 5 dodatkowych scen szczelnych (3.5 GB) i zmierzono
tym samym estymatorem, tą samą geometrią par (1–3 m, źródło oddalone), 40 par × 3 rendery.
Skrypt: `cross_engine_rt60.py`.

| scena szczelna | nasze SS 2.0 @1 kHz | SS 1.0 @1 kHz | nasze/SS 1.0 |
|---|---|---|---|
| `office_0` | 0.404 | 0.704 | **0.57×** |
| `room_1` | 0.337 | 0.471 | 0.76× |
| `room_0` | 0.433 | 0.483 | 0.89× |
| `office_1` | 0.358 | 0.394 | 0.94× |
| `room_2` | 0.549 | 0.449 | 1.21× |
| `hotel_0` | 0.547 | 0.331 | **1.65×** |

**Średnia geometryczna 0.95×, geometryczne SD 1.44×** (1σ: 0.66–1.37×; 2σ: 0.46–1.98×).
Kontrola poprawności strony SS 1.0: dla każdej sceny wszystkie 80 par dało skończone RT60,
więc rozrzut nie jest artefaktem odrzuconych par.

**Wniosek 1 — dobry dla pracy.** Silniki są **nieobciążone względem siebie w średniej**
(0.95×), ale rozjeżdżają się o **±44 % na pojedynczej scenie**, skrajnie 0.57× i 1.65×.
To empirycznie potwierdza — na 6 scenach — ograniczenie 1 z `GENERATOR_PARAMS.md` §5, które
do tej pory było ostrożnością *a priori*: **bezwzględne metryki nie są porównywalne między
SS 1.0 a SS 2.0 na poziomie sceny**. Wcześniejsze 0.94× na `office_1` było szczęśliwym
trafem, nie regułą — jedna scena to było za mało dowodu i moja teza z 30 lipca, że zgodność
na scenach szczelnych zwaliduje silnik, **była błędna**.

**Wniosek 2 — negatywny i to on rozstrzyga sprawę łatania.** Wobec takiego rozrzutu SS 1.0
**nie jest w stanie orzec**, czy łatanie poprawia wierność:

| `frl_apartment_2` | nasze/SS 1.0 | wobec rozrzutu scen szczelnych |
|---|---|---|
| geometria oryginalna | 0.49× | wewnątrz 2σ (granica 0.46×) |
| **załatana semantycznie** | **1.36×** | wewnątrz **1σ** (granica 1.37×) |

Oba warianty są „zgodne" z SS 1.0 w granicach szumu międzysilnikowego. Łatanie przesuwa
scenę z krawędzi 2σ do wnętrza 1σ, co jest słabym sygnałem pozytywnym, ale przy n = 1 scenie
otwartej i SD 1.44× **nie jest dowodem**. Żadne dalsze dopasowywanie materiału tego nie
zmieni — problemem jest rozrzut odniesienia, nie nasza parametryzacja.

**Fakt rozstrzygający osobno, sprawdzony w pobranych danych:** katalog RIR-ów SoundSpaces 1.0
dla każdej sceny ma **dokładnie cztery podkatalogi orientacji — `0/`, `90/`, `180/`, `270/`**.
SS 1.0 **z definicji nie może dostarczyć 36 orientacji**, więc „odtworzenie ich zbioru" nie
jest celem osiągalnym i nie powinno być stawiane jako kryterium. Jedyną drogą do gęstszej
siatki kątowej jest rendering on-the-fly SS 2.0 — **to jest podstawowe uzasadnienie istnienia
tej pracy** i warto je podać wprost, bo jest sprawdzalne w jednym `ls`.

**Co z tego wynika praktycznie:** SS 1.0 należy przestać traktować jako kryterium poprawności
i używać go tylko do tego, co potrafi udowodnić — że porównanie bezwzględne między silnikami
jest nieuprawnione, więc porównanie 36 vs 4 musi być wewnętrzne. Poprawność łaty trzeba
oceniać kryteriami niezależnymi od SS 1.0: ucieczką promieni (0.00 % po załataniu, §2.13),
zgodnością wysokości pomieszczenia ze scenami szczelnymi (2.71 m wobec 2.64–2.87 m, §2.13)
oraz zgodnością z Sabine/Eyringiem w tym samym pasmie, jakie ustalono dla scen szczelnych
(**1.13–1.75×**, §2.9) — to ostatnie **nie zostało jeszcze zmierzone dla scen załatanych**.

Uboczna obserwacja: RT60 w SS 1.0 jest **płaskie w częstotliwości** bardziej niż nasze —
mediana RT60(2 kHz)/RT60(250 Hz) wynosi 0.85 u nich wobec 0.77 u nas (zakresy 0.78–0.96
i 0.67–0.99). Kierunek jest spójny, ale zakresy się nakładają, więc przy n = 6 to przesłanka,
nie ustalenie.

Dane: `outputs/measurements/cross_engine/*.json` (w gicie). Status SS 1.0: to **baseline, nie
prawda podstawowa** — paper SoundSpaces 2.0 raportuje, że 2.0 zgadza się z pomiarami
rzeczywistymi *lepiej* niż 1.0 (błąd direct-to-reverberant 11.0 dB → 0.98 dB), więc
rozbieżność nie oznacza automatycznie błędu po naszej stronie.

---

### 2.15 Test rozstrzygający poprawność łaty — Sabine/Eyring, bez SS 1.0 **[Z]**

Po §2.14 wiadomo, że SS 1.0 nie może orzec o poprawności łaty (rozrzut ±44 %). Właściwym
kryterium jest **ten sam standard fizyczny, jaki stosujemy do scen o nienaruszonej
geometrii**: zgodność z Eyringiem. Obie strony równania przeliczone niezależnie i **na tej
samej siatce** — geometria (V, S, α) z załatanego PLY, RT60 z renderów tej samej sceny
(`rt60_vs_sabine.py --patched`, 30 renderów, rozgrzewka 50, pasma 500 Hz–2 kHz).

| grupa | n | zmierzone/Eyring — mediana | zakres |
|---|---|---|---|
| sceny szczelne, geometria oryginalna | 7 | **1.27×** | 1.07–1.74× |
| **`frl_apartment_*` ZAŁATANE** | 6 | **1.00×** | **0.92–1.05×** |
| `frl_apartment_*` bez łaty | 6 | **0.41×** | 0.34–0.47× |

**Wynik 1 — łata naprawia realną niefizyczność.** Bez sufitu sceny leżą na 0.41× Eyringa;
zamknięte pomieszczenie *nie może* tracić tyle energii, więc to nie „niezgodność modelu",
tylko jego niestosowalność. Po załataniu wychodzi **1.00×** — pełna zgodność z modelem
dyfuzyjnym. RT60 @ 1 kHz rośnie z 0.195–0.263 s do 0.548–0.633 s, przy Eyringu 0.574–0.614 s.
Objętość rośnie z 180–191 m³ do 201–207 m³ (+6…+12 %), pokrycie sufitem z 4–7 % do 100 %.

**Wynik 2 — ale łata NIE czyni sceny równoważną nienaruszonemu skanowi.** Sceny załatane są
**systematycznie bliżej** Eyringa niż sceny szczelne: rozdzielenie **zupełne**
(min szczelnych 1.07 > max załatanych 1.05), Mann-Whitney **p = 0.0032**. Wyjaśnienie jest
fizyczne i trzeba je podać: moja łata to **idealnie płaska, jednorodna płaszczyzna z jednego
materiału** — dokładnie to, co Sabine i Eyring zakładają. Prawdziwy zeskanowany sufit ma
oprawy, nierówności i kilka materiałów, więc odchyla się od modelu mocniej. Załatane sceny są
więc **bardziej wyidealizowane** niż rzeczywiste, nie „takie same".

**Wynik 3 — kontrola techniczna, która musiała wypaść pomyślnie.** Łata jest teraz
**jednostronna** (patrz niżej), więc istniało ryzyko, że ray tracer RLR odrzuca ściany tylne
inną konwencją niż rasteryzator głębi i „widzi" ją inaczej. Wzrost RT60 2.4–3.2× dowodzi, że
symulator akustyczny ją widzi — a ucieczka promieni 0.00 % dowodzi tego samego dla rasteryzatora.

**Poprawka geometrii wykryta przy tej okazji: łata musi być JEDNOSTRONNA.** Wcześniejsza
wersja zapisywała każdy czworokąt dwukrotnie, w obu nawinięciach, żeby nie zgadywać konwencji
odrzucania ścian tylnych. Akustycznie było to poprawne (dwie pokrywające się ściany zachowują
się jak jedna), ale **pole liczyło się podwójnie**: powierzchnia sufitu w `frl_apartment_2`
wychodziła 145.8 m² zamiast 74.8 m², co zawyżało objętość sceny z 203 do 396 m³ i unieważniało
całe równanie Sabine'a. Wersja jednostronna orientuje normalną w stronę, po której leżą
pozycje agenta, i sprawdza nawinięcie pomiarem na pierwszym czworokącie. Domknięcie po zmianie
potwierdzone na wszystkich 10 scenach (0.00–0.24 % mediany).

**Uzupełnienie do §2.9:** pasmo odniesienia dla scen szczelnych zmierzone teraz na **7 scenach**
(21 punktów scena×pasmo) wynosi **1.27×, zakres 1.02–1.75×**. Zastępuje to wartość z §2.9
(1.32×, zakres 1.13–1.75×, n = 12 z 4 scen) — ta sama wielkość, szersza podstawa. Przy cytowaniu
używać liczby z n = 21; §2.9 zostaje jako zapis wcześniejszego etapu.

> **Sprostowanie.** W pierwszej wersji §2.14 i w tej sekcji pasmo z §2.9 podano jako „1.15–1.41×".
> To jest wiersz JEDNEJ sceny (`office_1`), nie agregat — §2.9 mówi 1.13–1.75×. Poprawione
> 2026-08-01 w obu miejscach.

**Wniosek dla pracy.** Łatanie jest **fizycznie uzasadnione i zweryfikowane niezależnie od
jakiegokolwiek baseline'u**: usuwa niefizyczność (0.41× → 1.00× Eyringa), domyka scenę
(22 % → 0.00 % ucieczki) i zachowuje wysokość pomieszczenia zgodną ze scenami szczelnymi
(2.71 m). Jednocześnie **nie wolno twierdzić**, że załatana scena jest równoważna
nienaruszonej — jest mierzalnie bardziej idealna (p = 0.0032). To wystarczy, by użyć jej jako
**wariantu dodatkowego** datasetu z jawnym opisem, ale nie jako podmianę wariantu głównego.

Logi: `outputs/measurements/sabine/{A_sealed,B_frl_orig,C_frl_patched}.log`, zestawienie
w `outputs/measurements/sabine/summary.txt` (w gicie).

---

### 2.16 Kontrola wygenerowanego datasetu — oba warianty kompletne i spójne **[Z]**

Generacja zakończona 2026-08-01. Formalna walidacja (`--verify`, bez GPU) **28 plików
HDF5: 28 × PASS, 0 × FAIL**.

| | `main` | `patched` |
|---|---|---|
| scen | 18 | 10 |
| próbek | **62 640 / 62 640 (100 %)** | **44 064 / 44 064 (100 %)** |
| rozmiar | 13.3 GiB | 9.4 GiB |
| poniżej `TARGET_SNR = 3.5` | 1 (**0.0016 %**) | 1 (**0.0023 %**) |
| przy `N_MAX = 64` | 6 (0.010 %) | 43 (0.098 %) |
| mediana `N` po scenach | 12.0 | 11.5 |
| tempo | 0.1329 s/render | 0.1472 s/render |

Kategoria z §5 ogr. 6 (próbka przy `N_MAX`, która nie dobiła progu SNR) wystąpiła
**raz w każdym wariancie** — do wypunktowania w pracy, ale skala jest pomijalna.

**Zgodność wariantów tam, gdzie MUSI być** (`measurements/compare_variants.py`,
10 scen wspólnych): siatka próbkowania (`location_id`, `angle_deg`, `position`)
**identyczna we wszystkich scenach**, 19 sprawdzanych atrybutów konfiguracji
(wysokość słuchacza i kamery, HFOV, rozdzielczość, liczba promieni, potok
spektrogramu, sumy kontrolne chirpa i configu materiałów) **identycznych**,
atrybut `variant` poprawnie różny. Wariancie różnią się więc **wyłącznie geometrią**,
zgodnie z założeniem.

**Różnice, które mają być — i ich wielkość:**

| wielkość | patched / main | interpretacja |
|---|---|---|
| komórki `echo` różne | **82.5 %** | domknięcie zmienia pole akustyczne — o to chodzi |
| energia echa | **1.14×** | więcej pogłosu, kierunek zgodny z §2.15 |
| `sigma_1` | **1.24×** | więcej odbić stochastycznych = wyższy szum MC |
| `N` (reguła adaptacyjna) | **1.40×** | poprawna reakcja na wyższy szum, nie błąd |
| piksele `rgb` różne | **12.6 %** | **sufit JEST widoczny dla kamery** |
| piksele `depth` różne | **16.9 %** | jw. |

Rozrzut po scenach jest zgodny z typem łaty: `frl_apartment_*` (dołożony cały sufit)
zmieniają 12.4–14.4 % pikseli RGB i mają `N` do 1.67×, a `office_3` (samo okno i drzwi)
tylko 1.0 % pikseli i `N` 1.12×.

**ZASTRZEŻENIE, które musi trafić do pracy — wariant `patched` zmienia także CEL
predykcji.** Dołożony sufit widać w `depth`, a `depth` jest wielkością przewidywaną.
Wariantów **nie wolno więc porównywać po bezwzględnym błędzie modelu** — model uczony
na `patched` przewiduje inną wielkość niż model uczony na `main`. Poprawne porównanie to
**przewaga 36 orientacji nad 4 WEWNĄTRZ każdego wariantu**, a następnie zestawienie tych
dwóch przewag. Dodatkowo tylko `main` zachowuje zgodność obrazu z VisualEchoes, więc
wszelkie odniesienia do pracy źródłowej wolno robić wyłącznie na nim.

Dane: `outputs/measurements/dataset_check/` (werdykty `--verify` i
`variant_comparison.json`, w gicie).

---

### 2.17 Skąd `TARGET_SNR = 3.5` — mechanizm udowodniony, sama wartość przyjęta **[Z / Z-]**

Pytanie, które zada recenzent i na które trzeba mieć gotową odpowiedź. Rozbija się na dwie
części o **różnym statusie dowodowym** i nie wolno ich mieszać.

**Co ten próg znaczy.** `SIGNAL_10DEG = 0.0644` to RMSE między spektrogramami oddalonymi
o 10°, czyli **najmniejsza różnica, jaką zbiór ma w ogóle reprezentować** — krok siatki
36 orientacji. `sigma_N = sigma_1/sqrt(N)` to resztkowy szum Monte Carlo po uśrednieniu.
Warunek `SIGNAL_10DEG / sigma_N >= 3.5` mówi: *różnica między sąsiednimi orientacjami ma
być 3.5× większa niż szum renderowania*. Stąd `N >= (3.5 · sigma_1 / 0.0644)^2`.

**[Z] Że próg musi leżeć wyraźnie powyżej 1 — udowodnione pomiarem.** `sigma_1` na
lokalizację wynosi **0.0258–0.1327** (`main`) i **0.0383–0.1506** (`patched`) przy sygnale
0.0644. Przy **pojedynczym renderze** szum w najgorszych lokalizacjach jest więc **2.06× /
2.34× większy od mierzonego sygnału**. Bez uśredniania sąsiednie orientacje byłyby
nierozróżnialne, a zbiór 36-orientacyjny niósłby w dużej części szum zamiast informacji
kątowej. To jest realna przesłanka istnienia całej reguły adaptacyjnej — **jakieś**
uśrednianie jest konieczne, nie opcjonalne.

**[Z-] Że akurat 3.5 — BRAK WYPROWADZENIA.** Przeszukano `GENERATOR_PARAMS.md`,
`OBSERWACJE_METODOLOGICZNE.md`, ten raport oraz kod (`echo_core/`, `diagnostics/`):
wartość jest **przyjęta, nie policzona**. Dokument definiuje względem niej wszystko inne
(progi `N_MAX`, budżet, sformułowanie z §3.5 GENERATOR_PARAMS), ale samej liczby nigdzie
nie uzasadnia. **W pracy opisać jako wybór konserwatywny, nie jako wynik.** Podawanie jej
jako wielkości wyprowadzonej byłoby nadużyciem.

**Koszt alternatyw — policzony na gotowym zbiorze** (`N ~ SNR^2`):

| `TARGET_SNR` | średnie `N` (`main`) | względem 3.5 |
|---|---|---|
| 2.0 | 6.35 | 0.52× |
| **3.5** | **12.13** | **1.00×** |
| 5.0 | 23.40 | 1.93× |

**Jak wyszło w praktyce.** `snr_final` po dorenderowaniu: mediana **3.72** (`main`) /
**3.66** (`patched`), **79 % / 94 % próbek w przedziale 3.5–4.0**, **0.0 % powyżej 6**.
Reguła celuje dokładnie w próg i go nie przekracza — zamierzone, ale ma konsekwencję
praktyczną: **zbiór nie ma zapasu**. Obniżenie progu da się zrobić odrzuceniem renderów,
ale **podniesienie wymaga regeneracji**, bo dodatkowych renderów po prostu nie zapisano.

**Zastrzeżenie do samego pomiaru:** `snr_final` liczy ten sam estymator połówkowy, który ma
sufit dokładności 4–6 % (§2.6). „≥ 3.5" znaczy więc „≥ 3.5 według estymatora o ~5 %
rozrzutu" — stąd pojedyncze próbki z `snr_final` 3.43 (`main`) i 3.12 (`patched`).

Pełne omówienie wraz z tabelami: `GENERATOR_PARAMS.md` §3.2 („Skąd `TARGET_SNR = 3.5`").

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
| **warianty datasetu** | `main` + `patched` | §2.13–§2.16 **[Z]**, GENERATOR_PARAMS §4.5 |
| **zakres wariantu `patched`** | tylko 10 scen z łatą | sceny szczelne mają tę samą siatkę → echa byłyby identyczne |
| kolejka scen w `echo_ctl.py` | osobny proces-nadzorca | narzędzie, nie pomiar |

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
| **Pasmo Eyringa z §2.9 zacytowane jako „1.15–1.41×"** w §2.14 i §2.15 — to jest wiersz JEDNEJ sceny (`office_1`), nie agregat; §2.9 mówi 1.32×, zakres 1.13–1.75× (n = 12) | Audyt spójności liczb przed commitem (2026-08-01) | **Skorygowany** w obu miejscach, sprostowanie zapisane przy §2.15 |
| **Łata dziur zapisana jako zdegenerowane czworokąty** — nie dają powierzchni w tym potoku | Ucieczka promieni spadła 21.79 → 20.26 % zamiast do zera | Naprawione: siatka pełnych czworokątów, §2.13 |
| **Znak normalnej z SVD przyjęty bez sprawdzenia** — łata wyszła zwrócona w górę i renderer odrzucał ją jako tylną, przy poprawnej geometrii w pliku | Ucieczka promieni **nie zmieniła się ani o jotę** (21.79 % przed i po) | Naprawione: orientacja wg pozycji agenta + kontrola nawinięcia, §2.13 |
| **Łata dwustronna liczyła pole podwójnie** (145.8 m² zamiast 74.8 m²), zawyżając objętość sceny z 203 do 396 m³ | Kontrola geometrii przed testem Sabine'a | Naprawione: łata jednostronna, §2.15 |

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
| Ucieczka promieni, 1740 lokalizacji × 2 warianty | `outputs/measurements/ray_escape/*.csv` |
| Zgodność międzysilnikowa SS 2.0 vs 1.0 (6 scen szczelnych) | `outputs/measurements/cross_engine/summary.json` |
| Test fizyczny Sabine/Eyring, trzy przebiegi | `outputs/measurements/sabine/summary.txt` |
| Eksperyment z łatą: RT60 i sweep materiałowy | `outputs/measurements/ceiling_patch/*.json` |
| Kontrola gotowego datasetu (28 × `--verify` + porównanie wariantów) | `outputs/measurements/dataset_check/` |
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
