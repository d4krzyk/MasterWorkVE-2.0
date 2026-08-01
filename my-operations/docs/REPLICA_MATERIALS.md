# Materiały akustyczne dla scen Replica

Dokumentacja `my-operations/replica_material_config.json` — mapowania kategorii semantycznych
Repliki na materiały akustyczne RLRAudioPropagation. Plik generowany przez
`my-operations/make_replica_material_config.py`; inwentarz powierzchni liczy
`my-operations/replica_semantic_area.py` (wynik: `my-operations/replica_category_area.json`).

Stan przed zmianą: na scenach Repliki używaliśmy `sound-spaces/data/mp3d_material_config.json`,
czyli mapowania napisanego dla Matterport3D.

---

## 1. Jak RLRAudioPropagation faktycznie dopasowuje materiał

To nie jest dopasowanie przez równość napisów. Z `RLRAudioPropagation.h`, opis
`RLRA_SetMaterialDatabaseJSON()`:

> A material is determined from a material category string by inspecting all materials in the
> database, and finding the material which has the **greatest number of label substring matches**.
> A match is counted if the lowercase category name **contains a label as a substring**.
>
> NOTE: the material `"name"` attribute is not used for matching (…) and will be ignored.

Trzy konsekwencje, o które łatwo się potknąć:

1. **Nazwa materiału jest nieistotna** — liczy się wyłącznie lista `labels`.
2. **Krótka etykieta łapie dłuższe kategorie.** Realne kolizje na Replice:
   `wall` ⊂ `wall-cabinet`, `wall-plug`; `bin` ⊂ `ca**bin**et`; `table` ⊂ `tablet`;
   `pan` ⊂ `panel`; `door` ⊂ `in**door**-plant`.
3. **Przy remisie liczby dopasowań API nie określa zwycięzcy** — wynik jest niezdefiniowany.

Kolizje rozwiązujemy przez **powielenie etykiety** będącej dosłowną nazwą kategorii (duplikaty
liczą się podwójnie). Zasada: *dosłowna nazwa kategorii zawsze bije przypadkowy podłańcuch*.
To nie jest obejście — tej samej sztuczki użyto w oryginalnym `mp3d_material_config.json`, gdzie
materiał `Carpet` ma `"floor"` wpisane **dwa razy**. `make_replica_material_config.py` wykrywa
i naprawia kolizje automatycznie, przechodząc regułę po wszystkich kategoriach faktycznie
występujących w 18 scenach.

Automatycznie rozwiązane kolizje (5):

| kategoria | trafia do | kolidowało z | waga etykiety |
|---|---|---|---|
| `cabinet` | wood, Thick | wood, Thin (przez `bin`) | 2 |
| `indoor-plant` | Foliage | wood, Thick (przez `door`) | 2 |
| `panel` | wood, Thin | Steel (przez `pan`) | 2 |
| `tablet` | wood, Thin | wood, Thick (przez `table`) | 2 |
| `wall-plug` | wood, Thin | Gypsum Board (przez `wall`) | 2 |

**Weryfikacja modelu reguły.** Symulacja dopasowania podłańcuchowego przewiduje dla `room_0`
z configiem mp3d dokładnie 11 kategorii bez materiału: `basket, book, candle, lamp, picture,
pillar, plate, pot, switch, vase, vent` — i to jest **co do jednej** ta sama lista, którą
habitat-sim wypisuje w logu jako `Material for category 'X' was not found`. Model jest poprawny.
(Dla kontrastu: naiwne dopasowanie przez równość napisów przewidziałoby błędnie, że `plant-stand`
i `wall-plug` też są bez materiału — a one dopasowują się przez podłańcuch.)

---

## 2. Dlaczego powierzchnia, a nie liczba obiektów

Materiał jest przypisywany **per trójkąt** siatki semantycznej
(`AudioSensor::loadSemanticMesh()`), a wpływ materiału na pogłos jest proporcjonalny do
**powierzchni**. 255 książek w 10 scenach to akustycznie 9.0 m² (0.20 % sceny); 236 ścian to
1341 m² (29 %). Liczba obiektów jest więc myląca jako waga przy decydowaniu, co warto mapować
starannie. Nie wystarczy też pole bounding-boxa — dla cienkiej ściany bbox zawyża, dla wklęsłych
mebli zaniża.

`replica_semantic_area.py` liczy **rzeczywistą powierzchnię trójkątów** wprost z
`mesh_semantic.ply` (format Repliki: `element face` ma listę `uint32` o stałej długości 4 — same
quady — plus `object_id` typu `uint16`, więc rekord ma zawsze 19 bajtów i da się go wczytać jednym
`np.frombuffer`).

Łącznie 18 scen, **4611 m²**, 87 kategorii. Rozkład jest skrajnie nierówny — 10 kategorii pokrywa
77 % powierzchni:

| kategoria | % pola | skum. |
|---|---|---|
| wall | 29.09 | 29.1 |
| floor | 13.40 | 42.5 |
| ceiling | 11.14 | 53.6 |
| blinds | 5.01 | 58.6 |
| door | 3.73 | 62.4 |
| chair | 3.37 | 65.7 |
| table | 3.30 | 69.1 |
| rug | 3.14 | 72.2 |
| stair | 2.23 | 74.4 |
| sofa | 2.19 | 76.6 |

---

## 3. Stan wyjściowy: co naprawdę robił config mp3d na Replice

**Roboczą hipotezę, że „większość kategorii Repliki po cichu spada na materiał domyślny", trzeba
odrzucić.** Po policzeniu regułą podłańcuchową i po powierzchni:

| | kategorii | % powierzchni |
|---|---|---|
| jednoznacznie dopasowane | 43 | 88.6 % |
| niejednoznaczne (remis, wynik nieokreślony) | 1 (`wall-cabinet`) | 1.9 % |
| materiał domyślny | 43 | 9.0 % |

Nazwy kategorii Repliki w dużej mierze **pokrywają się** ze słownikiem etykiet mp3d (`wall`,
`floor`, `ceiling`, `door`, `chair`, `table`, `rug`, `window`, `sofa`, `bed`, …), więc pokrycie
było dobre. Wrażenie „materiały są fikcyjne" brało się z liczby **linii ostrzeżeń** w logu, a te
liczą kategorie, nie powierzchnię — 43 nietrafione kategorie to 43 komunikaty, ale tylko 9 %
sceny.

**Prawdziwy problem leży gdzie indziej: w samych przypisaniach.** Dwa z nich są rażąco błędne dla
Repliki i dotyczą 24.5 % całej powierzchni:

### 3.1 `ceiling → Acoustic Tile` (11.14 % powierzchni)

`Acoustic Tile` to podwieszany sufit biurowy z płyt mineralnych: absorpcja **0.50 / 0.70 / 0.60 /
0.70 / 0.70 / 0.50** (125 Hz → 4 kHz). To najsilniej pochłaniający materiał w całej bazie poza
`Sound Proof`, `Snow` i `Grass`.

Oględziny górnego pasa kadru (18 rzędów pikseli od góry, po 8 losowych lokalizacji na scenę, z
`scene_observations_128.pkl`) we **wszystkich 18 scenach** pokazują gładki, malowany, biały sufit —
także w `office_0..4`. Nigdzie nie ma rastru płyt akustycznych. Czarne obszary w
`frl_apartment_*` to dziury w rekonstrukcji skanu, nie ciemne płyty.

→ **`ceiling → Gypsum Board`** (0.29 / 0.10 / 0.05 / 0.04 / 0.07 / 0.09). Przy 500 Hz to
**dwunastokrotnie** mniejsza absorpcja niż dotąd.

### 3.2 `floor → Carpet` (13.40 % powierzchni)

Dwa niezależne argumenty:

1. **Materiał widoczny na obrazie.** Dolny pas kadru pokazuje w rodzinie apartamentów
   (`apartment_0..2` + `frl_apartment_0..5`) jasne deski drewniane z widocznymi słojami. Te
   9 scen to **72 % całej powierzchni Repliki**. Szare podłogi widać w `office_*`, `room_0`,
   `room_2` — ale jedno globalne przypisanie musi iść za dominującą powierzchnią.
2. **Argument niezależny od oględzin, mocniejszy:** miękkie pokrycie podłogi jest w Replice
   modelowane **osobno**, jako kategorie `rug` (144.7 m², 13 z 18 scen) i `mat` (7.8 m²).
   Przypisanie samej podłogi do dywanu liczyłoby tę absorpcję **drugi raz** — dywan leżałby na
   dywanie.

→ **`floor → Wood Floor`** (0.15 / 0.11 / 0.10 / 0.07 / 0.06 / 0.07). Przy 4 kHz to
**dziewięciokrotnie** mniejsza absorpcja niż `Carpet` (0.65).

### 3.3 `blinds → Glass` (5.01 % powierzchni)

Rolety i żaluzje to tkanina albo lamele, nie szyba; `Glass` przy 4 kHz odbija praktycznie
wszystko (absorpcja 0.05). To trzecia co do wielkości kategoria po ścianach/podłodze/suficie.

→ **`blinds → Curtain`** (0.07 / 0.31 / 0.49 / 0.75 / 0.70 / 0.60).

### 3.4 Drobniejsze poprawki

- `plant-stand` łapało się na `Foliage` przez podłańcuch `plant` — stojak na kwiaty jest drewniany,
  nie liściasty → `wood, Thick`.
- `wall-cabinet` (1.88 %) miało **remis** `Gypsum Board` vs `Wood Floor`, czyli wynik nieokreślony
  → jednoznacznie `wood, Thick`.
- `cabinet` / `base-cabinet` przechodzą z `Wood Floor` na `wood, Thick` — szafka to lita bryła
  mebla, nie podłoga (różnica niewielka, ale spójność opisu ma znaczenie w pracy).
- Dodane kategorie, które wcześniej szły na domyślny: `lamp`, `tv-screen`, `refrigerator`,
  `pillar`, `bike`, `tv-stand`, `pillow`, `rack`, `bin`, `picture`, `book`, `switch`, `bench`,
  `vent`, `panel`, `basket` i ~27 drobniejszych.

---

## 4. Pełne mapowanie

Bloki fizyczne materiałów (`absorption` / `scattering` / `transmission` / `damping` / `density` /
`speed`) są kopiowane **1:1** z `mp3d_material_config.json` — to baza dostarczona przez autorów
RLRAudioPropagation i nie mamy podstaw jej zmieniać. Zmieniamy wyłącznie `labels`.

Wykorzystane materiały (absorpcja 125 Hz → 4 kHz):

| materiał | 125 | 250 | 500 | 1k | 2k | 4k | kategorie |
|---|---|---|---|---|---|---|---|
| Gypsum Board | 0.29 | 0.10 | 0.05 | 0.04 | 0.07 | 0.09 | `wall`, `ceiling` |
| Wood Floor | 0.15 | 0.11 | 0.10 | 0.07 | 0.06 | 0.07 | `floor`, `stair` |
| Carpet, Heavy | 0.02 | 0.06 | 0.14 | 0.37 | 0.48 | 0.63 | `rug` |
| Carpet | 0.01 | 0.05 | 0.10 | 0.20 | 0.45 | 0.65 | `mat` |
| Curtain | 0.07 | 0.31 | 0.49 | 0.75 | 0.70 | 0.60 | `curtain`, `blinds`, `sofa`, `bed`, `comforter`, `blanket`, `pillow`, `cushion`, `beanbag`, `clothing`, `cloth`, `scarf`, `towel`, `handbag`, `bag`, `shoe`, `umbrella`, `lamp` |
| wood, Thick | 0.19 | 0.14 | 0.09 | 0.06 | 0.06 | 0.05 | `door`, `chair`, `table`, `desk`, `shelf`, `cabinet`, `wall-cabinet`, `base-cabinet`, `nightstand`, `countertop`, `stool`, `bench`, `tv-stand`, `plant-stand`, `rack`, `book`, `chopping-board`, `desk-organizer`, `utensil-holder`, `knife-block` |
| wood, Thin | 0.42 | 0.21 | 0.10 | 0.08 | 0.06 | 0.06 | `bin`, `box`, `basket`, `panel`, `switch`, `wall-plug`, `tissue-paper`, `coaster`, `tablet`, `remote-control`, `clock`, `camera` |
| Glass | 0.35 | 0.25 | 0.18 | 0.12 | 0.07 | 0.05 | `window`, `tv-screen`, `monitor`, `picture`, `bottle` |
| Steel | 0.05 | 0.10 | 0.10 | 0.10 | 0.07 | 0.02 | `handrail`, `refrigerator`, `sink`, `faucet`, `pipe`, `vent`, `bike`, `cooktop`, `pan`, `major-appliance`, `small-appliance`, `kitchen-utensil` |
| Tile, Ceramic | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | `shower-stall`, `toilet`, `bathtub`, `bowl`, `plate`, `cup`, `vase`, `pot`, `sculpture`, `candle` |
| Foliage | 0.03 | 0.06 | 0.11 | 0.17 | 0.27 | 0.31 | `indoor-plant` |
| Concrete | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | `pillar` |

Uzasadnienia grup, które nie są oczywiste:

- **`wood, Thin` jako „cienka pusta skorupa"** — plastikowy kosz, karton, wiklinowy koszyk,
  panel, klawisz włącznika. Akustycznie zachowują się jak płyta rezonansowa: wysoka absorpcja
  nisko (0.42 przy 125 Hz), niska wysoko. To fizycznie inne zachowanie niż lita bryła
  (`wood, Thick`: 0.19 przy 125 Hz) i lepiej pasuje do pustych obiektów.
- **`Curtain` jako jedyny materiał porowaty** — w bazie RLR nie ma „tapicerki". `Curtain` jest
  jedynym materiałem o wysokiej absorpcji średnich i wysokich częstotliwości, który nie jest
  dywanem, więc obsługuje wszystkie tkaniny miękkie. Oryginalny mp3d robi dokładnie to samo.
- **`Tile, Ceramic` dla naczyń** — twarda, gładka ceramika; te same własności co płytka.

### Przypisania niskiej pewności (do wypunktowania w pracy jako ograniczenie)

- **`lamp → Curtain` (1.34 % powierzchni)** — 135 obiektów `lamp` to mieszanka abażurów
  tkaninowych i opraw metalowo-szklanych. Abażur jako porowata tkanina jest rozsądnym przybliżeniem
  dla większości, ale dla opraw sufitowych jest zawyżeniem absorpcji. To największa pozycja o
  niskiej pewności.
- **`countertop → wood, Thick`** — blaty w kuchniach FRL wyglądają na kamień/laminat, twardsze niż
  drewno. Zachowane za mp3d; 0.40 % powierzchni, więc skutek pomijalny.
- **`book → wood, Thick`** — stosy papieru na półkach absorbują więcej niż lite drewno w
  średnich pasmach. 0.20 % powierzchni.

---

## 5. Czego nie da się naprawić configiem

**2.09 % powierzchni (491 obiektów) ma `class_id: -1`** i **zawsze** dostanie materiał domyślny,
niezależnie od zawartości JSON-a. Te obiekty mają w habitat-sim niezerowy `SemanticObject`, ale
**null `category()`** (to jest źródło SIGSEGV-a naprawionego lokalną modyfikacją (4) — patrz
`habitat-sim/local_changes.patch`), więc do RLR nie trafia żadna nazwa kategorii.

Rozkład jest bardzo nierówny między scenami — od 0.48 % (`frl_apartment_4`) do **11.53 %**
(`office_0`), 9.35 % (`office_4`), 8.20 % (`office_1`), 5.12 % (`hotel_0`). W scenach biurowych
to realny, niedający się usunąć błąd modelowania akustycznego.

Dodatkowo 0.06 % trójkątów ma `object_id`, którego nie ma w `info_semantic.json` — również
materiał domyślny.

**Osiągalne pokrycie: 97.86 %** powierzchni; 2.14 % na materiale domyślnym, całość z powyższych
dwóch przyczyn strukturalnych. Zero kategorii spada na domyślny z powodu braku mapowania.

---

## 6. Możliwe rozszerzenie: config per scena

`setAudioMaterialsJSON()` jest ustawiane na instancji `AudioSensor`, a ustalona architektura
generatora i tak buduje **jeden Simulator na scenę** (osobny proces OS na scenę — patrz
`GENERATOR_PARAMS.md` §4). Config per scena **nie kosztuje więc nic**.

Jedyna kategoria, dla której ma to realne znaczenie, to `floor`: podłogi drewniane w rodzinie
apartamentów kontra szare (dywan albo beton) w `office_0..4`, `room_0`, `room_2`. Nie zrobiono
tego w tej iteracji — globalny config jest prostszy do opisania w pracy, a różnica dotyczy
mniejszościowej części powierzchni. Gdyby wprowadzać: wystarczy wygenerować wariant z
`floor → Carpet` i wskazywać go dla tych 7 scen.

---

## 7. Weryfikacja (wykonana 2026-07-26)

`--exp materials_verify` w `my-operations/diagnose_rlr_noise.py`; wyniki w
`diagnostics_report.json` pod kluczem `materials_verify`. 3 pozycje × 2 kąty × 16 renderów na
config, `indirectRayCount=500`, `threadCount=1`.

Ostrzeżenia warstwy C++ są liczone przez przechwycenie **deskryptorów 1/2** (`os.dup2`), a nie
`sys.stdout` — habitat-sim i RLR piszą prosto do deskryptorów, więc przekierowanie w Pythonie by
ich nie złapało.

### 7.1 Pokrycie

| scena | mp3d | replica |
|---|---|---|
| `room_0` | 11 kategorii na domyślnym | **0** |
| `office_0` | 10 kategorii na domyślnym | **0** |

Obiekty z `class_id: -1` **nie generują ostrzeżenia** — do RLR nie trafia żadna nazwa kategorii,
więc biblioteka nie ma czego szukać. Ich udział (2.09 % powierzchni) trzeba liczyć osobno,
z `info_semantic.json`; log ich nie pokaże.

### 7.2 Kontrola pozytywna — czy config w ogóle działa

| scena | szum estymaty (16 renderów) | efekt replica vs mp3d | energia |
|---|---|---|---|
| `room_0` | 0.01269 RMSE / 0.714 % | **0.04650** (3.7× szum) | **+10.23 %** (14× szum energii) |
| `office_0` | 0.01757 RMSE / 0.519 % | **0.02748** (1.6× szum) | **+7.45 %** (14× szum energii) |

Uwaga metodologiczna: surowe `RMSE(replica, mp3d)` (0.0498 / 0.0371) **nie jest** efektem — zawiera
szum obu estymat. Efekt to `√(RMSE² − 2σ²)`, ta sama dekompozycja co w E2b. Pierwsza wersja tego
testu porównywała surowe RMSE z `RMSE(A,B)` i przez to wypisała fałszywy werdykt „config nie
działa"; próg został poprawiony.

**Statystyką rozstrzygającą jest energia, nie RMSE.** Energia to średnia po ~85 tys. komórek
spektrogramu, więc jej szum jest o rzędy wielkości mniejszy. Przesunięcie **+10.2 % i +7.4 %, tego
samego znaku w obu scenach, 14× ponad własny szum** — i dokładnie w kierunku przewidzianym:
utwardzenie sufitu (0.60 → 0.05 przy 500 Hz) i podłogi (0.65 → 0.07 przy 4 kHz) musi podnieść
energię echa.

Dla skali: efekt zmiany materiałów w `room_0` (0.0465) jest porównywalny z sygnałem obrotu o 10°
(0.0648). Wybór materiałów nie jest kosmetyką.

### 7.3 Kontrola negatywna

Config „wszystko Sound Proof" (absorpcja 1.0 na każdej częstotliwości), budowany z configu Repliki
tak, by różnić się **wyłącznie fizyką materiału**, a nie zestawem etykiet — plik tymczasowy,
kasowany po teście, niecommitowany.

| scena | efekt vs replica | energia |
|---|---|---|
| `room_0` | 0.13730 (10.8× szum) | **−67.5 %** |
| `office_0` | 0.16508 (9.4× szum) | **−65.8 %** |

Ścieżka materiałów faktycznie dociera do symulatora — nie tylko „nie krzyczy".

**Werdykt: OK, nowe mapowanie działa.**

---

## 8. Wpływ na charakterystykę szumu (Blok C, `--exp signal_noise_recheck`)

Cała wcześniejsza charakteryzacja szumu szła na configu mp3d. Twardsze materiały wydłużają pogłos,
a odbicia pośrednie są jedynym źródłem szumu Monte Carlo, więc trzeba było sprawdzić, czy sygnał i
szum się nie przesunęły. 4 pozycje (`room_0` 30/50/80, `office_0` 30), N=10 na połówkę.

| | mp3d | replica | zmiana |
|---|---|---|---|
| sygnał 10° (odszumiony) | 0.06460 | 0.06480 | **+0.3 %** |
| szum pojedynczego renderu | 0.05320 | 0.05393 | +1.4 % |
| szum estymaty przy N=10 | 0.02379 | 0.02412 | +1.4 % |
| SNR przy N=10 | 3.85 | 3.86 | +0.3 % |
| energia | 0.1238 | 0.1354 | +9.4 % |

**Sygnał i szum się nie przesunęły.** Historyczna liczba 0.064 i SNR ≈ 3–4 przy N=10 obowiązują
dalej. Energia rośnie o 9.4 %, spójnie z pomiarem z 7.2.

Jedyny wyjątek jest lokalny: w `office_0` szum pojedynczego renderu wzrósł z 0.0573 do 0.0634
(+11 %), przez co wymagane N dla SNR 3 poszło tam z 7 na 9. W `room_0` wymagane N **spadło**
(7/7/6 → 7/6/5). Interpretacja: w małym pokoju biurowym utwardzenie powierzchni wydłuża ogon
pogłosu bardziej względnie, więc w oknie 60 ms ląduje więcej stochastycznych odbić. Automatyczny
werdykt skryptu wypisał „PRZESUNIĘTE", ale liczy maksimum po pozycjach i jest zdominowany przez ten
jeden punkt — średnie się nie ruszyły.

Wszystko powyżej mierzone przy `indirectRayCount=500`. Produkcja ma iść na 5000–10000, gdzie szum
Monte Carlo jest niższy, więc wyliczone N jest **górnym ograniczeniem**, nie wartością docelową.
