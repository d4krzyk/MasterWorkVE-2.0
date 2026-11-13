# Obserwacje metodologiczne — konsekwencje dla pracy

Dokument pomocniczy do pisania rozdziału metodologicznego. Zbiera te ustalenia z sesji
2026-07-26 → 29, które **mają wpływ na sposób prowadzenia i opisania badania**, a nie tylko
na parametry generatora. Liczby i dowody: `RAPORT_SESJI_2026-07-26_29.md`, skrypty
w `my-operations/measurements/`.

---

## 1. Sześć scen Replica nie ma sufitu — i co z tego wynika

### Fakt

`frl_apartment_0..5` mają pokrycie sufitem **5–7 %** rzutu poziomego; pozostałe 12 scen
87–100 %. Skoro sufitu nie ma w siatce, nie ma go w symulacji akustycznej — energia ucieka
górą. Skutek jest mierzalny i systematyczny:

| grupa | scen | lokalizacji | mediana `sigma_1` | zakres |
|---|---|---|---|---|
| otwarte | 6 | **807 (46.4 %)** | 0.04570 | 0.04136–0.04892 |
| zamknięte | 12 | 933 (53.6 %) | 0.06322 | 0.05648–0.08830 |

Rozdzielenie **zupełne** (`max(otwarte) < min(zamknięte)`), Mann-Whitney **p = 0.00005**.

### Dlaczego skany FRL nie mają sufitu — co mówią źródła (sprawdzone 2026-07-30)

Do pracy potrzebne jest zdanie o przyczynie, a nie tylko o objawie. Ustalenia z literatury:

**1. Paper Repliki opisuje potok domykania dziur — i sufit poza niego wypada.**
[The Replica Dataset (arXiv 1906.05797)](https://arxiv.org/abs/1906.05797) opisuje automatyczne
wykrywanie dziur („searching for boundary edges that form closed cycles") i wypełnianie metodą
Liepy, a następnie stwierdza: *„To ensure the highest quality 3D meshes, we manually fix planar
reflective surfaces and **small holes** where surfaces were not sufficiently captured during
scanning."* Naprawiano **małe** dziury; brakujący sufit to kilkadziesiąt m² jednej płaszczyzny,
więc wypada poza to, co ten etap obejmował. **Nigdzie nie ma zdania, że sufity usunięto celowo.**

**2. Nasz pomiar wyklucza „ucięcie" na rzecz „niezeskanowania".** Ściany sięgają pełnej
wysokości, a ocalały fragment klasy `ceiling` leży na prawdziwej wysokości sufitu (wysokość
pomieszczenia 2.69–2.74 m, zgodna z 2.64–2.87 m w scenach szczelnych — raport §2.13). To jest
sygnatura powierzchni, której skaner nie złapał, a nie przyciętej sceny.

**3. To znany tryb awarii całej klasy datasetów, nie osobliwość Repliki.** Literatura
akustyczna opisuje analogiczny problem w ScanNecie (brak sufitu → ucieczka promieni →
nierealistyczny pogłos przy ray tracingu). **SoundSpaces 2.0 — silnik, którego używamy —
dokumentuje to wprost**: *„the scene meshes need to have high quality, i.e., no large open
holes on the mesh, otherwise the rays will leak from the holes, resulting in inaccurate
simulation"*, i przypisuje temu konkretny artefakt we własnych danych (skrzywiony rozkład
RT60 dla Matterport3D). **Nasza obserwacja odtwarza więc niezależnie udokumentowany efekt** —
to mocniejsze sformułowanie niż „u nas wyszło dziwnie".

**Czego NIE wolno napisać:** że sufity usunięto celowo albo że Replica dostarcza wersję
zamkniętą. Żadne źródło tego nie stwierdza, a issue w repozytorium Repliki opisującego to
zjawisko nie znaleziono. Przyczyna „powierzchnia nieuchwycona przez skaner, zbyt duża dla
etapu naprawy małych dziur" jest **wnioskiem zgodnym ze źródłami i z pomiarem**, ale nie
cytatem.

### Czy to psuje metodologię? Nie — i warto wiedzieć dlaczego

**Badanie porównuje 36 orientacji z 4 wewnątrz tego samego zbioru**, przy czym warunek
4-orientacyjny powstaje z **tych samych renderów** (kąty 0/90/180/270 leżą w siatce co 10°).
Własność sceny — otwarta czy zamknięta — działa więc **identycznie na oba warunki** i nie
może wygenerować różnicy między nimi. To jest zmienna wspólna, nie zakłócająca porównanie.

Gdyby warunki pochodziły z osobnych generacji albo z różnych podzbiorów scen, byłby to
poważny problem. W tym projekcie nie jest.

### Co to natomiast zmienia — trzy rzeczy

**(a) Zbiór jest dwumodalny akustycznie i trzeba to napisać.**
Nie „sceny Replica", tylko „12 scen zamkniętych i 6 otwartych". Przy 46 % lokalizacji
w scenach otwartych przemilczenie tego byłoby przemilczeniem połowy zbioru.

**(b) Podział train/test miesza oba typy — w proporcjach zbliżonych do całości.**

| podzbiór | scen | lokalizacji | otwarte |
|---|---|---|---|
| cały zbiór | 18 | 1740 | 46.4 % |
| treningowy | 15 | 1374 | 48.0 % |
| **held-out** | 3 | 366 | **40.4 %** (`frl_apartment_5`) |

Proporcje są zbliżone, więc podział nie wprowadza systematycznego przesunięcia. **To jest
argument, który warto podać jawnie** — sam fakt, że jedna z trzech scen testowych jest
otwarta, mógłby budzić wątpliwość, dopóki nie pokaże się, że odpowiada to składowi całości.

**(c) Otwiera naturalną analizę warstwową — i to jest szansa, nie problem.**
Skoro istnieją dwie populacje o różnym charakterze pogłosu, można zapytać: **czy przewaga
36 orientacji nad 4 jest taka sama w obu?** Hipoteza kierunkowa: informacja kierunkowa
w echu pochodzi głównie z wczesnych odbić geometrycznych, a nie z rozmytego ogona
pogłosowego, więc przewaga powinna utrzymać się w scenach otwartych. Jeśli wynik będzie
odwrotny, powie to coś istotnego o tym, skąd model bierze sygnał.

To jest **darmowa analiza** — nie wymaga żadnych dodatkowych danych, tylko podziału wyników
według listy sześciu scen.

### Czego NIE robić

Nie „naprawiać" scen przez dodanie sufitu. Zbiór lokalizacji i geometria pochodzą
z VisualEchoes (Gao i in., ECCV 2020); zmiana geometrii zerwałaby porównywalność z pracą
źródłową, a zyskałaby tylko zgodność z modelem Sabine'a, który i tak nie jest tu kryterium.

> **Uwaga — to nie jest ostatnie słowo w tej sprawie.** Argument „zyskałaby tylko zgodność
> z Sabine'em" był prawdziwy, dopóki nie porównano z SoundSpaces 1.0 (niżej): stawką jest
> też zgodność z baseline'ami literatury. Techniczna wykonalność i realny koszt naprawy są
> omówione w „Uściśleniu 2026-07-29" na końcu tej sekcji — rekomendacja się nie zmienia,
> ale uzasadnienie jest inne i mocniejsze.

### Czy SoundSpaces 1.0 miał ten sam problem? (sprawdzone 2026-07-29)

Pytanie jest istotne: jeśli tak, otwartość scen jest **własnością dzieloną z literaturą**
(Gao i in. używali tych samych scen), a nie osobliwością tej pracy — wtedy z ograniczenia
staje się przypisem.

**Co ustalono:**

1. **Replica dostarcza dokładnie JEDNĄ geometrię na scenę.** `mesh.ply` (77 MB, pełna
   teksturowana) i `habitat/mesh_semantic.ply` (19 MB) mają **bit-identyczne tablice
   wierzchołków** — 1 757 500 wierzchołków, ten sam zakres z (−1.641…1.440 m), ta sama
   liczba ścian (1 753 097). Drugi plik dodaje wyłącznie `object_id` per ścianę. **Nie
   istnieje alternatywna, zamknięta wersja siatki.**
2. **Brak sufitu jest własnością skanu, nie adnotacji.** W górnym paśmie z scena
   `frl_apartment_0` ma **4.4 m² powierzchni poziomej wobec 57.5 m² podłogi** — sprawdzone
   na normalnych trójkątów, nie na etykietach. Gdyby sufit istniał, ale był nieoznakowany,
   pokazałby się jako powierzchnia pozioma w kategorii `<class_id=-1>` albo innej.
3. **SoundSpaces 1.0 używał tych samych scen i tych samych metadanych.** Pliki
   `points.txt` i `graph.pkl` w `my-operations/metadata/replica/` pochodzą wprost
   z SoundSpaces.
4. **Brak jakichkolwiek śladów domykania siatki** w wendorowanym repozytorium. Jedyna
   wzmianka o obróbce geometrii (`meshSimplification`) dotyczy SoundSpaces **2.0**,
   a upraszczanie siatki nigdy nie dodaje powierzchni.

**Czego NIE da się ustalić lokalnie:** README SoundSpaces stwierdza wprost — *„we do not
open source the rendering code at this time"*. Kodu generującego RIR-y 1.0 nie ma, więc nie
można sprawdzić, czy ich potok nie domykał objętości przed symulacją.

**Wniosek:** bardzo prawdopodobnie ta sama otwarta geometria — bo **nie ma innej, której
mogliby użyć** — ale to wniosek z braku alternatywy, nie z bezpośredniego dowodu.

### Test rozstrzygający, gdyby był potrzebny

SoundSpaces udostępnia prekomputowane RIR-y **per scena**
(`scripts/download_data.py`, `<scena>.tar.gz`). Rozmiary sprawdzone: najmniejsza scena
otwarta `frl_apartment_2` = **6.90 GB**, najmniejsza zamknięta `office_1` = **0.10 GB**.

Metoda: zmierzyć RT60 ich RIR-ów tą samą metodą Schroedera i porównać **kontrast
otwarte/zamknięte wewnątrz danych SoundSpaces 1.0**.

**Dlaczego to omija problem różnych silników.** `GENERATOR_PARAMS.md` §5 ogr. 1 słusznie
zabrania porównywania metryk bezwzględnych między SoundSpaces 1.0 a 2.0. Ten test tego nie
robi: pyta wyłącznie, czy **ich własne dane** wykazują tę samą sygnaturę strukturalną
(sceny `frl_*` z krótszym pogłosem niż zamknięte). Wartości bezwzględne nigdy nie
przekraczają granicy silnika — porównanie jest wewnętrzne po obu stronach.

Zastrzeżenie do zaplanowania: w SoundSpaces 1.0 źródło i odbiornik to **różne** węzły grafu,
więc dominacja dźwięku bezpośredniego jest inna niż w echolokacji. Nachylenie późnej części
krzywej powinno być jednak porównywalne — a to ono niesie informację o pogłosie.

### Wynik testu (2026-07-29) — **NIE**, SoundSpaces 1.0 nie ma tej sygnatury

Skrypty: `measurements/fetch_soundspaces1_rirs.sh` (pobranie),
`measurements/soundspaces1_rt60.py` (analiza). Wynik: `outputs/measurements/ss1_verdict.txt`.

Pobrano prekomputowane RIR-y SoundSpaces 1.0 dla jednej sceny zamkniętej (`office_1`,
0.10 GB) i jednej otwartej (`frl_apartment_2`, 6.90 GB). RT60 policzono tą samą metodą
Schroedera, przy **dwóch kontrolach**:

1. **Dopasowanie odległości.** Losowe pary węzłów mają w `office_1` medianę 1.12 m, a
   w `frl_apartment_2` 4.03 m (16 vs 125 węzłów). Większa odległość osłabia dźwięk
   bezpośredni i przesuwa okno T20 w inną część krzywej. Ograniczono obie sceny do par
   1–3 m. Efekt był realny, ale nie decydujący (0.508 → 0.463 s).
2. **Odtworzenie geometrii źródła w naszym silniku.** U nas źródło jest **współlokowane**
   z odbiornikiem (echolokacja), u nich oddalone. To ta sama różnica, która tworzy
   wielonachyleniowy zanik (§2). Powtórzono więc nasz pomiar ze **źródłem oddalonym o ~2 m**.

**Wynik po obu kontrolach** (RT60 @ 1 kHz, ta sama geometria źródło–odbiornik):

| scena | sufit | V [m³] | nasze SS 2.0 | SoundSpaces 1.0 |
|---|---|---|---|---|
| `office_1` | zamknięta | 23 | 0.358 s | 0.396 s |
| `frl_apartment_2` | **brak** | 191 | **0.186 s** | **0.463 s** |

**Silniki zgadzają się na scenie zamkniętej (~10 %) i rozjeżdżają 2.5× wyłącznie na
otwartej — w przeciwnych kierunkach względem sceny zamkniętej.** W naszych danych scena
otwarta ma krótszy pogłos mimo 8× większej objętości; w ich danych dłuższy, zgodnie
z objętością.

**Interpretacja.** Obserwacja jest mocna i kontrolowana. Mechanizm — nie: wiodąca hipoteza
mówi, że potok SoundSpaces 1.0 domykał objętość przed symulacją albo używał modelu
nieczułego na otwarcie geometrii, ale **dowodu wprost nie ma**, bo kod renderujący nie
został opublikowany. Alternatywy, których nie da się wykluczyć: inne przetwarzanie RIR-ów
(są wyrównane czasowo i krótkie, ~0.35 s), inny model propagacji.

**Konsekwencja dla pracy — odwrotna do nadziei.** Otwartość scen **nie jest** własnością
dzieloną z literaturą, którą można zbyć przypisem. Nasze echa różnią się od baseline'ów
SoundSpaces 1.0 **strukturalnie i systematycznie na 46 % lokalizacji**. To wzmacnia
argument, który i tak jest w `GENERATOR_PARAMS.md` §5 ogr. 1: **porównywanie metryk
bezwzględnych z Gao i Paridą jest nieuprawnione**, a jedynym wiarygodnym porównaniem jest
36 vs 4 orientacje **wewnątrz naszego zbioru**. Teraz nie jest to już ostrożność
metodologiczna, tylko wniosek z pomiaru.

Uboczna obserwacja wzmacniająca wiarygodność obu potoków: na scenie **zamkniętej** dwa
niezależne silniki dają RT60 różniące się o ~10 %. To nie jest walidacja (jedna scena,
jedno pasmo), ale przesłanka, że rozbieżność na scenie otwartej nie bierze się z ogólnej
niezgodności metod.

### Uściślenie (2026-07-29): pomiar ucieczki promieni — **trzy grupy, nie dwie**

Podział na 6 „otwartych" i 12 „zamkniętych" powyżej pochodzi z **heurystyki geometrycznej**
(udział powierzchni poziomych w górnych 15 % wysokości sceny). Autorzy silnika wskazują
jednak inną, właściwą wielkość, więc została zmierzona wprost — i **częściowo tamtą
klasyfikację obala**.

**Dlaczego akurat ta wielkość.** SoundSpaces 2.0 (arXiv 2206.08312) pisze wprost, że siatki
muszą być pozbawione dużych otworów, *„otherwise the rays will leak from the holes, resulting
in inaccurate simulation"*, i przypisuje temu konkretny artefakt w swoich danych: *„The main
reason for the Matterport3D's RT60 distribution skewing towards left is because there are lots
of broken meshes in that dataset, which results in ray leaking from holes and smaller
reverberation in general."* To jest dokładnie kierunek, który zmierzyliśmy niezależnie na
Replice — więc nasza obserwacja nie jest lokalną anomalią, tylko **udokumentowanym trybem
awarii tej metody**.

Deklarują też API do jego pomiaru. W naszej kopii biblioteki funkcja istnieje
(`RLRAudioPropagation.h:503`, `RLRA_GetIndirectRayEfficiency()`, dokumentowana jako *„a measure
of how enclosed an acoustic space is"*), ale **habitat-sim jej nie eksponuje** — nie ma jej
w publicznym API `AudioSensor` ani w bindingach, więc bez zmiany w C++ i przebudowy jest
z Pythona nieosiągalna.

**Metoda zastępcza.** Równoprostokątny sensor **głębi** w punkcie słuchacza: jeden render daje
pełną sferę 256 × 512 kierunków, a piksel o głębi 0 to kierunek, w którym nie ma geometrii.
Udział takich kierunków ważony kątem bryłowym = ułamek uciekających promieni izotropowych.
Skrypt: `measurements/ray_escape_survey.py`, dane: `outputs/measurements/ray_escape/`.
Odrzucone drogi: `sim.cast_ray()` **na Replice w ogóle nie działa** — Bullet odrzuca siatkę
(czworokąty, nie trójkąty: `isMeshPrimitiveValid : Invalid primitive 0`), a konstrukcja
Simulatora kończy się `AssertionError`; trimesh nie ma w tym środowisku `rtree`/`embreex`.

**Kontrola negatywna jest wbudowana:** 8 scen daje 0.00–0.09 % w *każdej* lokalizacji. Gdyby
„głębia = 0" łapała cokolwiek poza brakiem geometrii, te sceny nie mogłyby wyjść zerowe.

| grupa | scen | lokalizacji | mediana ucieczki | max | lok. > 10 % | gdzie jest dziura |
|---|---|---|---|---|---|---|
| **szczelne** | 8 | 516 (29.7 %) | 0.00 % | 0.09 % | 0 % | — |
| **nieszczelne bokiem** | 4 | 417 (24.0 %) | 0.53–3.60 % | **48.3 %** | 0–36 % | przy horyzoncie i poniżej |
| **bez sufitu** | 6 | 807 (46.4 %) | 21.8–23.6 % | 29.0 % | 89–93 % | **wyłącznie** powyżej horyzontu |

- **szczelne:** `apartment_0`(211), `office_4`(76), `room_0`(57), `hotel_0`(48), `room_2`(47),
  `room_1`(35), `office_0`(26), `office_1`(16)
- **nieszczelne bokiem:** `apartment_1`(176), `apartment_2`(142), `office_3`(62), `office_2`(37)
- **bez sufitu:** wszystkie `frl_apartment_0..5`

**Dwa różne mechanizmy, rozróżnialne bezbłędnie po rozkładzie kątowym.** W rodzinie
`frl_apartment_*` pasmo powyżej 60° elewacji ucieka w **99–100 %**, a przy horyzoncie i niżej
w **0.0 %** — to podpis brakującej płaszczyzny sufitu. W czterech pozostałych scenach jest
odwrotnie: powyżej 60° **0.0 %**, a ucieczka siedzi w pasmach 10–30° i −10…10°, czyli
w otworach pionowych (przejścia, okna, niezeskanowane fragmenty ścian, krawędź sceny).

**Co to koryguje w tabeli na początku §1.** Cztery sceny opisane tam jako „zamknięte" **nie są
akustycznie szczelne**. Najważniejszy przypadek to `apartment_2` — scena **held-out**, 100 %
pokrycia sufitem, a mimo to **36 % jej lokalizacji traci ponad 10 % kąta bryłowego**, z
maksimum 41 %. Podobnie `apartment_1` (19 % lokalizacji > 10 %, maks. 48 %) i `office_2`
(16 %). Jedynie `office_3` jest graniczne i praktycznie nieszkodliwe (maks. 4.5 %, żadna
lokalizacja powyżej 5 %).

Dlaczego `sigma_1` tego nie pokazało: przeciek sufitowy jest **duży i jednorodny** — dotyka
każdej lokalizacji sceny, więc przesuwa medianę całej sceny. Przeciek boczny dotyczy
**mniejszości lokalizacji w scenie**, więc mediana sceny zostaje w paśmie „zamkniętych".
Rozdzielenie `sigma_1` opisane wyżej pozostaje prawdziwe — jest tylko **grubsze**, niż się
wydawało, i nie wykrywa drugiego mechanizmu.

**Skład podzbiorów po uściśleniu:**

| podzbiór | n | szczelne | nieszczelne bokiem | bez sufitu |
|---|---|---|---|---|
| cały zbiór | 1740 | 516 (29.7 %) | 417 (24.0 %) | 807 (46.4 %) |
| treningowy | 1374 | 440 (32.0 %) | 275 (20.0 %) | 659 (48.0 %) |
| **held-out** | 366 | 76 (20.8 %) | **142 (38.8 %)** | 148 (40.4 %) |

Zbiór testowy zawiera **dokładnie po jednej scenie każdego typu** (`office_4` szczelna,
`apartment_2` nieszczelna bokiem, `frl_apartment_5` bez sufitu) — to zaleta, bo pozwala
zbadać wszystkie trzy reżimy na danych trzymanych z boku. Trzeba jednak napisać uczciwie, że
proporcje **nie** odpowiadają całości: scen nieszczelnych bokiem jest w held-oucie 38.8 %
wobec 24.0 % w zbiorze. Wcześniejsze zdanie w §1 — że proporcje są zbliżone — było prawdziwe
tylko dla podziału dwudzielnego.

**Czy dodanie sufitu by to naprawiło — częściowo, i pomiar mówi dokładnie w jakiej części.**
Dla 6 scen `frl_apartment_*` cała ucieczka leży powyżej horyzontu, więc płaszczyzna sufitu
zamknęłaby praktycznie całość. Dla 4 scen nieszczelnych bokiem sufit **nie da nic** — tam
dziury są w pionie, a ich zamknięcie oznaczałoby dostawianie ścian w miejscach, gdzie
w oryginale są przejścia, co byłoby widoczne w RGB i w depth. Innymi słowy: łatwiejszy
przypadek to ten, który wygląda groźniej.

### Czy da się to naprawić „pod SoundSpaces 1.0"? Tak — i oto koszt

Wcześniejszy argument, że naprawa jest niewykonalna, bo sufit pojawiłby się w RGB i depth,
**jest nieprawdziwy** i trzeba go wycofać. Da się rozdzielić geometrię wizualną od akustycznej
bez żadnej zmiany w C++:

1. Załatana siatka (`mesh_semantic.ply` + sufit) jako **osobna scena** z własnym
   `stage_config.json`, z **przekopiowanym oryginalnym** `mesh_semantic.navmesh`, żeby zbiór
   lokalizacji został bit-w-bit ten sam.
2. Przebieg **audio** na scenie załatanej, przebieg **RGB/depth** na oryginalnej, scalenie
   w jeden HDF5. Koszt czasowy prawie bez zmian: audio to i tak ~cały budżet, a przebieg
   czysto wizualny jest o rzędy wielkości tańszy.

To nie jest nawet obejście — rozdział „acoustic geometry" od geometrii renderowanej jest
standardem w akustyce (robi tak m.in. Meta Acoustic Ray Tracing), a literatura opisuje wprost
domykanie skanów otoczką wypukłą scaloną z siatką oryginalną.

### Eksperyment wykonany (2026-07-29) — łata działa, ale przestrzeliwuje

Powyższe było rozumowaniem; poniżej pomiar. `patch_scene_ceiling.py` doklei sufit,
`ceiling_patch_rt60.py` mierzy RT60 tym samym estymatorem, którym policzono dane SS 1.0,
przy tych samych parach źródło–odbiornik (60 par, 1–3 m, średnia 1.99 m — tyle samo, co
w parach SS 1.0).

| pomiar | ucieczka promieni | RIR | RT60 @ 1 kHz | vs SS 1.0 (500 Hz–2 kHz) |
|---|---|---|---|---|
| `office_1` naturalnie zamknięta **(kontrola)** | 0.00 % | 1.08 s | 0.356 s | **0.94×** |
| `frl_apartment_2` oryginalna | 21.79 % | 0.48 s | 0.222 s | **0.48×** |
| `frl_apartment_2` **załatana** | **0.00 %** | 1.55 s | 0.628 s | **1.36×** |

**Hipoteza o domykaniu objętości przez SS 1.0 potwierdzona co do kierunku i rzędu
wielkości.** RIR wydłużył się 3.2×, RT60 wzrósł 2.5–3.4×, a błąd względem SS 1.0 spadł
z 2.1× za mało do 1.36× za dużo — logarytmicznie 2.4× bliżej. Brak sufitu odpowiada więc
za **większość** rozbieżności z baseline'ami.

**Ale łata przestrzeliwuje.** Kontrola na `office_1` pokazuje, że na scenie naturalnie
zamkniętej nasz silnik siedzi na 0.94× SS 1.0 — więc 1.36× nie jest ogólnym przesunięciem
silnika, tylko własnością łaty. Reszta to czynnik **1.45×**. Niezbadane przyczyny: materiał
`ceiling` z konfiguracji Repliki to „Gypsum Board" o α = 0.04–0.05 (bardzo odbijający),
a płaska płaszczyzna nie ma geometrii rozpraszającej prawdziwego sufitu. Materiału
**celowo nie dobierano pod wynik** — to byłoby dopasowywanie odpowiedzi.

**Dlaczego mimo to rekomendacja brzmi „nie robić tego w tej pracy":**

- **Nie wiemy, w co celujemy.** Eksperyment pokazał, że domknięcie objętości tłumaczy
  większość różnicy, ale nie *że* SS 1.0 zrobił dokładnie to — kod renderujący nie został
  opublikowany. Kalibrowanie łaty tak, żeby trafić w 0.463 s, byłoby dopasowaniem do jednej
  liczby na jednej scenie, a nie odtworzeniem cudzej metody.
- ~~**Naprawia najwyżej połowę problemu.**~~ **To okazało się nieprawdą — patrz niżej.**
  Domknięcie działa na wszystkich 10 nieszczelnych scenach, nie tylko na sześciu bez sufitu.
- **Nie zmienia wyniku badania.** Porównanie 36 vs 4 orientacje działa wewnątrz tego samego
  zbioru i te same rendery zasilają oba warunki — otwartość sceny jest zmienną wspólną
  (patrz początek §1).
- **Autorzy silnika sami tego nie naprawiają.** SoundSpaces 2.0 raportuje skrzywiony rozkład
  RT60 dla Matterport3D jako znane ograniczenie danych, nie łata siatek. Opisanie tego jako
  ograniczenia jest więc zgodne z praktyką w literaturze.

**Kiedy warto by to zrobić:** gdyby praca miała podawać **bezwzględne** metryki porównywalne
z Gao / Paridą. Wtedy różnica strukturalna wobec baseline'ów przestaje być przypisem i staje
się błędem systematycznym. Przy porównaniu wewnętrznym 36 vs 4 — nie.

**Co z tego wchodzi do pracy niezależnie od decyzji o generacji.** Sam eksperyment jest
wynikiem wartym opisania: pokazuje, że rozbieżność wobec baseline'ów ma **zidentyfikowaną
przyczynę geometryczną**, a nie nieznaną. Zdanie „nasze echa na `frl_apartment_*` mają
krótszy pogłos, bo skan nie ma sufitu — sprawdzone przez doklejenie sufitu, RT60 rośnie
wtedy 2.5–3.4× i przeskakuje wartość SS 1.0" jest znacznie mocniejsze niż „nie wiadomo,
skąd różnica". Kosztowało to jedną scenę i ok. 20 minut liczenia.

### Rozszerzenie (2026-07-29): domknięte są WSZYSTKIE sceny, a RT60 daje się dopasować do 0.96×

Pełny wynik i tabele: `RAPORT_SESJI_2026-07-26_29.md` §2.13. Tu tylko to, co zmienia wnioski
metodologiczne.

**Każda z 10 nieszczelnych scen ma 1–2 dziury o polu > 1 m²** — i nic więcej istotnego
(kilkaset pętli < 0.5 m² to brzegi mebli, których łatać nie wolno). Typy są rozpoznawalne
jednoznacznie i **potwierdzają intuicję o materiałach**:

| typ dziury | sceny | materiał |
|---|---|---|
| brakujący sufit (89–100 m²) | `frl_apartment_0..5` | ściany sceny (Gypsum Board) |
| urwana krawędź skanu (24 / 34 m²) | `apartment_1`, `apartment_2` | ściana |
| **okno i drzwi** (1.1–2.1 m²) | `office_2`, `office_3` | **Glass** / wood, Thick |

Okna i drzwi brakują, bo **szkło nie odbija światła strukturalnego IR** — skaner ich nie
widział. Materiał nie jest zgadywany: łata dziedziczy `object_id` z otoczenia dziury.

Po załataniu ucieczka promieni spada z 0.53–23.60 % do **0.00–0.24 %** we wszystkich 10
scenach. Twierdzenie „naprawa dotyczy tylko połowy problemu" było więc błędne — wycofane wyżej.

**Co to zmienia dla pracy:**

1. **Zdanie o ograniczeniu zbioru staje się mocniejsze.** Można teraz napisać nie tylko
   „6 scen nie ma sufitu", ale „wszystkie 18 scen dają się doprowadzić do stanu akustycznie
   szczelnego, a wpływ tego na pogłos jest zmierzony" — z liczbą 0.48× → 0.96× wobec SS 1.0.
2. **Powstaje realna opcja wariantu zbioru.** Skoro łatanie jest w pełni zautomatyzowane
   (jedno wywołanie na wszystkie sceny, 742 MB siatek, kilka minut), wygenerowanie
   **drugiego wariantu datasetu na scenach domkniętych** przestaje być projektem i staje się
   decyzją o czasie GPU. Byłaby to naturalna analiza dodatkowa: czy przewaga 36 orientacji
   nad 4 zależy od tego, czy scena jest szczelna.
3. **Rekomendacja dla wariantu głównego się NIE zmienia.** Nadal: generować na oryginalnej
   geometrii i opisać otwartość jako własność zbioru Replica. Powody: dopasowanie materiału
   jest dopasowaniem do jednej scenie (patrz niżej), zmiana geometrii zerywa porównywalność
   z VisualEchoes, a porównanie 36 vs 4 jest na to niewrażliwe.

**Decyzja (2026-07-30): dopasowany materiał USUNIĘTY z domyślnej konfiguracji łaty.**
`patch_scene_holes.py` bez dodatkowych flag daje wariant w pełni semantyczny (materiał
dziedziczony z otoczenia dziury, sufit na wysokości ocalałego fragmentu). Dopasowanie
zostaje odtwarzalne przez `--ceiling-class rug`, ale jako **eksperyment**, nie jako element
potoku. Powód poniżej: zgodność uzyskana strojeniem nie jest argumentem, a właściwym
sprawdzeniem poprawności silnika jest pomiar na scenach o **całej** geometrii, gdzie nie ma
czego stroić — `cross_engine_rt60.py`, §2.14 raportu.

**Granice tego, co wolno powiedzieć o zgodności 0.96×.** To **dopasowanie, nie walidacja** —
materiał sufitu wybrano tak, żeby trafić w RT60 tej jednej sceny (semantycznie poprawny
Gypsum Board daje 1.48×; dopasowany Carpet, Heavy o α = 0.37 daje 0.96×). Dodatkowo
dopasowanie poprawia **skalę, nie kształt widmowy**: zostaje 1.34× przy 250 Hz i 0.91× przy
2 kHz, bo RT60 w SS 1.0 jest płaskie w częstotliwości, a nasze opada. Zależność nie jest
nawet monotoniczna po α (Foliage 0.17 → 0.86×, Carpet 0.20 → 1.06×), więc „α = 0.37" nie jest
zmierzoną stałą, tylko najlepszym wierszem tabeli. Żeby to była walidacja, trzeba dociągnąć
RIR-y drugiej sceny otwartej i sprawdzić na niej materiał dobrany tutaj — to jedyny brakujący
krok, gdyby ten wynik miał wejść do pracy jako zgodność, a nie jako dopasowanie.

---

## 2. Zanik jest wielonachyleniowy, bo źródło jest w punkcie odbioru

### Fakt

W echolokacji źródło dźwięku jest **współlokowane z odbiornikiem**. Dźwięk bezpośredni ma
przez to energię nieporównywalnie większą od pola pogłosowego i sam tworzy stromy spadek na
początku krzywej Schroedera. Zmierzone na `apartment_0` przy 1 kHz:

| okno dopasowania | nachylenie | wynikowe RT60 |
|---|---|---|
| −5…−15 dB | −136 dB/s | 0.441 s |
| −15…−25 dB | −82 dB/s | 0.732 s |
| −25…−35 dB | −80 dB/s | **0.755 s** ← właściwy pogłos |

Punkt −5 dB wypada po **3 ms** — zanim pole pogłosowe zdąży się rozwinąć.

### Konsekwencja metodologiczna

Standardowe T20 (−5…−25 dB) **nie mierzy tu pogłosu pomieszczenia**, tylko zanik impulsu
bezpośredniego. Każdy pomiar czasu pogłosu w konfiguracji echolokacyjnej musi używać
późnego okna. To nie jest kwestia preferencji — to różnica 0.44 vs 0.76 s na tej samej
krzywej.

### Uboczny, ale wartościowy wniosek: okno 60 ms jest dobrze dobrane

`ECHO_MS = 60` odziedziczono z VisualEchoes bez własnego uzasadnienia. Pomiar krzywej zaniku
pokazuje, gdzie to okno się kończy:

| scena | poziom EDC przy 60 ms |
|---|---|
| `apartment_0` (zamknięta, RT60 ≈ 0.77 s) | ≈ −15 dB |
| `frl_apartment_0` (otwarta, RT60 ≈ 0.20 s) | ≈ −29 dB |

Okno obejmuje więc **pierwsze 15–30 dB zaniku**, czyli obszar zdominowany przez dźwięk
bezpośredni i wczesne odbicia — tam, gdzie siedzi informacja **geometryczna** (odległości do
konkretnych powierzchni), a nie uśredniona charakterystyka pomieszczenia. Dla zadania
predykcji głębi jest to właściwy zakres. **Odziedziczony parametr okazuje się uzasadniony
merytorycznie** — to warto napisać, zamiast zostawiać go jako „tak było u Gao".

---

## 3. Dwa estymatory szumu i dlaczego nie wolno ich mylić

W projekcie występują dwa estymatory `sigma_1` i różnią się **rodzajem** ograniczenia:

| estymator | wzór | ograniczenie | SD |
|---|---|---|---|
| połówkowy | `RMSE(A,B)/√2·√(n/2)` | **1 stopień swobody na komórkę, niezależnie od n** | 4–6 %, **nie poprawia się z n** |
| wariancyjny | `σ² = śr. po komórkach z Var po renderach` | `n−1` stopni swobody na komórkę | 0.1–1.1 % przy n = 80 |

Sufit dokładności estymatora połówkowego wynika z **korelacji przestrzennej spektrogramu**:
efektywna liczba niezależnych komórek to ~600–1200 z 85 324. Dokładanie renderów zmniejsza
amplitudę różnicy `A−B`, ale nie jej względną precyzję.

**Dlaczego generator używa gorszego:** musi być spójny z regułą wyznaczającą `N`, bo
`snr_probe` i `snr_final` mają odpowiadać tej samej wielkości co próg. To wybór spójności,
nie jakości — i tak należy go opisać.

**Praktyczna reguła:** wszystko dokładniejsze niż ~10 % liczyć estymatorem wariancyjnym.
Zignorowanie tego kosztowało w tej pracy jeden błędny wniosek (rzekome +7 % i +18 % różnicy
między ścieżkami audio, w rzeczywistości artefakt estymatora i rozgrzewki).

---

## 4. Próbkowanie systematyczne potrafi ominąć skupisko

Podłogę szumu szacowano najpierw z 12, potem z 52 pozycji dobieranych po **ustalonych
ułamkach** listy lokalizacji (0.20 i 0.75). Ekstrapolacja przewidywała **zero** lokalizacji
przekraczających `N_MAX = 40`. Pełny census 1740 lokalizacji znalazł **siedem**.

Przyczyna: gorące miejsce akustyczne w `apartment_0` leży przy `loc_id` 285–310, czyli około
ułamka 0.9 listy — poza oboma próbkowanymi punktami. `loc_id` rośnie wzdłuż siatki punktów,
więc **ustalone ułamki listy odpowiadają ustalonym obszarom sceny**, a nie losowym punktom.

**Wniosek do metodologii:** gdy jednostka próbkowania (lokalizacja) generuje wiele obserwacji
(36 orientacji), a koszt pełnego pomiaru jest rzędu 1 % kosztu zadania właściwego — warto
zmierzyć wszystko zamiast ekstrapolować. Census kosztował 35 min wobec ~32 h generacji.

To dobry przykład do rozdziału metodologicznego: **ekstrapolacja z małej, nielosowej próby
zawiodła w sposób przewidywalny i wykrywalny**, a nie przez pecha.

---

## 5. Co z tego wchodzi do pracy i w jakiej roli

| ustalenie | rola w pracy |
|---|---|
| Sceny otwarte vs zamknięte | **ograniczenie zbioru** + zmienna warstwująca w analizie |
| Podział train/test reprezentatywny pod tym względem | argument obronny, podać jawnie |
| Wielonachyleniowy zanik przy źródle w punkcie odbioru | uzasadnienie metody pomiaru RT60 |
| Okno 60 ms obejmuje wczesne odbicia | **uzasadnienie odziedziczonego parametru** |
| Sufit dokładności estymatora połówkowego | uwaga metodologiczna, tłumaczy wybór dwóch estymatorów |
| Porażka ekstrapolacji z 52 pozycji | przykład metodologiczny: kiedy mierzyć zamiast szacować |
| Test RT60 vs Eyring: sceny szczelne **1.27×, zakres 1.02–1.75×** (n = 21) | **jedyna walidacja fizyczna** — z jawnym zakresem stosowalności (raport §2.15) |
| Szum pojedynczego renderu do 2.34× większy od sygnału 10° | **uzasadnienie reguły adaptacyjnej** — dowodzi, że uśrednianie jest konieczne, nie kosmetyczne |
| `TARGET_SNR = 3.5` bez wyprowadzenia | opisać jako **wybór konserwatywny**, nigdy jako wielkość wyprowadzoną (raport §2.17) |
| Warianty `main` / `patched` różnią też `depth` | **zakaz porównywania po błędzie bezwzględnym** — porównywać przewagę 36 vs 4 wewnątrz każdego (raport §2.16) |

### Dwa fakty dopisane 2026-11-12 (do podania przy cytowaniu)

1. **`office_4` nie ma w teście RT60 vs Sabine/Eyring.** Jest to ósma scena szczelna (§1, grupa
   „szczelne"), a przebieg A testu objął **7** scen szczelnych: `room_0`, `office_1`, `hotel_0`,
   `apartment_0`, `room_1`, `room_2`, `office_0` (`outputs/measurements/sabine/A_sealed.log`,
   `summary.txt`). Sceny podano jawnie przez `--scenes`; `office_4` nie ma też w `DEFAULT_SCENES`
   (`measurements/rt60_vs_sabine.py:65`). **Powód pominięcia nie jest nigdzie zapisany** — ani
   w logu (brak błędu i wpisu o pominięciu), ani w `RAPORT_SESJI_2026-07-26_29.md` („na 7 scenach").
   Nie jest to wykluczenie sceny held-out jako takiej, bo `frl_apartment_5` (też held-out) jest
   w przebiegach B i C. Wynik „1.27×, n = 21" dotyczy więc 7 z 8 scen szczelnych.
2. **Trzy pliki dowodowe powstały ad hoc.** `outputs/ml/echo_ablation/final_results_2026-08-13.json`,
   `echo_ablation.json` i `echo_3seeds.json` nie mają programu, który by je zapisywał — w kodzie
   repo są wyłącznie ich odczyty (`ml/analysis/thesis_numbers.py`, `ml/analysis/figures.py`).
   `ml/analysis/final_results.py:16` potwierdza to wprost dla `final_results_2026-08-13.json`.
   Liczby z tych plików da się sprawdzić tylko z surowych `outputs/ml/eval/*/eval.json`
   i `compare_*.json`, nie odtworzyć jednym poleceniem.

## 6. Otwarte pytania do rozstrzygnięcia przy pisaniu

1. Czy raportować wyniki **z podziałem na sceny otwarte/zamknięte**? Rekomendacja: tak, jako
   analizę dodatkową — to darmowa informacja o źródle sygnału.
2. Czy zweryfikować, jak SoundSpaces 1.0 traktował sceny bez sufitu? Rozstrzygnęłoby, czy
   otwartość jest własnością dzieloną z literaturą.
3. Czy `office_1` (wygenerowana starymi parametrami: `N_MAX = 40`, `WARMUP_DISCARD = 20`)
   ma zostać przegenerowana dla jednorodności? Parametry są zapisane w atrybutach pliku,
   więc różnica jest wykrywalna; `echo_ctl.py regen office_1` robi to w ~30 min.
4. Czy walidacja wobec rzeczywistych pomiarów apartamentu FRL (SoundSpaces 2.0, Sek. 5.2)
   jest w zasięgu? Byłaby to mocniejsza walidacja fizyczna niż Sabine — i dotyczy dokładnie
   tej rodziny scen, która okazała się otwarta.

## Stan kodu treningu z 11–12.08.2026 (przebiegi sprzed commita e5eff1a1)

Kontrola z 2026-11-13, wyłącznie odczytowa (git diff/log/show, pliki JSON). Pytanie: czy kod,
na którym powstało 14 przebiegów sieci głębi z 11.08, różni się od dzisiejszego w sposób,
który mógłby zmieniać wyniki. Czasy commitów przeliczone na UTC: `e2fbf45e` = 2026-08-05
16:20:34 UTC (18:20 +0200), `e5eff1a1` = **2026-08-12 13:07:10 UTC** (15:07 +0200).

### (a) Które przebiegi powstały przed `e5eff1a1`

`started_utc` z `outputs/ml/runs/<id>/config.json`, `ended_utc` ze `status.json`,
ewaluacja = pole `created_utc` z `outputs/ml/eval/<id>/eval.json` (zgodne z datą modyfikacji
pliku, która jest w czasie lokalnym +0200: 23:21 dla SE/B, 13:30–13:31 dla echo2depth).

| przebieg | start (UTC) | koniec (UTC) | eval.json (UTC) |
|---|---|---|---|
| `SE_seed0`  | 08-11 19:05:07 | 08-11 20:01:19 | 08-11 21:21:14 |
| `B_seed0`   | 08-11 20:01:22 | 08-11 20:57:39 | 08-11 21:21:25 |
| `EA_seed0`  | 08-11 20:58:02 | 08-11 21:08:31 | 08-12 11:30:13 |
| `EB_seed0`  | 08-11 21:08:33 | 08-11 21:18:52 | 08-12 11:30:19 |
| `ED_seed0`  | 08-11 21:18:54 | 08-11 21:29:35 | 08-12 11:30:24 |
| `ESE_seed0` | 08-11 21:29:37 | 08-11 21:40:00 | 08-12 11:30:29 |
| `EA_seed1`  | 08-11 21:40:02 | 08-11 21:50:29 | 08-12 11:30:34 |
| `EB_seed1`  | 08-11 21:50:31 | 08-11 22:00:42 | 08-12 11:30:40 |
| `ED_seed1`  | 08-11 22:00:44 | 08-11 22:11:10 | 08-12 11:30:45 |
| `ESE_seed1` | 08-11 22:11:12 | 08-11 22:21:24 | 08-12 11:30:50 |
| `EA_seed2`  | 08-11 22:21:26 | 08-11 22:31:50 | 08-12 11:30:56 |
| `EB_seed2`  | 08-11 22:31:52 | 08-11 22:42:08 | 08-12 11:31:01 |
| `ED_seed2`  | 08-11 22:42:09 | 08-11 22:52:36 | 08-12 11:31:06 |
| `ESE_seed2` | 08-11 22:52:37 | 08-11 23:02:50 | 08-12 11:31:12 |

Wszystkie 14 przebiegów i wszystkie 14 ewaluacji mieszczą się między `e2fbf45e` a `e5eff1a1`.
Pozostałe przebiegi w `outputs/ml/runs/` (bez katalogów `*_reeval`, które mają skopiowany
`config.json` oryginału) startowały **po** `e5eff1a1`; najwcześniejszy z nich to `EPA_seed0`,
2026-08-13 05:38:11 UTC (~16,5 h po commicie), dalej `EPB/EPD_seed0`, `A_seed0`, `D_seed0`,
`EK*` (08-13), `EPA/EPB/EPD_seed1/2`, `A/B/D_seed1/2` (08-15/16), `ED_seed*_nf*` (11-12)
i `ED_seed0_as1` (11-13, w toku). Pliki `eval.json` tych 14 przebiegów są w `e5eff1a1`
zacommitowane i do dziś niezmienione (`git diff e5eff1a1 -- outputs/ml/eval/<id>/eval.json`
pusty dla każdego z 14).

### (b) Różnice `e2fbf45e` → `e5eff1a1` w plikach treningu, danych i ewaluacji

Pominięto zmiany ścieżek importów, komentarzy i czysty refaktor. Ocena „zmienia wynik
`ED_seed0`?” dotyczy treningu i ewaluacji test@36.

1. **`depth_model/train_condition.py` — protokół walidacji.** W `e2fbf45e` walidacja szła na
   podzbiorze kątów warunku (`angle_subset=cond.angle_subset`), w `e5eff1a1` zawsze na pełnych
   36 kątach (`VAL_ANGLE_SUBSET = "all"`), z dodatkowym `val@4` liczonym z tych samych predykcji
   i osobnymi wagami `best4_*.pth`. Checkpoint `best_*.pth` wybierany po `val@36`.
   Zmienia wynik `ED_seed0`? **Tak względem stanu z 05.08 — ale nie względem `e5eff1a1`**:
   14 przebiegów już używało nowego protokołu (`status.json`: `val_angle_subset = "all"`,
   istnieją `best_val36_rmse`, `best_val4_rmse`, `best_step_same`; wiersze `metrics.jsonl`
   mają klucze `all` + `val4` i `n_samples = 6588`, czyli 36 kątów zbioru walidacyjnego;
   `best_step_val36`/`best_step_val4` w `status.json` równe argminom z `metrics.jsonl`).
2. **`train_condition.py::evaluate()` — agregacja metryk walidacji.** `Evaluator` →
   `SampleStatsCollector`/`SampleTable` (`depth_model/metrics.py`), klucz `overall` → `all`.
   Definicja RMSE ta sama w obu: pierwiastek z sumy kwadratów błędu przez liczbę pikseli
   `gt > 0` (RMSE po pikselach całego zbioru, nie średnia po próbkach). Zmienia wynik? **Nie** —
   ta sama definicja, a 14 przebiegów zapisało już format `all`.
3. **`train_condition.py` — kasowanie `metrics.jsonl` i `train_loss.csv` przy świeżym
   przebiegu (bez `--resume`).** W kodzie z 11.08 tego jeszcze nie było: `EA/EB/ED/ESE_seed0`
   mają po **80** wierszy w `metrics.jsonl` (40 starych w formacie `overall`, z wcześniejszego
   przebiegu na starym protokole walidacji, + 40 nowych) i 801 linii w `train_loss.csv`.
   Zmienia wynik? **Nie** — najlepszy krok trzymany jest w pamięci od `inf`, wiersze nowego
   przebiegu zaczynają się od kroku 1000, a `best_*.pth` mają czasy modyfikacji z nowego
   przebiegu (np. `ED_seed0`: 23:21:19 +0200 = krok 9000, ~2,5 min po starcie). Skutek tylko
   diagnostyczny — patrz pkt 4.
4. **`train_condition.py::_plateau_diagnostics()` + `budget_ceiling_warning`** (nowe). Tylko
   pola diagnostyczne `status.json`. Zmienia wynik? **Nie.** Uwaga faktograficzna: funkcja czyta
   `r["overall"]["RMSE"]`, a `evaluate()` zapisuje klucz `all`, więc dla przebiegów z samymi
   nowymi wierszami zwraca `n_validations = 0` (tak jest we wszystkich przebiegach poza
   czterema), a w `EA/EB/ED/ESE_seed0` (`n_validations = 40`) liczy plateau z **wierszy
   poprzedniego przebiegu**. Pola `plateau`/`still_improving_at_end`/`budget_ceiling_warning`
   nie są więc wiarygodne w żadnym przebiegu; nie wpływają na wagi ani na ewaluację.
5. **`train_condition.py` + `dataset/echo_h5_dataset.py` — permutacja echa
   (`shuffle_echo_seed`, `_echo_permutation`)**, stosowana też do walidacji. W `e2fbf45e`
   nie istniała wcale (nie było też warunków SE/ESE), więc SE/ESE z 11.08 musiały powstać na
   kodzie niezacommitowanym. Zmienia wynik `ED_seed0`? **Nie** (`shuffle_echo_seed = null`
   w `config.json` ED). Dla treningu `SE_seed0`/`ESE_seed*`: **nieustalone** — nie da się
   sprawdzić, czy implementacja z 11.08 była identyczna z zacommitowaną (komentarz w kodzie
   opisuje zastąpienie błędnej wersji wektorowej pętlą zamian; obie zużywają RNG inaczej).
   Ponowna ewaluacja tego nie rozstrzyga, bo ewaluacja SE/ESE idzie na prawdziwym echu
   (`shuffled_echo: false` w oryginale i w powtórce).
6. **`echo_h5_dataset.py` — maska przecięcia (`mask_variant`, `mask_mode`, `valid_ref`) i
   wydzielenie `_lookup`.** Maska domyślnie wyłączona (`None`), używana tylko w ewaluacjach
   `*_mask-*`; `_lookup` to ta sama logika wyszukiwania wierszy. Zmienia wynik? **Nie.**
7. **`dataset/angles.py` — `random_K_of_G` i `select_angles`.** `rng.choice(N_ANGLES)` →
   `rng.choice(pool.size)` z `pool = grid` dla `random_K`; `pool.size == 36 == N_ANGLES`,
   ten sam RNG z tego samego skrótu SHA-256. Sprawdzone wykonaniem obu wersji z gita:
   wynik bitowo identyczny w 10 800 przypadkach (`random_4`/`cardinal`/`all` × ziarna 0–2 ×
   4 sceny × 300 lokalizacji). Zmienia wynik? **Nie.**
8. **`dataset/splits.py`, `dataset/echo_data.py`, `depth_model/fast_bilinear.py`.** Tylko
   importy/tekst pomocy (`fast_bilinear.py` bez zmian). Odcisk podziału `e0bf7547668d9e0a`
   w `config.json` wszystkich przebiegów. Zmienia wynik? **Nie.**
9. **`matrix/experiments.py` (dawniej `ml/experiments.py`).** Doszło pole
   `Condition.shuffle_echo_seed` i warunki SE/ESE/EK*/PD/EPA/EPB/EPD oraz planowanie
   (`PLANNED_SEEDS`, `DEFERRED_GROUPS`, `disk_budget`). Definicja ED i wartości domyślne
   `RunSpec` (`TOTAL_STEPS = 40 000`, `BATCH_SIZE = 32`, Adam, lr 1e-4, wd 5e-4, β1 0,9,
   AMP, walidacja co 1000) bez zmian. Zmienia wynik? **Nie.**
10. **`depth_model/metrics.py`.** `Evaluator`/`MetricAccumulator` zachowane z tą samą
    definicją RMSE; dodano `SampleStatsCollector`, `SampleTable`, bootstrap po lokalizacjach.
    Zmienia wynik? **Nie.**
11. **`depth_model/evaluate.py` — nowy w `e5eff1a1`.** W `e2fbf45e` nie było żadnego skryptu
    ewaluacji testowej: istniała tylko walidacyjna `evaluate()` w `train_condition.py` i
    `Evaluator` w `metrics.py` (zbiór `test` pojawia się jedynie w liczeniu liczności i
    `verify_loader.py`). Ewaluacje z 11–12.08 musiały więc użyć niezacommitowanej wersji
    `evaluate.py`. Ich `eval.json` mają ten sam układ pól co późniejsze (jedyna różnica:
    `mask_mode`, dodane po `e5eff1a1`). Zmienia wynik? **Nie** — ustalone empirycznie, pkt (c).

Zmiany po `e5eff1a1` w tych plikach (`git diff e5eff1a1 HEAD`): komentarz o statystykach
ImageNet, opcjonalny `dataset_filter` (domyślnie `None`), `--tag` w `train_condition.py`,
`--mask-mode` w `evaluate.py`; w drzewie roboczym niezacommitowany `--angle-seed`
(domyślnie `None` = dotychczasowe zachowanie). Żadna nie zmienia zachowania domyślnego.

### (c) Dowody spójności

- **Pola `status.json`.** Wszystkie 14 przebiegów ma ten sam zestaw kluczy co przebiegi
  późniejsze (`A_seed0`, `EPA_seed0`, `EK6_seed0`, `ED_seed0_nf0`): m.in. `val_angle_subset`
  (= `"all"`), `best_val36_rmse`, `best_step_val36`, `best_val4_rmse`, `best_step_val4`,
  `best_step_same`, `plateau`. Jedyny wyjątek czasowy: `EA_seed0/status.json` ma datę
  modyfikacji 2026-08-12 14:24 +0200 (pozostałe pliki tego katalogu 08-11 ~23:00); jego
  zawartość zgadza się z `metrics.jsonl` (`best_step_val36 = 2000`, `best_step_val4 = 11000`).
- **`config.json`.** `spec` (bez `condition`/`seed`/`tag`) identyczny we wszystkich przebiegach
  echo2depth (`EA/EB/ED/ESE` z 08-11, `EK*`, `EP*`, `ED_*_nf*`): `total_steps 40000`,
  `batch_size 32`, `amp true`, `lr 1e-4`, `weight_decay 5e-4`, `beta1 0.9`, `adam`,
  `validation_freq 1000`, `edge_threshold_m 0.1`, `num_workers 8`, `fast_bilinear false`;
  w pełnym modelu `SE_seed0`/`B_seed0` identyczny z `A/B/D_seed*` (`fast_bilinear true`).
  Słownik `condition` każdego z warunków B/SE/EA/EB/ED/ESE ma jedną wersję we wszystkich
  przebiegach; `n_train`/`n_val` też zgodne (ED: 5 496 / 6 588; B: 49 464 / 6 588).
- **Ponowna ewaluacja** (`outputs/ml/overnight_2026-11/reeval_14_runs.md`, katalogi
  `eval/<id>_reeval`): te same wagi oceniane dzisiejszym `evaluate.py` dają test@36 różniący
  się od zapisanego o **maks. |Δ RMSE| = 2,41·10⁻⁷** (14 przebiegów); dla `ED_seed0/1/2`
  odpowiednio 1,5·10⁻⁸, 1,13·10⁻⁷, 2,7·10⁻⁸, czyli ≤ 1,1·10⁻⁷ — poziom szumu numerycznego
  float32/AMP. Ewaluacja z 11–12.08 jest więc odtwarzalna dzisiejszym kodem.

### (d) Co jest, a czego nie ma w gicie

**Stanu kodu treningowego z 11–12.08.2026 nie da się odtworzyć z gita — nie był
zacommitowany.** W gicie są tylko stany z 05.08 (`e2fbf45e`) i z 12.08 15:07 (`e5eff1a1`);
różnice między nimi opisano w (b). Kod z 11.08 jest „pomiędzy”: miał już walidację na 36
kątach, `val@4`, format `all` i permutację echa (dowód w artefaktach), ale jeszcze nie miał
kasowania logów przy świeżym przebiegu (dowód: 80 wierszy w czterech `metrics.jsonl`).

**Co z tego wynika dla wiarygodności.** Fakty: ewaluacja tych 14 przebiegów jest odtworzona
dzisiejszym kodem do 2,4·10⁻⁷, a wszystkie widoczne w artefaktach parametry treningu
(hiperparametry, podział, definicja warunku, protokół walidacji i wyboru checkpointu) są
zgodne z kodem zacommitowanym i z przebiegami późniejszymi. Nie da się natomiast wykluczyć
różnic w niezapisanej w artefaktach części ścieżki treningu (np. implementacji permutacji
echa dla SE/ESE) — to pozostaje niesprawdzalne, a nie wykazane. Rozbieżność `ED_seed0`
(test@36 0,64695) wobec powtórzeń `ED_seed0_nf0–nf4` (0,64986–0,65477) **nie wynika
z ewaluacji**; czy wynika ze stanu kodu z 11.08, czy z niedeterminizmu treningu, ta kontrola
nie rozstrzyga.
