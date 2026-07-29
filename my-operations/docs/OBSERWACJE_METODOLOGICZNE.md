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

### Stan testu (2026-07-29, w toku)

Uruchomiony: `measurements/fetch_soundspaces1_rirs.sh` pobiera i analizuje.
Skrypt analizy: `measurements/soundspaces1_rt60.py`. Wynik trafi do
`outputs/measurements/ss1_verdict.txt`, log do `ss1_check.log`.

**Wynik cząstkowy — scena zamknięta `office_1` (0.10 GB, już pobrana):**

| | RT60 @ 1 kHz |
|---|---|
| SoundSpaces 1.0 (prekomputowane RIR-y) | **0.395 s** |
| nasze SoundSpaces 2.0 / RLR | **0.401 s** |

Zgodność 1.5 % — mimo że to **różne silniki**. Nie należy z tego robić twierdzenia
o porównywalności (jedna scena, jeden punkt pasma, a §5 ogr. 1 dotyczy metryk zadaniowych,
nie RT60), ale jest to przesłanka, że oba potoki „widzą" tę samą geometrię i zbliżone
materiały. RIR-y SS 1.0 mają 44100 Hz i długość ~0.33 s; dla `office_1` katalog kąta 0
zawiera 256 plików = 16 × 16, czyli dokładnie tyle węzłów, ile lokalizacji ma ta scena w pkl.

**Brakuje sceny otwartej** (`frl_apartment_2`, 6.90 GB) — pobieranie w toku. Dopiero ona
rozstrzyga, bo test opiera się na kontraście, nie na pojedynczej scenie.

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
| Test RT60 vs Eyring, 1.13–1.75× | **jedyna walidacja fizyczna** — z jawnym zakresem stosowalności |

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
