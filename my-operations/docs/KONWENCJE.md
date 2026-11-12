# Konwencje i słownik oznaczeń

Dokument dodany **2026-09-21**. Nie zawiera żadnych nowych wyników. Zbiera w jednym miejscu
konwencje, które w raportach sesji ustalały się stopniowo i nie zawsze konsekwentnie, oraz
rozwija skróty używane w pozostałych dokumentach. Czytać **przed** `STAN_WYNIKOW.md`.

---

## 1. Który dokument do czego

| dokument | rola | czy przepisywać z niego liczby do pracy |
|---|---|---|
| `STAN_WYNIKOW.md` | stan obowiązujący, bez historii | **tak — jedyne źródło** |
| `GENERATOR_PARAMS.md` | zamrożona specyfikacja generatora (rozdz. 3 pracy) | tak, dla parametrów generatora |
| `LICZBY_DO_PRACY.md` | **katalog** każdej zmierzonej liczby z plikiem dowodowym | nie bezpośrednio — służy do sprawdzenia, skąd liczba pochodzi |
| `RAPORT_SESJI_*.md` | dziennik chronologiczny, zawiera wartości później zastąpione | **nie** — tylko do uzasadnień i historii decyzji |
| `MODELE.md` | nazwy modeli | — |
| `KONWENCJE.md` | ten plik | — |

`LICZBY_DO_PRACY.md` celowo zawiera też liczby techniczne (przepustowość dataloadera, rozbieżność
wag w kontroli determinizmu) i pozycje oznaczone „ZASTĄPIONE". Do tekstu pracy trafia z niego
niewielka część. Filtr: status **[Z]** albo **[Z-]** (ten drugi zawsze razem z zastrzeżeniem)
i **brak** słów „ZASTĄPIONE" / „nie cytować" w nazwie lub uwadze.

---

## 2. Konwencja znaku różnic

**Zasada: etykieta `X − Y` znaczy dosłownie `wartość(X) − wartość(Y)`.** Żadnych wyjątków.

Dla metryk błędu (RMSE, MAAE) — im mniej, tym lepiej — kontrasty zapisuje się jako
**Δ = nowy − bazowy**, np. `D − A`, `B − D`, `K36 − K4`, `pretext − scratch`:

- **Δ < 0 → nowy warunek ma mniejszy błąd, czyli jest lepszy**,
- Δ > 0 → nowy warunek jest gorszy.

Przykład: `ED − EA = 0,64432 − 0,79104 = −0,14672` → gęstość kątowa obniża RMSE.

W zdaniach do pracy wygodniej jest podawać **wielkość** poprawy. Wtedy nazwa musi to mówić
wprost: „**spadek RMSE** o 0,147", „przejście 4→9 daje **spadek RMSE** 0,128". Wartość dodatnia
z etykietą `D − A` jest błędem zapisu.

Wyjątki od „mniej = lepiej", które mają własne nazwy i są zawsze dodatnie:

| wielkość | definicja | znaczenie wartości dodatniej |
|---|---|---|
| **wkład echa** (`c_full`, `ESE − EB`) | RMSE(echo przetasowane) − RMSE(echo prawdziwe) | model korzysta z echa |
| **luka generalizacji** | RMSE na kątach niewidzianych − RMSE na widzianych | kara za brak pokrycia orientacji |
| **pp rozpiętości** (sondowanie) | część drogi od enkodera losowego (0 %) do uczonego na głębi (100 %) | więcej informacji o głębi |
| **udział gęstości** | (D − A) / (B − A) · 100 % | — |

**Stan w repozytorium (2026-09-21).** Plik `outputs/ml/echo_ablation/echo_3seeds.json` zapisuje
kontrasty jako **A − D, D − B, A − B** (wartości dodatnie) pod etykietami `D-A`, `B-D`, `B-A`.
Tak samo `bootstrap` w `evaluate.py --compare X Y` zwraca **X − Y**, a w raportach bywał wołany
z warunkiem gorszym na pierwszym miejscu. Pliki dowodowe zostają bez zmian (odwołują się do nich
raporty); znak jest sprowadzany do konwencji w `thesis_numbers.py` i w `STAN_WYNIKOW.md`.
Miejsca w raportach sesji z odwróconą etykietą są wypisane w ramkach **ERRATA 2026-09-21** na
początku tych raportów.

---

## 3. Dwie różne wielkości nazywane „RMSE"

To jest najczęstsze źródło pomyłek przy czytaniu dokumentów. **W pracy używać dwóch różnych nazw.**

### 3.1 RMSE głębi — metryka jakości modelu, w metrach

Dla wszystkich **ważnych** pikseli zbioru testowego (głębia prawdziwa > 0; wybrana maska może
ten zbiór dodatkowo zawęzić, §4.4):

```
RMSE_głębi = sqrt( Σ (d_pred − d_gt)² / liczba_ważnych_pikseli )      [m]
```

Suma idzie po pikselach **całego** zbioru (albo grupy: sceny, kubełka kątowego, lokalizacji
w bootstrapie), nie jest to średnia z RMSE poszczególnych obrazów. Implementacja
`ml/depth_model/metrics.py` (`MetricAccumulator`) jest zgodna z `compute_errors` Paridy do 1,5·10⁻⁶ (`test_matches_parida()`).
Wartości w pracy: 0,24–0,79 m.

Tego dotyczą wszystkie liczby w §1–§3 i §5 `STAN_WYNIKOW.md` oraz „podłoga szumu frameworka"
(0,0023–0,0073).

### 3.2 Odległość RMS spektrogramów — miara różnicy dwóch ech, bez jednostki fizycznej

Dwa spektrogramy o kształcie (2, 257, 166), czyli 85 324 komórki:

```
d_spec(S1, S2) = sqrt( średnia po komórkach (S1 − S2)² )
```

To **nie jest** ocena modelu. Służy do charakteryzacji silnika akustycznego (rozdział 3 pracy):

| porównanie | typowa wartość | nazwa w dokumentach |
|---|---|---|
| ten sam kąt, szum pojedynczego renderu | 0,0253–0,1345 (census, 1 740 lok.; maks. ≈ 2,1× Δ) | szum renderu, `sigma_1` |
| kąty 0° i 10° | 0,0644 | `SIGNAL_10DEG`, „sygnał 10°" |
| kąty oddalone o 90° | 0,30–0,35 | — |

Źródło zakresu `sigma_1`: `outputs/probe_census/_all_scenes.csv` (kolumna `sigma_1`, sonda
8 renderów). Wcześniej podawane „0,03–0,16" to RMSE **dwóch** renderów tego samego kąta z wczesnej
diagnostyki (≈ √2·`sigma_1`), a nie `sigma_1` — nie cytować.

W `GENERATOR_PARAMS.md`, `RAPORT_SESJI_2026-07-26_29.md` i w `diagnose_rlr_noise*` słowo „RMSE"
oznacza **zawsze** tę wielkość, chyba że zaznaczono inaczej.

---

## 4. Oznaczenia warunków eksperymentalnych

### 4.1 Rdzeń — trzy warunki

| warunek | nazwa w kodzie | kątów na lokalizację | próbek treningowych | co izoluje |
|---|---|---|---|---|
| `A` | `cardinal` | 4 stałe: 0°, 90°, 180°, 270° (jak Gao) | 5 496 | baseline |
| `D` | `random_4` | 4 losowane z 36, inne w każdej lokalizacji | 5 496 | — |
| `B` | `all` | wszystkie 36 | 49 464 | — |

- **`D − A`** = efekt samej gęstości kątowej (ta sama liczba próbek),
- **`B − D`** = efekt samej ilości danych (to samo pokrycie orientacji),
- **`B − A`** = efekt łączny.

### 4.2 Przedrostki

| przedrostek | znaczenie | przykład |
|---|---|---|
| brak | sieć głębi **pełna** (obraz + echo, architektura Paridy) | `A`, `B`, `D`, `SE` |
| `E` | sieć głębi **tylko echo** (`echo2depth`) | `EA`, `EB`, `ED`, `ESE` |
| `EP` | tylko echo, geometria **`patched`** | `EPA`, `EPB`, `EPD` |
| `EK` | tylko echo, krzywa stałego budżetu (4 próbki/lokalizację, siatka K) | `EK6` … `EK18` |
| `SE` / `ESE` | kontrola: echa **przetasowane** między próbkami, więc nie pasują do miejsca | — |

`SE`/`ESE` to **warunek kontrolny, nie test statystyczny permutacyjny**. Mierzy, ile model zyskuje
z echa: `ESE − EB` = wkład echa.

### 4.3 Zadanie pretekstowe i transfer

| oznaczenie | znaczenie |
|---|---|
| `K4`, `K12`, `K36` | liczba klas obrotu (siatka orientacji w zadaniu pretekstowym) |
| `K36@16par` | kontrola: siatka 36, ale tylko 16 par na lokalizację (tyle co `K4`) |
| `scratch` | zadanie docelowe z losową inicjalizacją enkodera |
| `_f10`, `_f25` | zadanie docelowe na 10 % / 25 % zbioru treningowego |
| `random` / `depth_trained` | w sondowaniu: enkoder losowy (podłoga) / uczony wprost na głębi (górna granica) |

`K36 − K36@16par` izoluje **liczbę par**, `K36@16par − K4` izoluje **rozdzielczość kątową**.

### 4.4 Zbiory testowe, maski, geometria

| oznaczenie | znaczenie |
|---|---|
| `test@36` | wszystkie 36 orientacji 183 lokalizacji testowych (6 588 próbek) — **podstawowy** |
| `test@4` | tylko 4 kierunki kardynalne (732 próbki) — dla zgodności z układem Gao |
| `main` / `patched` | geometria scen oryginalna / z załatanymi dziurami (`GENERATOR_PARAMS.md` §4.5) |
| maska `pełna` | każdy wariant punktowany na **własnych** ważnych pikselach |
| maska `przecięcie` | oba warianty na pikselach ważnych w **obu** — porównywalne wprost |
| maska `ścisła` | przecięcie minus piksele, w których głębia różni się między wariantami |

---

## 5. Statusy

| status | znaczenie |
|---|---|
| **[Z]** | zmierzone: skrypt, surowe wyjście, liczba w dokumencie |
| **[Z-]** | zmierzone z zastrzeżeniem, które trzeba cytować razem z liczbą |
| **[W]** | wywnioskowane z kodu lub arytmetyki, nie z pomiaru |
| **[P]** | przejęte z wcześniejszej fazy projektu, niezweryfikowane ponownie |
| **[X]** | nie sprawdzone — wymienione, żeby nie powstało wrażenie, że było |

---

## 6. Statystyka — co jest czym

Dwa **różne** źródła zmienności, raportowane osobno i nigdy nie łączone:

| źródło | jak mierzone | co mówi |
|---|---|---|
| **ziarna** (inicjalizacja wag, niedeterminizm GPU) | 3–5 treningów tego samego warunku; `± sd` | ile wyniku to „szczęście startu" |
| **lokalizacje** (dobór zbioru testowego) | bootstrap po 183 lokalizacjach; 95 % CI | czy wynik nie zależy od tego, które pokoje trafiły do testu |

- **Test sparowany po ziarnie.** Warunki z tym samym ziarnem mają tę samą inicjalizację, więc
  liczy się różnicę w obrębie ziarna (np. `D − A` dla ziarna 0, 1, 2) i sprawdza, czy te różnice
  są konsekwentnie po jednej stronie zera. Stąd p w tabelach pełnego modelu.
- **Test Welcha.** Porównanie dwóch średnich bez parowania — tam, gdzie parowanie nie ma sensu
  (np. `main` vs `patched`, transfer).
- **Bootstrap sparowany po lokalizacjach.** 1 000–2 000 razy losuje się 183 lokalizacje ze
  zwracaniem, przelicza metrykę **obu** warunków na tej samej próbie i zapisuje różnicę. Środkowe
  95 % tych różnic to przedział ufności. Losuje się lokalizacje, nie próbki, bo 36 próbek jednej
  lokalizacji różni się tylko orientacją i nie jest niezależne.
- **Podłoga szumu frameworka** (0,0023–0,0073 RMSE głębi). Ten sam kod, to samo ziarno, dwa
  uruchomienia na GPU dają nieco inne RMSE. Efekt raportuje się jako „k× podłogi szumu".
- **MAAE** — średni bezwzględny błąd kątowy liczony po okręgu, w stopniach. Poziom losowy 90°
  niezależnie od K, dlatego porównywalny między K (w przeciwieństwie do trafności top-1).

---

## 7. Generator — skróty

| oznaczenie | znaczenie |
|---|---|
| `sigma_1` | szum pojedynczego renderu (odległość RMS spektrogramów, §3.2), własność pozycji |
| `SIGNAL_10DEG` = 0,0644 | różnica ech sąsiednich orientacji (0° vs 10°) |
| `TARGET_SNR` = 3,5 | wymagany stosunek `SIGNAL_10DEG / szum po uśrednieniu`; wybór projektowy |
| `N` | liczba renderów uśrednianych na próbkę; adaptacyjna, `N ∈ [6, 64]` |
| sonda | 8 renderów przy 0° na początku lokalizacji, z których liczy się `sigma_1` i `N` |
| `snr_probe` / `snr_final` | SNR przed / po dorenderowaniu (`GENERATOR_PARAMS.md` §3.4.1) |
| `c_full`, `f_ang`, `bound` | heurystyka alokacji ziaren z 2026-08-11 — **nie wynik**, nie cytować jako liczbę |
