# Stan wyników — dokument obowiązujący

**To jest jedyny dokument, z którego należy przepisywać liczby do pracy.**

Raporty sesji (`RAPORT_SESJI_*.md`) są **dziennikiem chronologicznym** i zawierają wartości
później zastąpione, oznaczone ramkami „SKORYGOWANE" / „WYCOFANE" / „ZASTĄPIONE". Ten plik zawiera
wyłącznie stan **obowiązujący**, bez historii. Przy rozbieżności między tym plikiem a raportem
sesji — **obowiązuje ten plik**.

Ostatnia aktualizacja: **2026-08-17**, po zamknięciu fazy eksperymentalnej (84 przebiegi GPU).
Poprawka redakcyjna **2026-09-21**: ujednolicona konwencja znaku różnic (bez zmiany żadnej wartości
bezwzględnej) — patrz ramka niżej.
Poprawka **2026-11-12** (bez zmiany żadnej zmierzonej wartości): §1 „16 %" → 18,5 % ze średnich
3 ziaren; zastrzeżenia (a)–(c) do twierdzenia 3; mechanizm transferu (§5, §10.1) opisany jako
hipoteza, nie ustalenie.
Aktualizacja **2026-11-13** (bez zmiany żadnej wartości z §0–§10): nowy §11 — poprawka Holma, bootstrap
z n_boot = 2000, rozkład głębi względem okna echa, diagnostyka gałęzi modelu pełnego, niedeterminizm GPU
na pełnym budżecie, tabele §5.3–5.5; §10.2 przeformułowane po pomiarze (okno echa dotyczy ok. 0,01 %
pikseli); dopisane zastrzeżenia w §3, §6, §8, §9.

**Rysunki** (`outputs/ml/figures/`, odtwarzalne przez `ml/analysis/figures.py`, zero GPU):

| plik | co pokazuje | gdzie w pracy |
|---|---|---|
| `rys_1_krzywa_nasycenia.png` | RMSE w funkcji siatki K, nasycenie przy K = 9–12 | **rysunek główny**, §1 |
| `rys_2_generalizacja_katowa.png` | RMSE w funkcji odległości od siatki treningowej | §1, luka generalizacji |
| `rys_3_rozklad_efektu.png` | gęstość vs ilość danych w obu modelach | §1 i §2 |
| `rys_4_sonda_glebi.png` | ile głębi niesie zamrożony koder + rozkład na gęstość / liczbę par | §5, sondowanie |
| `gallery/depth_gallery.png` | predykcje obok siebie, 6 próbek | rozdział jakościowy |
| `rys_1_krzywa_nasycenia_v2.png`, `rys_2_generalizacja_katowa_v2.png`, `rys_3_rozklad_efektu_v2.png` | wersje do pracy: bez tytułów (tytuł w podpisie), 15,5 cm, 300 dpi; dane i skale te same co wyżej | rysunki 1–3 |
| `rys_2b_krzywe_K.png` | RMSE wg odległości od siatki dla K = 4, 6, 9, 12, 18 (średnia ± sd po 3 ziarnach) | §1, luka generalizacji; §11.6 |

Każdy punkt ma słupek błędu z rzeczywistego sd po ziarnach — przy 3 ziarnach sd sięga 0,0136 RMSE
(K=18), czyli **więcej niż cała różnica K=18 → K=36**. Gładka linia bez wąsów sugerowałaby
precyzję, której te pomiary nie mają.

Źródła maszynowe: `outputs/ml/echo_ablation/final_results_2026-08-15.json`,
`outputs/ml/echo_ablation/echo_3seeds.json`, `outputs/ml/pretext/summary.json`,
`outputs/ml/thesis_numbers.json`. Pełny wykaz liczb: `LICZBY_DO_PRACY.md`.

Statusy: **[Z]** zmierzone · **[Z-]** zmierzone z zastrzeżeniem, które trzeba cytować razem
z liczbą · **[W]** wywnioskowane z kodu, nie z pomiaru. Pełny słownik oznaczeń: `KONWENCJE.md`.

> **Konwencja znaku (obowiązuje w całym pliku od 2026-09-21).** Etykieta `X − Y` znaczy dosłownie
> wartość(X) − wartość(Y). Dla RMSE i MAAE **wartość ujemna = mniejszy błąd = lepiej**; przykład:
> `D − A` = 0,64432 − 0,79104 = **−0,14672**. Tam, gdzie podawana jest wielkość poprawy, kolumna
> nazywa się wprost „spadek RMSE" i wartości są dodatnie. Wyjątki z własnymi nazwami (wkład echa,
> luka generalizacji, pp rozpiętości) — `KONWENCJE.md` §2. Wcześniejsza wersja tego pliku podawała
> w §0 i §1 te same liczby bez minusa, choć §2 i §3 miały już minus; wartości bezwzględne się
> nie zmieniły.
>
> **„RMSE" w tym pliku to zawsze RMSE głębi w metrach** (metryka modelu). Odległość RMS
> spektrogramów, używana w `GENERATOR_PARAMS.md` (np. `SIGNAL_10DEG` = 0,0644), to inna wielkość —
> `KONWENCJE.md` §3.

---

## 0. Co praca pokazuje — siedem twierdzeń

| # | twierdzenie | liczba | poparcie | status |
|---|---|---|---|---|
| 1 | Gęstość kątowa echa poprawia predykcję głębi | D − A = **−0,14672** (echo2depth) | 3 ziarna, CI z bootstrapu wyklucza zero | **[Z]** |
| 2 | Krzywa nasyca się przy **K = 9–12** — dalsze zagęszczanie nie zwraca kosztu | spadek RMSE 4→9: **0,128** · 9→36: **0,019** | 3 ziarna, stały budżet próbek, 6 punktów | **[Z]** |
| 3 | Efekt utrzymuje się w pełnym modelu | D − A = **−0,02048** | 3 ziarna, p = 0,0096; **zastrzeżenia (a)–(c) pod tabelą** | **[Z]** |
| 4 | Efekt nie zależy od wariantu geometrii sceny | main −0,14672 vs patched −0,13504 | 3 ziarna, różnica nieistotna (p = 0,26) | **[Z]** |
| 5 | Zadanie pretekstowe orientacji jest rozwiązywalne | MAAE **25,65 ± 0,74°** wobec 90° losowo | 3 ziarna | **[Z]** |
| 6 | Pretrening **uczy koder cech geometrycznych**, ale standardowe dostrajanie nie daje z nich mierzalnej korzyści (przyczyna: hipoteza, §5) | sonda: K36 − random = −0,168 (63,4 % rozpiętości); dostrajanie: wszystkie p > 0,07 | 3 ziarna sondy, 5 ziaren transferu | **[Z]** |
| 7 | **Gęstsze echo kształtuje koder nawet gdy nie poprawia zadania pretekstowego** | gęstość kątowa: p = 0,83 w MAAE, ale **+14,0 pp (p = 0,0042)** w sondzie głębi | 3 ziarna | **[Z]** |

**Zastrzeżenia do twierdzenia 3 (model pełny)** — podawać razem z liczbą:

- **(a)** W warunkach `A` / `D` / `B` **nie ma kontroli RGB-only** (ten sam model bez echa przy tej
  samej siatce kątów), więc efekt gęstości w modelu pełnym **nie jest przypisany samemu echu** —
  siatka kątów zmienia też obrazy RGB widziane w treningu.
- **(b)** Model pełny zawiera `MaterialPropertyNet` (wagi losowe, **bez straty materiałowej** —
  jedyną stratą jest `LogDepthLoss` na głębi) oraz `attentionNet`; ich udziału w efekcie
  **nie rozdzielono**.
- **(c)** Efekt nie występuje istotnie w każdej scenie testowej: `D − A` na `office_4` =
  **−0,01630 ± 0,02188, p = 0,326** (ziarno 0: +0,00013). Na `apartment_2` −0,02465 (p = 0,011),
  na `frl_apartment_5` −0,01775 (p = 0,027); test t na różnicach po 3 ziarnach,
  `LICZBY_DO_PRACY.md`.

Twierdzenie 2 jest **głównym wkładem praktycznym** — nikt wcześniej nie mógł go sformułować,
bo nikt nie miał 36 orientacji do podpróbkowania. Twierdzenia 6 i 7 są **głównym wkładem
poznawczym**: pokazują, *czego* uczy się koder wizualny z echa i dlaczego standardowe przenoszenie
tego nie wykorzystuje. Pełna tabela zamykająca z plikami dowodowymi: `RAPORT_SESJI_2026-08-17.md` §7.

---

## 1. Model głębi, sama gałąź audio (`echo2depth`) [Z]

Wszystko na 3 ziarnach, RMSE `test@36`, wariant geometrii `main`.

| warunek | siatka kątów | próbek treningowych | RMSE |
|---|---|---|---|
| `EA` | 4 kardynalne | 5 496 | 0,79104 ± 0,01066 |
| `ED` | 4 losowe z 36 | 5 496 | 0,64432 ± 0,00238 |
| `EB` | wszystkie 36 | 49 464 | 0,58223 ± 0,00348 |
| `ESE` | 36, echo permutowane (kontrola) | 49 464 | 1,16292 ± 0,00362 |

| składowa | wartość | udział |
|---|---|---|
| **gęstość kątowa** (D − A) | **−0,14672 ± 0,01303** | **70,2 ± 3,0 %** |
| ilość danych (B − D) | −0,06209 ± 0,00435 | 29,8 % |
| łączny (B − A) | −0,20882 ± 0,01132 | 100 % |
| całkowity wkład echa (ESE − EB) | +0,58070 ± 0,00076 | — |

Udział = (D − A) / (B − A); wartość dodatnia wkładu echa znaczy, że model z prawdziwym echem ma
mniejszy błąd niż z przetasowanym.

**Gęstość kątowa waży 2,36× więcej niż ilość danych.** Warunek `ED` jest tu kluczowy: ma
**dokładnie tyle samo próbek** co `EA`, więc różnica `D − A` izoluje samą różnorodność kątową.

> **Terminologia: co znaczy „gęstość kątowa" (nota z 2026-11-13).** W warunku `D` (`random_4`) każda lokalizacja ma
> **4 orientacje losowane z 36** (`angles.select_angles`; losowanie zakotwiczone w hashu
> `f"{angle_seed}|{scene}|{location_id}"`, `angle_seed = 0`), a `A` ma te same 4 orientacje kardynalne we wszystkich
> lokalizacjach. W całym zbiorze treningowym (1 374 lokalizacje, 5 496 próbek, tyle samo w `A` i `D`) w `D` występują
> **wszystkie 36 orientacji**, każda w średnio 11,1 % lokalizacji (min 9,1 %, maks. 13,5 %; oczekiwane 4/36 = 11,1 %).
> Różnica `D − A` mierzy więc **różnorodność orientacji w zbiorze** przy tej samej liczbie próbek i tej samej liczbie ech
> na lokalizację (4), a **nie** liczbę ech na lokalizację. W pracy „gęstość kątowa" oznacza właśnie tę różnorodność.
> W krzywej stałego budżetu (`random_4_of_K`) liczba różnych orientacji w zbiorze to dokładnie K, a ułamek lokalizacji
> z daną orientacją to średnio 4/K: K = 6 → 0,667 (min 0,643, maks. 0,684); K = 9 → 0,444 (0,421–0,467);
> K = 12 → 0,333 (0,308–0,355); K = 18 → 0,222 (0,205–0,241); K = 36 → 0,111 (0,091–0,135); `A` (K = 4) → 1,000 we
> wszystkich lokalizacjach. Źródło: `outputs/ml/overnight_2026-11/angle_diversity.{py,json,md}` (CPU, z kodu i podziału).

### Krzywa nasycenia przy stałym budżecie próbek — **główny rysunek** [Z]

4 próbki na lokalizację (5 496) w **każdym** punkcie; zmienia się wyłącznie siatka K, z której
losowane są kąty. Końcami są `EA` i `ED`.

| siatka K | RMSE | sd (3 ziarna) |
|---|---|---|
| 4 | 0,79104 | 0,01066 |
| 6 | 0,70623 | 0,00192 |
| 9 | 0,66342 | 0,00372 |
| 12 | 0,65331 | 0,00490 |
| 18 | 0,64804 | 0,01362 |
| 36 | 0,64432 | 0,00238 |

`K36 − K4` = **−0,13200**, 95 % CI [−0,15178; −0,11231] (bootstrap sparowany po 183 lokalizacjach,
ziarno 0).

**Punkt odcięcia:** spadek RMSE 4→9 wynosi **0,128**, a 9→36 tylko **0,019** — przejście do
9 orientacji daje **6,7× więcej** niż całe pozostałe rozszerzenie do 36.

> **Zalecenie praktyczne do pracy:** przy stałym koszcie generowania (4 rendery na pozycję)
> losować kąty z siatki **9–12 orientacji** zamiast przybijać je do 4 kierunków kardynalnych.
> Rozszerzanie siatki powyżej 12 nie zwraca kosztu.

**Uwaga terminologiczna — „nasycenie" NIE znaczy „dalej jest gorzej".** RMSE spada monotonicznie
na całej krzywej; najniższe jest przy K = 36. Nasycenie oznacza, że **dalsza poprawa przestaje się
opłacać**, a nie że coś się psuje. Stąd dwa różne zalecenia, których nie wolno mylić:

- **generujesz nowy zbiór przy ograniczonym budżecie** → losuj z siatki 9–12; powyżej 12 nie zwraca kosztu;
- **masz już wyrenderowane 36** → używaj wszystkich 36, bo są minimalnie najlepsze i nic nie tracisz.

Zdanie, którego **nie wolno** napisać: „wystarczy 9–12 orientacji, więcej szkodzi". Nie szkodzi —
przestaje pomagać.

**Co czyni ten wynik darmowym.** Liczba renderów jest identyczna w każdym punkcie krzywej — 4 na
lokalizację. Zmienia się wyłącznie to, **z jakiej siatki losowane są te 4 kąty**: baseline bierze
zawsze te same cztery kierunki kardynalne, warunek gęstszy losuje 4 kąty z szerszej siatki, inne
dla każdej lokalizacji. Ten sam koszt generowania, RMSE niższe o **18,5 %** ze średnich z 3 ziaren
(D − A = −0,14672 przy `EA` = 0,79104; per ziarno 16,9 / 19,6 / 19,0 %). Wcześniej podawane
„16 %" odpowiadało ziarnu 0 (`K36 − K4` = −0,13200 z bootstrapu wyżej) — nie cytować go jako
wartości z 3 ziaren.

### Luka generalizacji kątowej [Z]

`EA − EB` na 4 kątach, **które oba modele widziały** (sparowane, 732 próbki): **+0,01787 ±
0,01128**. Na wszystkich 36 kątach: **+0,20882**. Wartości dodatnie, bo tu odejmowany jest warunek
lepszy od gorszego — to jest wielkość kary warunku `EA`.

**91,4 % kary warunku `EA` powstaje na orientacjach, których nie widział.** Baseline Gao głównie
**nie pokrywa przestrzeni orientacji**, a nie „uczy się gorzej" — ale **nie wolno pisać**, że
modele są na kątach treningowych identyczne. Różnica jest niezerowa (~2,4× podłogi szumu).

---

## 2. Model pełny (obraz + echo + materiał + uwaga) [Z]

3 ziarna, RMSE `test@36`.

| warunek | RMSE | wartości per ziarno |
|---|---|---|
| `A` `cardinal` | 0,29248 ± 0,00488 | 0,28739 · 0,29712 · 0,29292 |
| `D` `random_4` | 0,27199 ± 0,00265 | 0,26909 · 0,27260 · 0,27429 |
| `B` `all` | 0,24367 ± 0,00142 | 0,24205 · 0,24470 · 0,24425 |

| składowa | wartość | p (sparowany) | bootstrap wyklucza zero |
|---|---|---|---|
| **gęstość kątowa** (D − A) | **−0,02048 ± 0,00350** | **0,0096** | **3/3 ziarna** |
| ilość danych (B − D) | −0,02833 ± 0,00153 | **0,0010** | — |
| łączny (B − A) | −0,04881 ± 0,00357 | **0,0018** | **3/3 ziarna** |

Względnie wobec baseline'u 4-kierunkowego: `D` daje **−7,0 %**, `B` daje **−16,7 %**.

### Sformułowanie do pracy — wersja obowiązująca

> Przy stałej architekturze, stałym silniku akustycznym i stałym budżecie optymalizacji rozszerzenie
> siatki orientacji z 4 kierunków kardynalnych do 36 obniża RMSE predykcji głębi o **16,7 %**
> (0,29248 → 0,24367; p = 0,0018), z czego **7,0 punktu procentowego pochodzi z samej gęstości
> kątowej**, przy niezmienionej liczbie próbek treningowych.

**To zdanie celowo nie wspomina Gao — i to jest jego siła.** Baseline 4-kierunkowy jest tu
odtworzony **wewnętrznie**, jako warunek `A` we własnym zbiorze i własnym protokole, a nie
przepisany z cudzej tabeli. Porównanie jest więc odporne na zarzut o niezgodność silników
akustycznych, architektur i przetworzenia scen. Konstrukcja „własny baseline zamiast cytowanego"
warto nazwać w pracy wprost, bo jest mocniejsza niż porównanie międzypracowe.

> **[!] Czego NIE wolno napisać.** Zestawienia naszego 0,24367 z 0,346 Gao — wyglądałoby na ~30 %
> przewagi i byłoby bezwartościowe, bo miesza **trzy** rzeczy naraz: (1) inny silnik akustyczny
> (sam wariant geometrii zmienia energię pogłosu późnego o 46 %, dwa silniki to różnica większego
> rzędu), (2) **inną architekturę** — nasz model pełny to sieć Paridy, 316,9 M parametrów
> z mechanizmem uwagi i siecią materiału, podczas gdy Gao używał prostszej sieci, więc większość tej
> różnicy to prawdopodobnie architektura, nie kąty, (3) inne przetworzenie scen i inny zbiór
> lokalizacji. Na pytanie „ile z tego to Parida, a ile echo" nie mielibyśmy odpowiedzi.

**Odwrócenie proporcji wobec `echo2depth`:** udział gęstości spada z 70,2 % do **42,0 %** — w pełnym
modelu ilość danych waży więcej niż gęstość. Zgodne z obrazem, w którym prior wizualny przykrywa
część struktury kątowej echa: sam efekt gęstości (spadek RMSE) maleje 7,2× (0,147 → 0,020).

**Bramka wykonalności [Z-]:** całkowity wkład echa do pełnego modelu `c_full` = **0,02228**,
95 % CI [0,01840; 0,02643] (warunek `SE`, echo permutowane, **1 ziarno**, CI z bootstrapu po
lokalizacjach). U Gao echo daje 7,5 % — zgodność rzędu wielkości potwierdza poprawność potoku,
**nie jest zestawieniem wyników**.

---

## 3. Wariant geometrii `patched` [Z]

`echo2depth`, 3 ziarna, maska pełna.

| warunek | `main` | `patched` | Δ = patched − main | p |
|---|---|---|---|---|
| `cardinal` | 0,79104 | 0,78982 | **−0,00123** | 0,870 |
| `all` | 0,58223 | 0,59458 | **+0,01235** | **0,013** |
| `random_4` | 0,64432 | 0,65477 | **+0,01045** | 0,091 |

*Po poprawce Holma w rodzinie tych trzech testów (§11.1): `all` p = 0,038, `random_4` p = 0,18, `cardinal` p = 0,87 — czyli
jedyny efekt geometrii, który przeżywa korektę, jest na granicy istotności; po konserwatywnej korekcie
na wszystkie 17 testów (§11.1) nie przeżywa.*

Domknięcie dziur w geometrii **pogarsza wynik przy gęstej siatce**, mimo że dodaje +46,3 % energii
pogłosu. Mechanizm: obniża **względny kontrast kątowy** pola późnego o 17,5 % (niezależny pomiar
fizyczny), a model uczy się z kontrastu między orientacjami, nie z bezwzględnej energii.

**Przy `cardinal` efektu nie ma** — i to jest spójne z mechanizmem: 4 orientacje mają najmniej
kontrastu kątowego do stracenia. **Nie pisać, że pogorszenie występuje we wszystkich trzech
warunkach.**

### Efekt gęstości w obu geometriach — właściwa wielkość porównywana [Z]

| składowa | `main` | `patched` | różnica | p |
|---|---|---|---|---|
| **gęstość** (D−A) | −0,14672 | −0,13504 | +0,01168 | 0,259 |
| łączny (B−A) | −0,20882 | −0,19524 | +0,01358 | 0,162 |
| ilość danych (B−D) | −0,06209 | −0,06019 | +0,00190 | 0,736 |

Surowe RMSE obu wariantów liczą się na **różnych zbiorach pikseli ważnych**, więc porównywalną
wielkością jest różnica **wewnątrz** geometrii — tam wybór maski skraca się w odejmowaniu.

**Wniosek o gęstości kątowej nie zależy od wariantu geometrii.** Zarzut „efekt bierze się z dziur
w skanach" jest odparty. Osłabienie o 8,0 % w `patched` ma **ten sam kierunek** co przewidywanie
z pomiaru fizycznego (−17,5 %), ale **połowę wielkości** i jest nieistotne — przewidywanie nie
zostało obalone ani potwierdzone ilościowo.

### Wrażliwość na wybór maski — efekt geometrii **nie jest odporny** [Z]

Δ = `patched` − `main` na trzech maskach, 3 ziarna, test sparowany po ziarnie. Maska `pełna`
punktuje każdy wariant na **jego własnych** pikselach ważnych; `przecięcie` i `ścisła` liczą oba
warianty na **dokładnie tych samych** pikselach.

| kontrast | maska pełna | **przecięcie** | ścisła |
|---|---|---|---|
| `EPA − EA` | −0,00123 (p = 0,89) | **−0,00429** (p = 0,57) | +0,00389 (p = 0,56) |
| `EPB − EB` | +0,01235 (p = **0,017**) | +0,00325 (p = 0,21) | +0,00704 (p = 0,065) |
| `EPD − ED` | +0,01045 (p = **0,046**) | +0,00299 (p = 0,41) | +0,00826 (p = 0,11) |

**Dwie rzeczy, które trzeba napisać razem:**

1. **Teza „znak jest odporny na wybór maski" NIE utrzymuje się.** Na 1 ziarnie wszystkie
   9 komórek było dodatnich; na 3 ziarnach **dwie są ujemne** — obie dotyczą `EPA`, które jest
   nierozróżnialne od zera na każdej masce (p ≥ 0,56).
2. **Na masce przecięcia — tej, którą zalecono jako podstawową, bo jest konserwatywna — żaden
   kontrast nie jest istotny** (p = 0,21 / 0,41 / 0,57). Efekt geometrii w dużej mierze **znika**,
   gdy oba warianty punktuje się na identycznym zbiorze pikseli.

> **Ostrożne sformułowanie do pracy:** domknięcie dziur w geometrii pogarsza wynik przy gęstej
> siatce kątów, ale efekt jest **na granicy wykrywalności** i jego wielkość zależy od tego, na
> których pikselach się go mierzy (0,003–0,012 przy podłodze szumu 0,0023–0,0073). Wniosek
> jakościowy — mocniejszy, ale bardziej jednorodny pogłos jest gorszym sygnałem — pozostaje
> spójny z pomiarem fizycznym, ale **nie należy podawać dla niego jednej liczby** bez wskazania maski.

---

## 4. Zadanie pretekstowe orientacji [Z]

Para (widok *i*, echo *j*) → obrót *j−i*. Poziom losowy MAAE = **90° niezależnie od K**, dlatego
metryką porównywalną jest MAAE, a **nie** trafność top-1 (jej poziom losowy spada z 25 % do 2,8 %).

Wszystko na **3 ziarnach**.

| wariant | par/lokalizację | MAAE | wartości per ziarno |
|---|---|---|---|
| K=4 | 16 | 59,94 ± 2,10 | 61,23 · 57,51 · 61,08 |
| K=12 | 144 | 58,73 ± 2,61 | 55,73 · 60,06 · 60,40 |
| **K=36** | 1 296 | **25,65 ± 0,74** | 25,13 · 26,50 · 25,33 |
| K=36 @ 16 par | 16 (kontrola) | 58,70 ± 7,40 | 61,77 · 64,07 · 50,26 |

**Zadanie jest wykonalne** — przy K=36 MAAE spada o **71,5 %** wobec poziomu losowego 90°,
i jest to najstabilniejszy z czterech wariantów (sd 0,74°).

> **Uwaga — poprawka wobec wersji z 1 ziarna.** K=4 (59,94) i K=12 (58,73) są na 3 ziarnach
> **nierozróżnialne**; ich przedziały nakładają się niemal całkowicie. Pojedyncze ziarno dawało
> 61,23 wobec 55,73, co sugerowało poprawę przy przejściu z 4 do 12 klas — **to był artefakt
> jednego przebiegu**. Nie pisać, że MAAE poprawia się monotonicznie z K; poprawa pojawia się
> dopiero przy K=36 i jest ogromna.

| składowa (test sparowany po ziarnie) | wartość | p |
|---|---|---|
| **ilość par** (K36 − K36@16par) | **−33,05 ± 7,05°** | **0,015** |
| **rozdzielczość kątowa** (K36@16par − K4) | −1,24 ± 8,82° | 0,831 |

**Cała przewaga K=36 pochodzi z 81× większej liczby par; rozdzielczość kątowa zadania nie wnosi
nic** — i na 3 ziarnach składowa „ilość par" jest **istotna** (p = 0,015), a składowa
„rozdzielczość" nierozróżnialna od zera (p = 0,83). Bez warunku kontrolnego wniosek brzmiałby
„gęstsza siatka poprawia zadanie pretekstowe" i byłby **fałszywy**.

Sieć porządkuje kąty: przy K=36 **37,8 %** błędów trafia w klasę sąsiednią przy poziomie losowym
5,7 %. Przy K=4 wartość 69,4 % jest myląca — tam poziom losowy to 67 %.

MAAE dla przesunięć ≤ 20° wynosi 26,17°, dla > 20° — 24,98° (ziarno 0): **sieć nie jest bezradna
przy najdrobniejszej granulacji**.

---

## 5. Transfer na zadanie docelowe — **wynik negatywny** [Z]

RGB2Depth **bez audio w czasie testu**; różni się wyłącznie inicjalizacją enkodera.

| inicjalizacja | RMSE (5 ziaren) | Δ = X − `scratch` | p | odniesienie Gao |
|---|---|---|---|---|
| pretrening K=4 | 0,28699 ± 0,00433 | −0,00287 | 0,231 | 0,332 |
| K=36 @ 16 par | 0,28927 ± 0,00340 | −0,00059 | 0,751 | — |
| **`scratch`** | **0,28986 ± 0,00204** | — | — | 0,360 |
| pretrening K=36 | 0,29439 ± 0,00664 | +0,00453 | 0,207 | — |
| pretrening K=12 | 0,29688 ± 0,00657 | +0,00702 | 0,074 | — |

**Żadna różnica nie jest istotna.** Porządek warunków też się nie odtwarza: u Gao poprawa rosła
z liczbą klas, u nas K=36 wypada **gorzej** niż K=4, mimo że rozwiązuje zadanie pretekstowe
nieporównanie lepiej (MAAE 25° wobec 61°).

To jest wynik negatywny **z kontrolą**: warunek `K36@16par` pokazuje, że nawet gdy zadanie
pretekstowe jest równie trudne jak K=4, transfer wychodzi tak samo nijako.

### Reżim małej ilości danych — sprawdzony, **nie pomaga** [Z]

Ograniczany wyłącznie zbiór treningowy (podzbiór stratyfikowany po lokalizacji, stałe ziarno
podzbioru, budżet 40 000 kroków bez zmian). Walidacja i test zawsze pełne.

| ułamek | próbek | `scratch` (RMSE) | K=4 − `scratch` | K=36 − `scratch` |
|---|---|---|---|---|
| 10 % | 4 946 | 0,35396 | +0,00439 (p = 0,54) | **−0,00005** (p = 0,99) |
| 25 % | 12 366 | 0,30083 | +0,00486 (p = 0,26) | +0,00433 (p = 0,16) |
| 100 % | 49 464 | 0,28986 | −0,00287 (p = 0,23) | +0,00453 (p = 0,21) |

**Δ nie rośnie, gdy zbiór maleje.** Ograniczenie zadziałało jako manipulacja (RMSE rośnie o 22 %),
więc eksperyment miał moc — pretrening po prostu nie zaczyna pomagać.

### Sondowanie zamrożonych reprezentacji — pomiar [Z], mechanizm [W] (hipoteza)

Enkoder zamrożony (`requires_grad=False` **oraz** `eval()`, weryfikowane sumą kontrolną obejmującą
bufory BatchNormu), uczony wyłącznie dekoder. 3 ziarna, identyczny dekoder we wszystkich warunkach
przy danym ziarnie.

| enkoder (zamrożony) | RMSE sondy | Δ = X − `random` | p | pokrycie rozpiętości |
|---|---|---|---|---|
| `depth_trained` *(górna granica)* | 0,29234 ± 0,00157 | −0,26462 | 0,0002 | 100 % |
| **`pretext_K36`** | **0,38908 ± 0,00243** | **−0,16787** | **0,0004** | **63,4 %** |
| `pretext_K36@16par` *(kontrola)* | 0,43232 ± 0,00154 | −0,12463 | 0,0012 | 47,1 % |
| `pretext_K4` | 0,46932 ± 0,00535 | −0,08763 | 0,0004 | 33,1 % |
| `random` *(podłoga)* | 0,55695 ± 0,00878 | — | — | 0 % |

**Reprezentacja z pretreningu zawiera dużo informacji o głębi** — 63,4 % drogi od enkodera
losowego do enkodera uczonego wprost na głębi, przy Δ **23× ponad podłogą szumu**.

#### Rozkład przewagi `K=36` nad `K=4` [Z]

Kontrola `K36@16par` (gęsta siatka 36 orientacji, budżet par równy `K=4`) rozdziela dwa źródła:

| składowa | porównanie | ΔRMSE | pp rozpiętości | p |
|---|---|---|---|---|
| **gęstość kątowa danych** | `K36@16par` − `K4` | −0,03700 | **+14,0** | **0,0042** |
| **liczba par** | `K36` − `K36@16par` | −0,04324 | **+16,3** | **0,0001** |

**Oba czynniki wnoszą prawie po równo i oba są istotne.**

> **DYSOCJACJA — do wypunktowania w pracy.** W samym zadaniu pretekstowym gęstsza siatka
> **nie poprawiała wyniku w ogóle** (`K36@16par − K4` = −1,24° MAAE, p = 0,83; §4). W zamrożonym koderze poprawia
> zawartość geometryczną o **14 pp**, mimo że `K36@16par` ma MAAE 58,70° — praktycznie tyle samo
> co `K4` (59,94°). **Gęstsze echo kształtuje koder wizualny nawet wtedy, gdy nie poprawia wyniku
> w zadaniu orientacyjnym.** MAAE zadania pretekstowego **nie jest** wskaźnikiem tego, czego uczy
> się koder. Dla tezy o zagęszczaniu ech to jest wzmocnienie: efekt działa **bezpośrednio**, przez
> różnorodność sygnału treningowego, a nie pośrednio przez lepsze rozwiązanie zadania.

**Kontrola z losowym zamrożonym enkoderem jest tu obowiązkowa.** `random` daje 0,55695, a nie
wartość bliską bezużyteczności, bo `RGBDepthNet` jest U-Netem ze **skrótami** — `conv1feature`
trafia wprost do ostatniej warstwy dekodera. Rozstrzyga **różnica**, nie wartość bezwzględna.

Sondy pomocnicze (liniowa głowa na uśrednionych cechach `conv5`) pokazują, że **nie są to cechy
„rozpoznaj, w którą stronę patrzysz"**:

| zadanie | poziom losowy | `pretext_K36` | `random` | `depth_trained` |
|---|---|---|---|---|
| orientacja bezwzględna (MAAE) | 90,0° | 64,92 ± 0,11 | 70,84 ± 0,38 | **59,61 ± 0,45** |
| tożsamość sceny (top-1) | 6,7 % | 64,0 ± 0,2 % | 60,9 ± 0,3 % | **74,4 ± 0,2 %** |

Przy głębi `K=36` bije losowy o 63,4 % rozpiętości; przy orientacji i tożsamości sceny przewaga
jest niewielka, a **najlepszy jest `depth_trained`** — enkoder, który nigdy nie widział zadania
orientacyjnego. CKA potwierdza: `pretext_K36` jest bliżej `depth_trained` niż `random` na każdej
warstwie, a przewaga **rośnie z głębokością** (0,010 na `conv1` → 0,041 na `conv5`).

> **Wniosek obowiązujący, zastępujący „transfer nie działa":** zadanie pretekstowe uczy koder
> wizualny cech geometrycznych użytecznych dla głębi [Z], a standardowe dostrajanie całej sieci
> przepisuje enkoder (odległość wag końcowych od startowych 0,95–0,98) i nie daje mierzalnej
> korzyści [Z]. **Hipoteza [W]:** wąskim gardłem jest protokół przenoszenia, a nie pretrening.
> Nie jest to ustalone: protokołów chroniących cechy nie zmierzono, a zamrożony koder `K=36`
> (RMSE sondy 0,38908) wypada **gorzej** niż trening od zera nawet na 10 % danych (0,35396).

**Przewidywanie [W], niezmierzone:** protokoły chroniące cechy (zamrożenie enkodera, niższy krok
uczenia na enkoderze, stopniowe odmrażanie) powinny tu działać, bo sonda pokazuje, że jest co
chronić. Kandydat numer jeden do rozdziału o dalszych badaniach.

### Co wiadomo o mechanizmie [Z]

Cztery pomiary, które **pozostają w mocy**:

1. Pretrening realnie przebudowuje enkoder (względna odległość L2 od losowej inicjalizacji:
   K4 0,79 · K12 0,82 · K36 0,97 · K36@16par 0,99).
2. Zadanie docelowe przepisuje go z powrotem (odległość wag końcowych od startowych 0,95–0,98).
3. Końcowe enkodery nie pamiętają startu: dwa przebiegi z **tej samej** inicjalizacji kończą 1,256
   od siebie, pretrenowany od losowego dzieli 1,30–1,38 — **ten sam rząd wielkości**.
4. Przewaga startowa **istnieje i zanika**: na kroku 1 000 K36 jest o 14 % lepszy od `scratch`
   (−0,07325, p = 0,060), przewaga znika około kroku 4 000.

> **Mechanizm — hipoteza z 2026-08-17 [W]**, patrz sekcja „Sondowanie zamrożonych reprezentacji" wyżej.
> Wykluczono cztery wyjaśnienia: że pretrening nic nie zmienia w enkoderze, że zadanie
> pretekstowe jest za łatwe, że zbiór docelowy jest za duży, oraz — **sprawdzone i obalone
> sondowaniem** — że cechy przydatne do przewidywania obrotu są po prostu innymi cechami niż
> cechy przydatne do przewidywania głębi. Ta ostatnia hipoteza była tu wpisana jako otwarta;
> pomiar pokazał, że jest **fałszywa**: cechy pretekstowe pokrywają 63,4 % rozpiętości do
> enkodera uczonego wprost na głębi, a przy orientacji i tożsamości sceny wypadają **gorzej**
> niż enkoder uczony na głębi. Pozostała hipoteza, nieweryfikowana bezpośrednio: wąskim gardłem
> jest protokół dostrajania.

---

## 6. Parametry, które trzeba podać w rozdziale o metodzie [Z]

| parametr | wartość |
|---|---|
| Scen Replica / treningowych / held-out | 18 / 15 / 3 (`apartment_2`, `frl_apartment_5`, `office_4`) |
| Lokalizacji train / val / test | 1 374 / 183 / 183 |
| Odcisk podziału (**zamrożony**) | `e0bf7547668d9e0a` |
| Orientacji na lokalizację | 36 (co 10°) |
| Próbek łącznie, wariant `main` | 62 640 |
| Kształt spektrogramu | (2, 257, 166) — bit-zgodny z `generate_spectrogram()` Paridy |
| `max_depth` (Replica) | 14,104 m |
| Budżet optymalizacji | **40 000 kroków**, batch 32 — **stała liczba kroków, NIE epok** |
| Wybór checkpointu | najlepszy RMSE walidacyjny na pełnych 36 kątach |
| `indirectRayCount` / `threadCount` | 500 / 1 (wątki dzielą budżet promieni) |
| Podłoga szumu frameworka | 0,0023–0,0073 RMSE (2 000 kroków, warunek A); na pełnym budżecie (ED, ziarno 0, 5 powtórzeń): sd **0,00188**, rozstęp **0,00491** (§11.5) |
| Parametry: pełny model / `echo2depth` / pretekst | 316 918 781 / 8 984 073 / 25 733 446 |

**Zasada nadrzędna do wypunktowania w pracy:** każdy warunek dostaje tę samą liczbę kroków
gradientu niezależnie od rozmiaru zbioru. `cardinal` obejdzie swój zbiór ~233 razy, `all` ~26 —
dlatego każdy warunek ma własną walidację i własny wybór checkpointu, a nie ostatni krok.
Przy stałej liczbie **epok** warunek `all` dostałby 9× więcej kroków i wygrałby z tego powodu,
co unieważniłoby cały wniosek o gęstości.

---

## 7. Nazewnictwo — używać opisowego, nie „Model 1 / Model 2"

| w kodzie i raportach | wejście | przewiduje | **nazwa w pracy** |
|---|---|---|---|
| Model 1, wersja `full` | obraz + echo | głębię | **sieć głębi (pełna)** |
| Model 1, wersja `echo2depth` | samo echo | głębię | **sieć głębi (tylko echo)** |
| Model 2 | para (widok, echo) | obrót | **zadanie pretekstowe orientacji** |
| transfer | sam obraz | głębię | **zadanie docelowe (transfer)** |

Numeracja „Model 1 / Model 2" w raportach jest **odwrotna do intuicji**; szczegóły w `MODELE.md`.

---

## 8. Zastrzeżenia, które muszą iść razem z liczbami

- **`c_full` = 0,02228 ma n = 1** — przedział ufności pochodzi z bootstrapu po lokalizacjach,
  nie z rozrzutu po ziarnach.
- **Efekt geometrii `patched` zależy od wyboru maski** i na masce przecięcia nie jest istotny —
  patrz §3. Nie podawać jednej liczby bez wskazania maski.
- **MAAE dla K=4 i K=12 są nierozróżnialne** — nie opisywać jako poprawy z liczbą klas.
- **Kolumna „sam obraz" w galerii NIE mierzy wkładu echa** — modele różnią się obecnością echa,
  mechanizmem uwagi, siecią materiału i 19× liczbą parametrów naraz. Czystym pomiarem jest
  warunek `SE`: **0,0223** [+0,01840; +0,02643].
- **`MaterialPropertyNet` startuje z wag losowych**, nie z ImageNetu. Normalizacja obrazu
  statystykami ImageNetu jest wierna Paridzie, ale **nie** uzasadniona pretrenowanym ResNetem.
- **Odniesienia Gao (0,360 / 0,332 / 0,374 / 0,346) NIE są baseline'em do przepisania** — inny
  silnik akustyczny, inne przetworzenie scen. Służą wyłącznie do sprawdzenia **porządku** warunków
  i **rzędu wielkości** efektu.
- **Wariant `patched` jest mierzalnie bardziej wyidealizowany niż skany** — patrz
  `GENERATOR_PARAMS.md` §4.5.
- **Poprawka Holma (§11.1):** w rodzinach testów żaden wniosek o istotności się nie zmienia, ale po
  konserwatywnej korekcie na wszystkie 17 testów naraz p dla `D − A` w modelu pełnym wynosi 0,096 (nie
  istotne), a `EPB − EB` 0,11. Przy cytowaniu tych dwóch wyników podawać, którą korektę zastosowano.
- **Niedeterminizm GPU (§11.5):** przebiegi tego samego warunku i ziarna różnią się o sd ≈ 0,002
  (rozstęp 0,0049); oryginalny `ED_seed0` leży poza zakresem 5 powtórzeń (−2,9 sd). Różnice rzędu
  0,005 między przebiegami uruchomionymi w różnych terminach nie są efektem warunku. Efekty geometrii
  (0,010–0,012) to tylko 2–2,5× ten rozstęp.
- **Diagnostyka gałęzi (§11.4):** gałęzie `audio_depth` i `img_depth` są uczone wyłącznie jako składniki
  mieszaniny, więc ich RMSE osobno (4,3–4,6 m i 1,4–1,5 m) nie jest „wkładem audio / obrazu". Zerowanie
  i tasowanie cech `MaterialPropertyNet` pogarsza wynik o ok. 0,5 m — attention silnie z nich korzysta,
  ale nie wynika stąd, że cechy kodują materiał (sieć ma wagi losowe i nie ma straty materiałowej).
- **Piksele poza oknem echa (§11.3):** tylko 0,014 % ważnych pikseli test@36 ma głębię > 10,29 m;
  argument o 73 % zakresu głębi dotyczy zakresu, nie pikseli.

---

## 9. Czego świadomie nie policzono

| czego nie ma | powód | czy zostawiać lukę w tekście |
|---|---|---|
| Krzywa na naturalnej liczności (`C6/C9/C12/C18`) | rośnie po gęstości **i** rozmiarze zbioru naraz; krzywa stałego budżetu jest ostrzejsza i jest policzona | **NIE** |
| `patched` na pełnym modelu (`PA/PB/PD`) | wada geometrii jest akustyczna — `geometria_echo` bada ją ostrzej i ~20× taniej | **NIE** |
| Warunek `ESA` (permutacja kątów w obrębie lokalizacji) | rozdzieliłby „echo niesie pozycję" od „echo niesie orientację"; `ESE` odpowiada na słabszą wersję i jest policzony | **NIE** |
| Protokoły chroniące pretrenowane cechy | sonda pokazała, że jest co chronić — samych protokołów nie zmierzono | **TAK** — rozwinięte w §10.1 |
| Sonda geometrii na `office_4` | niski priorytet, nigdy nie był w planie | nie |
| Dłuższe okno czasowe echa (> 60 ms) | wartość odziedziczona z implementacji referencyjnej; zmiana wymaga regeneracji zbioru | **TAK**, ale po pomiarze z niską przesłanką — §10.2, §11.3 |

---

## 10. Dalsze prace — dwie pozycje z dowodem liczbowym

Pozycja 10.1 ma **zmierzoną przesłankę** (sonda), a nie tylko przypuszczenie, i jest tania. Pozycja 10.2 miała
na początku tylko arytmetykę okna; pomiar z 2026-11-13 (§11.3) pokazał, że jej zasięg jest ograniczony do
ułamka promila pikseli. Podawać w tej kolejności i z liczbami — brzmi to inaczej niż ogólnikowe „warto rozszerzyć".

### 10.1 Protokół dostrajania chroniący pretrenowane cechy — **kandydat nr 1** [W]

**Dowód, że jest co zyskać:** zamrożony koder z pretreningu `K=36` niesie **63,4 %** informacji
o głębi względem kodera trenowanego wprost na głębi (Δ = −0,16787 wobec losowego, p = 0,0004,
23× podłoga szumu). Jednocześnie standardowe dostrajanie całej sieci nie daje **nic** (wszystkie
p > 0,07), a odległość wag końcowych od startowych wynosi 0,95–0,98 — **hipoteza [W]: optymalizacja
przepisuje koder i wyrzuca to, co pretrening zbudował**. Przeciwko zbyt mocnemu odczytaniu: sam
zamrożony koder `K=36` (RMSE sondy 0,38908) jest gorszy niż trening od zera na 10 % danych
(0,35396), więc nie wiadomo, ile protokół chroniący cechy faktycznie by odzyskał.

**Co zrobić:** zamrożenie enkodera, niższy krok uczenia na enkoderze albo stopniowe odmrażanie.
To jest zmiana kilku linii w `ml/pretext_model/transfer.py` — infrastruktura sondy
(`probe.py::freeze_encoder`, kontrola sumą kontrolną z buforami BatchNormu) już istnieje
i jest zweryfikowana na 27 przebiegach.

**Dlaczego to jest mocna pozycja:** mamy **zmierzony dowód**, że użyteczna reprezentacja istnieje
(sonda) i że standardowe dostrajanie nie daje z niej korzyści. To, że jest **tracona przez
protokół**, a nie np. po prostu słabsza niż to, co sieć uczy się wprost z danych docelowych,
pozostaje **hipotezą** — rozstrzygnąłby ją dopiero pomiar protokołu chroniącego cechy.

### 10.2 Wydłużenie okna czasowego echa [W] — przesłanka słaba po pomiarze (§11.3)

**Arytmetyka okna** (343 m/s):

| wielkość | wartość |
|---|---|
| okno echa `ECHO_MS = 60` → droga akustyczna | 20,6 m |
| promień dwustronny (echo tam i z powrotem) | **10,29 m** |
| `max_depth` Repliki, którą model ma przewidzieć | **14,104 m** |
| echo od najdalszych powierzchni wraca po | **82,2 ms** |

**Okno pokrywa 73 % zakresu głębi.** Powierzchnie dalsze niż ~10,3 m **nie mają swojego echa
pierwszego rzędu w sygnale w ogóle** — model musi je odtwarzać z pogłosu i z obrazu.

**Ale pomiar rozkładu głębi (§11.3) ogranicza skalę:** dalej niż 10,29 m leży tylko **0,014 %** ważnych
pikseli test@36 (13 456 z 96,85 mln; `frl_apartment_5` 0,036 %, pozostałe dwie sceny ≈ 0) i 0,018 % pikseli
treningowych; dalej niż 8 m — 0,14 %. Wydłużenie okna może więc poprawić wynik najwyżej na ułamku promila
pikseli; wpływ na globalne RMSE byłby bardzo mały. Uzasadnienie o 73 % zakresu dotyczy zakresu głębi, a nie
liczby pikseli — **nie pisać, że okno nie pokrywa istotnej części sceny**.

**To jest ograniczenie ODZIEDZICZONE, nie błąd projektowy.** `audio_length = 0.06` pochodzi wprost
z `beyond-image-to-depth/options/base_options.py:21`; trzymanie tej wartości było **warunkiem
porównywalności** z implementacją referencyjną, a zamrożenie referencji jest fundamentem całej
atrybucji efektu w tej pracy. Zmiana tutaj wymaga ponownego wygenerowania zbioru, ale **nie** ruszania
sieci.

**Przewidywanie do sprawdzenia (po pomiarze: o niskim priorytecie):** zysk powinien być największy na dużych głębiach — czyli dokładnie
tam, gdzie wariant „samo echo" jest dziś najsłabszy. Da się to zweryfikować metryką stratyfikowaną
po odległości, którą `evaluate.py` już liczy.

---

## 11. Dodatkowe kontrole z 2026-11-13 [Z]

Źródła: `outputs/ml/overnight_2026-11/` (nocny zestaw: `holm_families.py`, `depth_range_stats.py`,
`bootstrap_ea_eb.py`, `branch_diagnostics.py`; plan i logi: `PLAN.md`, `PROGRESS.md`, `SUMMARY.md`),
`outputs/ml/tables_5_3_5_5/` (`echo_only_tables.py`), `outputs/ml/determinism/noise_floor_ED_40k.json` i
`noise_floor_ED_seeds12_40k.json`. Żadna wartość z §0–§10 się nie zmieniła.

### 11.1 Poprawka Holma — w rodzinach testów wnioski się nie zmieniają

p surowe liczone dokładnie jak w `final_results.py` (test t po ziarnach lub Welch), korekta Holma w obrębie rodziny.

| rodzina | test | p surowe | p Holm (rodzina) | p Holm (wszystkie 17) |
|---|---|---|---|---|
| (a) tylko-echo | `D − A` | 0,0026 | 0,0033 | 0,031 |
| | `B − D` | 0,0016 | 0,0033 | 0,023 |
| | `B − A` | 0,0010 | 0,0029 | 0,016 |
| (b) krzywa vs K=36 (Welch) | K=6 | 6,0·10⁻⁶ | 2,4·10⁻⁵ | 1,0·10⁻⁴ |
| | K=9 | 0,0032 | 0,0095 | 0,035 |
| | K=12 | 0,0675 | 0,135 | 0,54 |
| | K=18 | 0,685 | 0,685 | 1,0 |
| (c) model pełny | `D − A` | 0,0096 | 0,0096 | **0,096** |
| | `B − D` | 0,0010 | 0,0030 | 0,016 |
| | `B − A` | 0,0018 | 0,0035 | 0,023 |
| (d) geometria (Welch) | `EPA − EA` | 0,87 | 0,87 | 1,0 |
| | `EPD − ED` | 0,091 | 0,18 | 0,54 |
| | `EPB − EB` | 0,0126 | **0,038** | **0,11** |
| (e) transfer vs `scratch` (Welch) | K=4 / K=36@16par / K=36 / K=12 | 0,231 / 0,751 / 0,207 / 0,074 | 0,62 / 0,75 / 0,62 / 0,30 | 1,0 / 1,0 / 1,0 / 0,54 |

**Wniosek:** w rodzinach każdy wynik istotny przed korektą pozostaje istotny, a nieistotny — nieistotny.
Korekta na wszystkie 17 testów naraz jest celowo konserwatywna (testy z różnych pytań nie tworzą jednej
rodziny); przy niej istotność tracą `D − A` w modelu pełnym i `EPB − EB`. Podawać, którą korektę zastosowano.

### 11.2 Bootstrap EA vs EB z n_boot = 2000 (nowe pliki, istniejące `compare_*.json` nietknięte)

`EA − EB`, RMSE test@36, bootstrap sparowany po 183 lokalizacjach, `boot_seed` = 0:

| ziarno | delta | CI 95 % (n_boot = 2000) | CI w dotychczasowym `compare_*.json` |
|---|---|---|---|
| 0 | +0,19584 | [+0,16936; +0,22596] | [+0,17170; +0,22283] (n_boot = 200) |
| 1 | +0,21392 | [+0,18549; +0,24552] | [+0,18604; +0,24450] (n_boot = 1000) |
| 2 | +0,21669 | [+0,19016; +0,24446] | [+0,18985; +0,24497] (n_boot = 1000) |

Wszystkie przedziały wykluczają zero. Do cytowania używać wersji z 2000 losowań
(`bootstrap_EA_vs_EB_seed{0,1,2}_nboot2000.json`, n_boot zapisane w pliku).

### 11.3 Piksele poza zasięgiem okna echa (głębia > 10,29 m)

Odsetek **ważnych** pikseli (depth > 0), wariant `main`, odcisk podziału `e0bf7547668d9e0a`
(`depth_range_stats.json` — liczniki i mianowniki w pliku):

| zbiór | ważnych pikseli | > 8 m | **> 10,29 m** | > 12 m |
|---|---|---|---|---|
| test@36 (6 588 próbek) | 96 853 180 | 0,143 % | **0,014 %** (13 456) | 0,001 % |
| — `apartment_2` | 37 090 393 | 0,031 % | 0,000 % (35) | 0 |
| — `frl_apartment_5` | 37 349 731 | 0,340 % | 0,036 % (13 421) | 0,003 % |
| — `office_4` | 22 413 056 | 0 | 0 | 0 |
| trening, `B` (36 orientacji, 49 464 próbek) | 745 360 080 | 0,196 % | 0,018 % | 0,001 % |
| trening, `A` (kardynalne) / `ED` (`random_4`) | 82,9 mln / 82,8 mln | 0,212 % / 0,197 % | 0,022 % / 0,016 % | 0,001 % |

Pikseli > `max_depth` (14,104 m): test@36 — 0, trening `B` — 131. Konsekwencja dla §10.2: patrz tam.

### 11.4 Diagnostyka gałęzi modelu pełnego (`A`, `D`, `B`, 3 ziarna, test@36)

Ta sama maska i agregacja co `evaluate.py`; kontrola odtworzenia: baseline = zapisane `eval.json`
(A_seed0: 0,28739149 wobec 0,28739153). RMSE [m], średnia po ziarnach:

| | `A` | `D` | `B` |
|---|---|---|---|
| baseline (`depth_predicted`) | 0,2925 | 0,2720 | 0,2437 |
| sama gałąź `audio_depth` | 4,54 | 4,29 | 4,62 |
| sama gałąź `img_depth` | 1,49 | 1,41 | 1,40 |
| średnia α (piksele ważne) | 0,232 | 0,238 | 0,223 |
| α := średnia α (stała) | 0,740 | 0,679 | 0,664 |
| α := 0,5 | 1,754 | 1,647 | 1,797 |
| cechy materiału := 0 | 0,765 | 0,751 | 0,703 |
| cechy materiału przetasowane w batchu | 0,822 | 0,758 | 0,733 |

Odczyt (to są obserwacje, nie wyjaśnienie mechanizmu):
- gałęzie osobno są bardzo słabe, bo sieć jest uczona tylko na mieszaninie `α·audio + (1−α)·obraz`;
- średnia α ≈ 0,23, a stała α pogarsza wynik o 0,41–0,45 m — kluczowy jest **przestrzenny układ** α;
- zerowanie i tasowanie cech `MaterialPropertyNet` pogarsza wynik o 0,46–0,53 m we wszystkich trzech
  warunkach; przy tasowaniu średnia α prawie się nie zmienia (≤ 0,0015), więc psuje się układ α, a nie jej poziom;
- skala szkody jest podobna w `A`, `D`, `B` (tasowanie: +0,53 / +0,49 / +0,49).

Zastrzeżenia: zerowanie cech jest zaburzeniem poza rozkładem (czystsze jest tasowanie); stała α to średnia z
`test@36` tego samego modelu; tasowanie zależy od rozmiaru batcha (32) i zostawia część próbek na miejscu
(np. 206 z 6 588 dla `A_seed0`); pod AMP przepisany forward różni się od `model(dev)` o ≤ 0,004 m (niedeterminizm
cuDNN, bez AMP ~10⁻⁶). Wniosek dla zastrzeżenia (b) z §0: wpływ `MaterialPropertyNet` i attention na wynik jest
**duży**, ale nadal **nierozdzielony** i nie dowodzi, że cechy kodują materiał (wagi losowe, brak straty materiałowej).

### 11.5 Niedeterminizm GPU na pełnym budżecie (40 000 kroków, `ED`, konfiguracja jak oryginał)

Pięć powtórzeń przy ziarnie 0 (`ED_seed0_nf0..4`): RMSE test@36 = 0,65143 / 0,64986 / 0,65477 / 0,65331 / 0,65288,
średnia **0,65245**, sd **0,00188**, rozstęp **0,00491**. Oryginalny `ED_seed0` = 0,64695 leży **poza** zakresem
powtórzeń (−0,0055, −2,9 sd), choć na walidacji (0,63236) mieści się w zakresie powtórzeń (0,62859–0,63508).

| ziarno | oryginał (sierpień) | powtórzenie (listopad) | różnica |
|---|---|---|---|
| 0 | 0,64695 | 0,65245 (średnia z 5) | +0,0055 |
| 1 | 0,64232 | 0,64163 | −0,0007 |
| 2 | 0,64369 | 0,64863 | +0,0049 |

Ponowna ewaluacja **oryginalnych wag** `ED_seed{0,1,2}` obecnym kodem i sterownikiem (580.178.04) daje te same
RMSE (|Δ| ≤ 1,1·10⁻⁷) — ewaluacja jest deterministyczna, rozbieżność powstaje w treningu. `config.json` powtórzeń
różni się od oryginału tylko polem `tag` i datą; żaden kod wpływający na trening ani ewaluację nie zmienił się po sierpniu.
Przyczyna różnicy jest **nieustalona** (hipotezy: zmiana sterownika 580.173.02 → 580.178.04, zwykły rozrzut).
**Wniosek praktyczny:** rozrzut tego samego warunku i ziarna ma sd ≈ 0,002 i rozstęp ≈ 0,005; różnice rzędu 0,005 między
przebiegami z różnych terminów nie są efektem warunku. Efekt gęstości (0,147 w `echo2depth`) jest > 25× większy;
efekty geometrii (0,010–0,012) to zaledwie 2–2,5× rozstęp, co zgadza się z ich granicznym statusem (§3, §11.1).

### 11.6 Tabele §5.3–5.5 (`echo_only_tables.py`)

Wszystkie 14 liczb kontrolnych z §1 zgadza się ze skryptem; poza nimi:
- **`D − A` w `echo2depth` wg sceny** (`ED − EA`; p: test t po ziarnach): `apartment_2` −0,06332 (0,022),
  `frl_apartment_5` −0,26265 (0,0012), `office_4` −0,07417 (0,0016); w modelu pełnym `D − A`: −0,02465 (0,011),
  −0,01775 (0,027), **−0,01630 ± 0,02188 (0,33)** — na `office_4` efekt w modelu pełnym nie jest widoczny.
- **Krzywa K:** udział zysku 4→36 dla K = 6 / 9 / 12 / 18: 57,8 / 87,0 / 93,9 / 97,5 %; kroki między sąsiednimi K
  (K = 4→6→9→12→18→36): −0,0848 / −0,0428 / −0,0101 / −0,0053 / −0,0037, czyli 17,3× / 8,7× / 2,1× / 1,1× / 0,8×
  rozstępu 0,00491 (§11.5). **Kroki powyżej K = 12 mieszczą się w zakresie niedeterminizmu GPU.**
- **Kara wg odległości od siatki** (względem 0°): EA 10° +0,067, 20° +0,186, 30° +0,274, 40° +0,315 (wszystkie p < 0,004);
  luka 0°→max: K=6 (30°) +0,184, K=9 (20°) +0,079, K=12 (10°) +0,036, K=18 (10°) +0,001 (p = 0,47, nieistotna);
  krzywa EA ściśle rosnąca w każdym ziarnie. Odległość liczona od najbliższej orientacji **siatki** K, a nie
  od faktycznie wylosowanych czterech orientacji lokalizacji.


### 11.7 `ED` z różnym `angle_seed` (krok 6; ziarno treningu 0, konfiguracja jak `ED_seed0`) [Z]

RMSE test@36: `angle_seed` 1 = **0,64461**, 2 = **0,66028**, 3 = **0,64627** (`ED_seed0_as1..3`; `angle_seed` 0: oryginał 0,64695,
powtórzenia `nf` średnia 0,65245, sd 0,00188, rozstęp 0,00491 — §11.5). Średnia `as1..3` = 0,65039, sd 0,00861,
rozstęp 0,01568 (≈ 3,2× rozstęp `nf`); średnia `as` − średnia `nf` = −0,00207. Żaden z trzech wyników nie mieści się w zakresie
powtórzeń `nf` (0,64986–0,65477); `as1` i `as3` są poniżej, `as2` powyżej.
Wniosek: losowanie czterech orientacji na lokalizację (`angle_seed`) jest osobnym źródłem rozrzutu, większym niż niedeterminizm
GPU samego treningu, ale ma **n = 3** i zmienia wynik w obu kierunkach (średnio −0,002, w granicach szumu). Rozrzut ±0,008
to nadal < 1/17 efektu gęstości (0,147). Przy porównaniach `D` z jednym `angle_seed` trzeba więc liczyć się z szumem
rzędu 0,01 na pojedynczym przebiegu. Dowody: `outputs/ml/overnight_2026-11/step6_summary.md`, `outputs/ml/eval/ED_seed0_as{1,2,3}/`.

### 11.8 Kontrola RGB-only (krok 7: `RGBDepthNet` od zera na tych samych parach co `A`/`D`/`B`) [Z]

RMSE test@36 [m], średnia ± sd po 3 ziarnach (pełne liczby, test@4 i wartości per ziarno: `overnight_2026-11/checks_after_B.md`):

| warunek | RGB-only | model pełny | pełny − RGB-only (sparowane, p) |
|---|---|---|---|
| A | 0,36365 ± 0,00359 | 0,29248 ± 0,00488 | −0,07117 (0,0002) |
| D | 0,34549 ± 0,00243 | 0,27199 ± 0,00265 | −0,07349 (0,0008) |
| B | 0,28710 ± 0,00345 | 0,24367 ± 0,00142 | −0,04343 (0,0041) |

Kontrasty (X − Y; p: test t po ziarnach, df = 2):

| kontrast | RGB-only | model pełny |
|---|---|---|
| D − A | −0,01817 (p = 0,027) | −0,02048 (p = 0,010) |
| B − D | −0,05839 (p = 0,0006) | −0,02833 (p = 0,0010) |
| B − A | −0,07655 (p = 0,0028) | −0,04881 (p = 0,0018) |

- **Czego nie wolno już powiedzieć:** efekt `D − A` w modelu pełnym (−0,020) istnieje także bez echa (−0,018), więc **na test@36
  prawie cały „efekt gęstości kątowej" `D − A` w modelu pełnym jest nierozróżnialny od efektu różnorodności obrazów** (orientacji
  kamery). Na test@4 `D − A` w RGB-only to −0,0047 (p = 0,21, nieistotne), w modelu pełnym −0,0077 (p = 0,037).
- Udział `(D − A)/(B − A)` per ziarno, test@36: RGB-only 0,193 / 0,287 / 0,223 (śr. 0,235), model pełny 0,404 / 0,468 / 0,383 (śr. 0,418).
  W RGB-only `B − D` (−0,058) jest ponad dwa razy większe niż w modelu pełnym (−0,028): to samo przejście 4 → 36 orientacji w danych
  uczących pomaga samemu obrazowi bardziej niż modelowi z echem — echo „zjada" część zysku, nie dodaje go.
- Wkład echa+materiału+uwagi (pełny − RGB-only) maleje z −0,071 (`A`) / −0,073 (`D`) do −0,043 (`B`).
- RGB-only `B` (0,28710 ± 0,00345) jest gorszy od `SE_seed0` (potasowane echo, 0,26433) o 0,0228 (≈ 6,6 sd RGB-only B; SE n = 1): nawet
  echo bez informacji kątowej poprawia wynik względem braku echa.
- Zastrzeżenia: 3 ziarna (df = 2), brak korekty Holma dla tej rodziny; RGB-only trenowany tym samym protokołem (40 000 kroków,
  batch 32, wybór po walidacji na 36 orientacjach), ewaluowany `evaluate.py`-owym kodem. Dowody: `overnight_2026-11/step7_summary.md`,
  `overnight_2026-11/rgbonly/RGBONLY_*/eval.json`.

### 11.9 Poprawka diagnostyki plateau

`train_condition._plateau_diagnostics` czytała `r["overall"]` zamiast `r["all"]` (pola plateau i `budget_ceiling_warning` w `status.json`
przebiegów sieci głębi były puste/nierzetelne). Poprawione w kodzie (nowe przebiegi); istniejące `status.json` nietknięte. Wartości
przeliczone z `metrics.jsonl`: `overnight_2026-11/plateau_recomputed.md` (43 przebiegi: mediana poprawy po kroku 30 000 = 0,0066,
sd plateau mediana 0,0052; najlepszy krok w ostatniej ćwiartce: wszystkie `A`, `D`, `B`, 2/3 `EB`, `SE`; w pozostałych warunkach echo-only — 0).
