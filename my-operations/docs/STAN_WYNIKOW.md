# Stan wyników — dokument obowiązujący

**To jest jedyny dokument, z którego należy przepisywać liczby do pracy.**

Raporty sesji (`RAPORT_SESJI_*.md`) są **dziennikiem chronologicznym** i zawierają wartości
później zastąpione, oznaczone ramkami „SKORYGOWANE" / „WYCOFANE" / „ZASTĄPIONE". Ten plik zawiera
wyłącznie stan **obowiązujący**, bez historii. Przy rozbieżności między tym plikiem a raportem
sesji — **obowiązuje ten plik**.

Ostatnia aktualizacja: **2026-08-16**, po zamknięciu fazy eksperymentalnej.

**Rysunki** (`outputs/ml/figures/`, odtwarzalne przez `ml/analysis/figures.py`, zero GPU):

| plik | co pokazuje | gdzie w pracy |
|---|---|---|
| `rys_1_krzywa_nasycenia.png` | RMSE w funkcji siatki K, nasycenie przy K = 9–12 | **rysunek główny**, §1 |
| `rys_2_generalizacja_katowa.png` | RMSE w funkcji odległości od siatki treningowej | §1, luka generalizacji |
| `rys_3_rozklad_efektu.png` | gęstość vs ilość danych w obu modelach | §1 i §2 |
| `gallery/depth_gallery.png` | predykcje obok siebie, 6 próbek | rozdział jakościowy |

Każdy punkt ma słupek błędu z rzeczywistego sd po ziarnach — przy 3 ziarnach sd sięga 0,0136 RMSE
(K=18), czyli **więcej niż cała różnica K=18 → K=36**. Gładka linia bez wąsów sugerowałaby
precyzję, której te pomiary nie mają.

Źródła maszynowe: `outputs/ml/echo_ablation/final_results_2026-08-15.json`,
`outputs/ml/echo_ablation/echo_3seeds.json`, `outputs/ml/pretext/summary.json`,
`outputs/ml/thesis_numbers.json`. Pełny wykaz liczb: `LICZBY_DO_PRACY.md`.

Statusy: **[Z]** zmierzone · **[Z-]** zmierzone z zastrzeżeniem, które trzeba cytować razem
z liczbą · **[W]** wywnioskowane z kodu, nie z pomiaru.

---

## 0. Co praca pokazuje — sześć twierdzeń

| # | twierdzenie | liczba | poparcie | status |
|---|---|---|---|---|
| 1 | Gęstość kątowa echa poprawia predykcję głębi | D−A = **0,14672** (echo2depth) | 3 ziarna, CI z bootstrapu wyklucza zero | **[Z]** |
| 2 | Krzywa nasyca się przy **K = 9–12** — dalsze zagęszczanie nie zwraca kosztu | 4→9: **0,128** · 9→36: **0,019** | 3 ziarna, stały budżet próbek, 6 punktów | **[Z]** |
| 3 | Efekt utrzymuje się w pełnym modelu | D−A = **0,02048** | 3 ziarna, p = 0,0096 | **[Z]** |
| 4 | Efekt nie zależy od wariantu geometrii sceny | main −0,14672 vs patched −0,13504 | 3 ziarna, różnica nieistotna (p = 0,26) | **[Z]** |
| 5 | Zadanie pretekstowe orientacji jest rozwiązywalne | MAAE **25,65 ± 0,74°** wobec 90° losowo | 3 ziarna | **[Z]** |
| 6 | Pretrening orientacyjny **nie** przenosi się na predykcję głębi | wszystkie p > 0,07 | 5 ziaren @ 100 %, 3 @ 10 % i 25 % | **[Z]** wynik negatywny |

Twierdzenie 2 jest **głównym wkładem praktycznym** — nikt wcześniej nie mógł go sformułować,
bo nikt nie miał 36 orientacji do podpróbkowania.

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
| **gęstość kątowa** (D − A) | **0,14672** | **70,2 ± 3,0 %** |
| ilość danych (B − D) | 0,06209 | 29,8 % |
| łączny (B − A) | 0,20882 | 100 % |
| całkowity wkład echa (ESE − EB) | 0,58070 | — |

**Gęstość kątowa waży 2,36× więcej niż ilość danych.** Warunek `ED` jest tu kluczowy: ma
**dokładnie tyle samo próbek** co `EA`, więc różnica `D − A` izoluje samą różnorodność kątową.

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

`K=4 → K=36`: **−0,13200**, 95 % CI [−0,15178; −0,11231] (bootstrap sparowany po 183 lokalizacjach).

**Punkt odcięcia:** 4→9 daje **0,128**, a 9→36 tylko **0,019** — przejście do 9 orientacji daje
**6,7× więcej** niż całe pozostałe rozszerzenie do 36.

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
dla każdej lokalizacji. Ten sam koszt generowania, RMSE niższe o 16 %.

### Luka generalizacji kątowej [Z]

`EA` − `EB` na 4 kątach, **które oba modele widziały** (sparowane, 732 próbki): **+0,01787 ±
0,01128**. Na wszystkich 36 kątach: **+0,20882**.

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
część struktury kątowej echa: sam efekt gęstości spada 7,2× (0,147 → 0,020).

**Bramka wykonalności [Z-]:** całkowity wkład echa do pełnego modelu `c_full` = **0,02228**,
95 % CI [0,01840; 0,02643] (warunek `SE`, echo permutowane, **1 ziarno**, CI z bootstrapu po
lokalizacjach). U Gao echo daje 7,5 % — zgodność rzędu wielkości potwierdza poprawność potoku,
**nie jest zestawieniem wyników**.

---

## 3. Wariant geometrii `patched` [Z]

`echo2depth`, 3 ziarna, maska pełna.

| warunek | `main` | `patched` | Δ | p |
|---|---|---|---|---|
| `cardinal` | 0,79104 | 0,78982 | **−0,00123** | 0,870 |
| `all` | 0,58223 | 0,59458 | **+0,01235** | **0,013** |
| `random_4` | 0,64432 | 0,65477 | **+0,01045** | 0,091 |

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

| inicjalizacja | RMSE (5 ziaren) | Δ vs `scratch` | p | odniesienie Gao |
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

| ułamek | próbek | `scratch` | K=4 | K=36 |
|---|---|---|---|---|
| 10 % | 4 946 | 0,35396 | +0,00439 (p = 0,54) | **−0,00005** (p = 0,99) |
| 25 % | 12 366 | 0,30083 | +0,00486 (p = 0,26) | +0,00433 (p = 0,16) |
| 100 % | 49 464 | 0,28986 | −0,00287 (p = 0,23) | +0,00453 (p = 0,21) |

**Δ nie rośnie, gdy zbiór maleje.** Ograniczenie zadziałało jako manipulacja (RMSE rośnie o 22 %),
więc eksperyment miał moc — pretrening po prostu nie zaczyna pomagać.

### Co wiadomo o mechanizmie [Z]

Cztery pomiary, które **pozostają w mocy**:

1. Pretrening realnie przebudowuje enkoder (względna odległość L2 od losowej inicjalizacji:
   K4 0,79 · K12 0,82 · K36 0,97 · K36@16par 0,99).
2. Zadanie docelowe przepisuje go z powrotem (odległość wag końcowych od startowych 0,95–0,98).
3. Końcowe enkodery nie pamiętają startu: dwa przebiegi z **tej samej** inicjalizacji kończą 1,256
   od siebie, pretrenowany od losowego dzieli 1,30–1,38 — **ten sam rząd wielkości**.
4. Przewaga startowa **istnieje i zanika**: na kroku 1 000 K36 jest o 14 % lepszy od `scratch`
   (−0,07325, p = 0,060), przewaga znika około kroku 4 000.

> **Czego NIE wiadomo — i tak trzeba to napisać.** Dlaczego transfer nie działa, pozostaje
> **nierozstrzygnięte**. Wykluczono trzy wyjaśnienia: że pretrening nic nie zmienia w enkoderze,
> że zadanie pretekstowe jest za łatwe, i że zbiór docelowy jest za duży. Otwarta, **niesprawdzona
> [X]** hipoteza: cechy przydatne do przewidywania **obrotu** mogą być po prostu innymi cechami
> niż cechy przydatne do przewidywania **głębi**. Test wymagałby analizy reprezentacji
> (np. sondowania liniowego), a nie kolejnego treningu — materiał na „dalsze prace".

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
| Podłoga szumu frameworka | 0,0023–0,0073 RMSE |
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

---

## 9. Czego świadomie nie policzono

| czego nie ma | powód | czy zostawiać lukę w tekście |
|---|---|---|
| Krzywa na naturalnej liczności (`C6/C9/C12/C18`) | rośnie po gęstości **i** rozmiarze zbioru naraz; krzywa stałego budżetu jest ostrzejsza i jest policzona | **NIE** |
| `patched` na pełnym modelu (`PA/PB/PD`) | wada geometrii jest akustyczna — `geometria_echo` bada ją ostrzej i ~20× taniej | **NIE** |
| Warunek `ESA` (permutacja kątów w obrębie lokalizacji) | rozdzieliłby „echo niesie pozycję" od „echo niesie orientację"; `ESE` odpowiada na słabszą wersję i jest policzony | **NIE** |
| Przyczyna negatywnego transferu | wymaga analizy reprezentacji, nie treningu | **TAK** — „dalsze prace" |
| Sonda geometrii na `office_4` | niski priorytet, nigdy nie był w planie | nie |
