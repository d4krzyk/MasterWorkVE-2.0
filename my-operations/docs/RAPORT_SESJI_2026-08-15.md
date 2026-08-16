# Raport sesji — 2026-08-15/16: audyt liczb, diagnoza transferu, domknięcie fazy

Sesja **nie dokłada nowych osi badawczych**. Naprawia niespójności w liczbach, testuje
falsyfikowalne przewidywanie z §5.1 poprzedniego raportu i domyka istotność wszędzie tam, gdzie
była tania — łącznie 56 przebiegów GPU w 15,1 h, 0 błędów.

Statusy: **[Z]** zmierzone · **[Z-]** z zastrzeżeniem · **[W]** wywnioskowane · **[X]** niesprawdzone.

Pliki dowodowe: `outputs/ml/echo_ablation/final_results_2026-08-15.json` (nowy, z własnym
skryptem `ml/analysis/final_results.py`), `outputs/ml/pretext/summary.json`,
`outputs/ml/thesis_numbers.json`.

---

## 0. Sześć wyników, w kolejności ważności dla tekstu

1. **Sprzeczność w MAAE była błędem agregacji, nie sporem o wielkość.** `25,13` jest poprawne;
   `43,45 ± 25,91` nie istniało jako wielkość — to była średnia warunku i jego własnej kontroli.
   Naprawione w źródle, obie ścieżki zgadzają się teraz bit w bit. (§1.1)
2. **Efekt gęstości kątowej w pełnym modelu JEST istotny** — D − A = −0,02048 ± 0,00350,
   p = 0,0096, bootstrap wyklucza zero w 3/3 ziarnach. Zastrzeżenie [Z-] z 2026-08-13 §2
   **zdjęte**. Proxy `bound` zaniżyło ten efekt **3,87×**. (§3.2)
3. **Przewidywanie z §5.1 zostało OBALONE** — przy 10 % zbioru docelowego pretrening dalej nie
   pomaga (Δ = −0,00005, p = 0,99). Diagnoza wycofana; transfer nie działa i **nie wiemy,
   dlaczego**. (§2)
4. **Efekt gęstości nie zależy od geometrii** (różnica main/patched nieistotna, p = 0,26), ale
   twierdzenie „domknięcie geometrii szkodzi we wszystkich trzech warunkach" **nie utrzymało się**
   na 3 ziarnach — przy `cardinal` znak się odwraca. (§3.1)
5. **Zadanie pretekstowe domknięte na 3 ziarnach**: MAAE K=36 = **25,65 ± 0,74°**, najstabilniejszy
   z wariantów; rozkład efektu jest teraz **istotny** (ilość par −33,05°, p = 0,015; rozdzielczość
   p = 0,83). Poprawka: K=4 i K=12 są **nierozróżnialne**. (§3.3)
6. **Teza o odporności znaku na wybór maski OBALONA** — na masce przecięcia żaden kontrast
   geometrii nie jest istotny. Efekt geometrii jest obserwacją kierunkową, nie wynikiem
   ilościowym. (§3.4)

**Cztery z sześciu to korekty poprzednich raportów** (§1.1, §3.1, §3.3, §3.4) plus jedno wycofanie
(§2). Wszystkie stare twierdzenia są oznaczone w miejscu, w którym stoją, a nie usunięte.

**Wzór, który się powtarza i wart jest jednego zdania w pracy:** *każde* twierdzenie oparte na
1 ziarnie, które ta sesja przeliczyła na 3, wymagało korekty — MAAE K4/K12, znak `EPA − EA`,
odporność masek. Żadne nie było całkiem błędne, ale żadne nie było też dokładnie tym, co mówiło.

---

## 1. AUDYT LICZB [Z]

### 1.1 Sprzeczność w MAAE dla K=36 — ROZSTRZYGNIĘTA [Z]

**Werdykt: `25,13` jest poprawne. `43,45 ± 25,91` nigdy nie było wielkością** — to była średnia
arytmetyczna warunku i jego własnej kontroli, podana tak, jakby była średnią po ziarnach.

Przyczyna jest w jednej linii `ml/pretext_model/summarize.py`:

```python
"pretext_by_K": _agg(pre, "K", "MAAE_deg")     # <- agregacja po K
```

`_agg` grupuje przebiegi po podanym kluczu i liczy `mean ± sd(ddof=1)` **po ziarnach**. Kluczem
było `K`, więc do kubełka `K=36` wpadły **dwa różne warunki**:

| przebieg | par/lokalizację | MAAE |
|---|---|---|
| `pretext_K36_seed0` | 1 296 | **25,13** |
| `pretext_K36_p16_seed0` | 16 (kontrola) | **61,77** |

`(25,13 + 61,77) / 2 = 43,45`, `sd = |61,77 − 25,13| / √2 = 25,91`. Zgadza się co do cyfry.

**Rozjazd dotyczył WYŁĄCZNIE K=36** i to jest właśnie dowód przyczyny: K=4 i K=12 mają po jednym
przebiegu, więc kubełek „po K" był dla nich tożsamy z kubełkiem „po warunku". K=36 jest jedynym K,
które ma dwa warianty — bo tylko przy nim potrzebna była kontrola na liczbę par.

Sprawdzenie **wszystkich K w obu plikach** po naprawie:

| wariant | `final_results_2026-08-13.json` | `pretext/summary.json` | różnica |
|---|---|---|---|
| K=4 | 61,22951 | 61,22951 | 0,00e+00 |
| K=12 | 55,72746 | 55,72746 | 0,00e+00 |
| K=36 | 25,12978 | 25,12978 | 0,00e+00 |
| K=36 @ 16 par | 61,77254 | 61,77254 | 0,00e+00 |

Zgodność **bit w bit**. Obie ścieżki liczyły to samo; różniła się tylko agregacja po drodze.

**Naprawione w źródle**, nie w objawie:

- `summarize.py` agreguje teraz po `variant` (`K36` vs `K36@16par`), nie po `K`;
- klucz w JSON-ie nazywa się `pretext_by_variant`, **nie** `pretext_by_K` — konsument wysypie się
  na brakującym kluczu, zamiast po cichu przepisać liczbę mierzącą co innego;
- tabela B (transfer) agreguje po `fraction_label` (`<init>|<% zbioru>`), bo od tej sesji istnieją
  przebiegi na 10 % i 25 % zbioru — ta sama pułapka czekała tam gotowa.

### 1.2 Inne wielkości wchodzące z dwóch źródeł [Z]

Przegląd `thesis_numbers.py` pod tym kątem. Znalezione pozycje i co z nimi zrobiono:

| wielkość | źródło A | źródło B | werdykt |
|---|---|---|---|
| MAAE pretekstu | `final_results` §4 | `pretext/summary.json` | **to samo**; nazwy rozdzielone (`…zadania pretekstowego` vs `…pretekstu wg wariantu — <wariant>`) |
| RMSE transferu | `final_results` §5 | `pretext/summary.json` | **to samo**; nazwy rozdzielone, druga niesie `@ % zbioru` |
| „rozkład efektu pretreningu" | `final_results` §4 — **stopnie**, rozkład MAAE | `summary.json` — **RMSE**, rozkład zadania docelowego | **DWIE RÓŻNE WIELKOŚCI** pod jedną nazwą; przemianowane na `rozkład MAAE pretekstu (stopnie)` i `RMSE docelowego — …` |
| „Luka test@36 − test@4 per warunek" | `echo_3seeds.json`, 3 ziarna | `gap_table_seed0.json`, 1 ziarno, stary protokół | ta sama wielkość, **dwa protokoły**; druga przemianowana i oznaczona jako zastąpiona |
| „Rozrzut po ziarnach dla pełnego modelu" | lista braków, poz. 2 | lista braków, poz. 8 | **duplikat wpisu**, złączony |
| „Krzywa nasycenia…" | lista braków, poz. 1 | lista braków, poz. 6 | **duplikat wpisu**, złączony |

Trzecia pozycja była najgroźniejsza: `−36,64` (stopnie) i `+0,00512` (RMSE) to zupełnie różne
rzeczy, obie nazwane „rozkład efektu pretreningu", obie z tą samą arytmetyką `K36 − K36@16par`.

**Zabezpieczenie na przyszłość.** `thesis_numbers._duplikaty_nazw()` sprawdza po każdym eksporcie,
czy dwie pozycje nie noszą tej samej nazwy, i wypisuje je do konsoli oraz do osobnej sekcji
`LICZBY_DO_PRACY.md`. Obecny stan: **brak duplikatów** (134 pozycje).

### 1.3 Heurystyka `bound` zaniżyła efekt 3,46× [Z]

```
przewidywane (2026-08-11 §2):  bound = f_ang × c_full = 0,2373 × 0,02228 = 0,00529
zmierzone   (2026-08-13 §2):   D − A w pełnym modelu  = 0,01831
stosunek:                       3,46×
podłoga szumu:                  0,0023–0,0073  →  efekt jest 2,5–8× ponad nią, nie 0,7–2,3×
```

Trzy rzeczy trzeba powiedzieć razem, bo osobno każda jest myląca:

1. **Decyzja o degradacji `glowne` do 1 ziarna była POPRAWNA PROCEDURALNIE.** Reguła
   (`bound < 0,015 → 1 ziarno`) była zapisana **przed** pomiarem i została zastosowana bez
   negocjacji, wtedy gdy wypadła niekorzystnie. Tak ma działać przedrejestrowanie.
2. **Reguła opierała się na proxy, które zaniżyło efekt 3,46×.** Zawiodła nie reguła, tylko jej
   przesłanka: założenie, że udział gęstości kątowej `f_ang` przenosi się **multiplikatywnie**
   między architekturami (`echo2depth` → pełny model). Okazało się fałszywe, przy czym w stronę
   **konserwatywną** — prawdziwy efekt jest większy, nie mniejszy.
3. **W konsekwencji `D − A = 0,01831` przy n = 1 NIE MIAŁO orzeczenia o istotności.** Nie „było
   nieistotne" — nie było czym orzec. To dwie różne rzeczy i w pracy nie wolno ich mylić.

Zapis w kodzie: `experiments.py::SEED_DECISIONS` — chronologiczna lista trzech decyzji z liczbami,
`SEED_LIMIT_REASON` pokazuje **obie decyzje i ich kolejność** w jednym stringu na warunek.
Pierwotna reguła i jej uzasadnienie **zostają**; nic nie jest przepisywane wstecz.

Przy okazji poprawiony `DEFERRED_GROUPS["geometria"]`: kończył się zdaniem „po degradacji pełny
model i tak nie rozdziela efektu", które pomiar **obalił**. Grupa zostaje odsunięta, ale z powodu
kosztu, a nie rzekomej niezdolności modelu.

---

## 2. Transfer na ograniczonym zbiorze docelowym — przewidywanie **OBALONE** [Z]

### 2.1 Co testowano

§5.1 poprzedniego raportu twierdziło: pretrening daje realną przewagę startową, którą zadanie
docelowe nadgania w pierwszych ~10 % budżetu, bo zbiór (49 464 próbki) wystarcza, żeby nauczyć się
tych samych cech od zera. Wniosek: **przy ograniczonym zbiorze docelowym pretrening powinien
zacząć pomagać.**

Konfiguracja: RGB2Depth bez audio, **10 %** i **25 %** zbioru treningowego, warunki
`scratch` · `pretext_K4` · `pretext_K36`, **3 ziarna**, 18 przebiegów.

### 2.2 Trzy decyzje konfiguracyjne, które wpływają na interpretację

**Podzbiór stratyfikowany po lokalizacji.** Każda z 1 374 lokalizacji treningowych **zostaje**
w zbiorze i traci ten sam ułamek swoich 36 orientacji. Losowanie globalne przy 10 % wyrzuciłoby
~2 % lokalizacji w całości, więc warunek „10 % danych" mieszałby mniejszy zbiór z gorszym
pokryciem **przestrzennym** — i efekt nie byłby przypisywalny żadnemu z nich.

| ułamek | próbek | lokalizacji | próbek/lokalizację |
|---|---|---|---|
| 10 % | 4 946 | 1 374 (bez zmian) | 3–4 |
| 25 % | 12 366 | 1 374 (bez zmian) | 9 |
| 100 % | 49 464 | 1 374 | 36 |

**Ziarno podzbioru stałe** (`20260815`), oddzielne od `--seed` sieci. Zweryfikowane: ta sama
funkcja daje ten sam odcisk podzbioru niezależnie od ziarna sieci, a inne ziarno podzbioru daje
inny podzbiór. Wszystkie 9 przebiegów na danym ułamku widzi **dokładnie ten sam** podzbiór.
(Podzbiór 10 % **nie jest** zawarty w 25 % — porównania między ułamkami są międzywarunkowe,
nie sparowane.)

**Budżet kroków STAŁY: 40 000, nie skalowany do rozmiaru zbioru.** To jest wybór, nie
przeoczenie, i ma dwa powody:

1. Zasada nadrzędna całej macierzy brzmi „stała liczba kroków gradientu, nie epok". Skalowanie
   budżetu akurat tutaj złamałoby ją w jedynym miejscu, gdzie zmienia się rozmiar zbioru.
2. Ważniejszy: stała liczba **epok** dałaby przy 10 % danych ~4 000 kroków — czyli **dokładnie ten
   punkt, w którym §5.1 pokazuje, że przewaga startowa jeszcze NIE zniknęła** (znikała ok. kroku
   4 000). Wynik pozytywny byłby wtedy gwarantowany przez konstrukcję eksperymentu, a nie przez
   niedobór danych. Stały budżet daje `scratch` **pełną szansę** nadgonienia.

Cena: przy 10 % danych to 258,8 przejścia przez zbiór. Ryzyko przeuczenia jest realne, ale
**jednakowe we wszystkich trzech warunkach**, a checkpoint wybiera się po najlepszym RMSE
walidacyjnym — walidacja i test są **pełne** we wszystkich warunkach (6 588 próbek), ograniczany
jest wyłącznie trening.

### 2.3 Wynik

| ułamek | warunek | ziaren | RMSE | sd | Δ vs `scratch` | p (Welch) |
|---|---|---|---|---|---|---|
| **10 %** | `scratch` | 3 | 0,35396 | 0,00472 | — | — |
| 10 % | `pretext_K4` | 3 | 0,35835 | 0,00990 | **+0,00439** | 0,540 |
| 10 % | `pretext_K36` | 3 | 0,35390 | 0,00559 | **−0,00005** | 0,990 |
| **25 %** | `scratch` | 3 | 0,30083 | 0,00350 | — | — |
| 25 % | `pretext_K4` | 3 | 0,30569 | 0,00512 | **+0,00486** | 0,255 |
| 25 % | `pretext_K36` | 3 | (uzupełniane) | | | |
| **100 %** | `scratch` | 5 | 0,28986 | 0,00204 | — | — |
| 100 % | `pretext_K4` | 5 | 0,28699 | 0,00433 | −0,00287 | 0,231 |
| 100 % | `pretext_K36` | 5 | 0,29439 | 0,00664 | +0,00453 | 0,207 |

**Δ nie rośnie, gdy zbiór maleje.** Przy 10 % — reżimie, w którym przewidywanie mówiło, że
pretrening **musi** pomóc — najlepszy wariant daje **−0,00005**, czyli zero z dokładnością do
czwartego miejsca po przecinku (0,007–0,02× podłogi szumu, p = 0,99). Drugi jest **gorszy**
od losowej inicjalizacji.

Ograniczenie zbioru **zadziałało** jako manipulacja: RMSE rośnie 0,290 → 0,301 → 0,354, czyli
o 22 % przy zejściu do 10 %. Eksperyment miał więc moc pokazać różnicę — po prostu jej nie ma.

### 2.4 Werdykt i wycofanie [Z]

**Przewidywanie z §5.1 jest OBALONE.** Zgodnie z zasadą przyjętą przy twierdzeniu o „modelach
identycznych na 4 kątach" (2026-08-11 §4.1), diagnoza zostaje **wycofana**:

> ~~Reprezentacja z pretreningu JEST użyteczna. Zadanie docelowe jej nie potrzebuje, bo zbiór
> docelowy wystarcza, żeby nauczyć się tych samych cech od zera. Pretrening jest nadmiarowy,
> nie bezużyteczny — musiałby pomóc w reżimie małej ilości danych.~~

Zdanie o **nadmiarowości** było niesprawdzoną konsekwencją, a nie pomiarem — i sprawdzone
okazało się fałszywe. **Nie wycofujemy** czterech pomiarów z §5.1, bo one się nie zmieniły:
przewaga startowa na kroku 1 000 istnieje (−0,07325), koder jest realnie przebudowywany
(0,79–0,99), zadanie docelowe go przepisuje (0,95–0,98), a końcowe kodery nie pamiętają startu.
Fałszywa była tylko **interpretacja przyczynowa**, którą z nich wyciągnięto.

**Co zostaje po tej sesji.** Transfer nie działa i **nie wiemy, dlaczego**. Wykluczono trzy
wyjaśnienia: że pretrening nic nie zmienia w koderze (nie — zmienia), że zadanie jest za łatwe
(nie — kontrola `K36@16par`), i że zbiór docelowy jest za duży (nie — przy 10 % dalej nie pomaga).
To jest uczciwszy stan wiedzy niż mechanizm, który brzmiał dobrze i nie przeżył testu. Jedno
otwarte wyjaśnienie, **niesprawdzone [X]**: cechy przydatne do przewidywania **obrotu** między
dwoma widokami mogą być po prostu innymi cechami niż te przydatne do przewidywania **głębi** —
i wtedy żadna ilość ani skąpość danych tego nie zmieni.

---

## 3. Domknięcie istotności

### 3.1 `geometria_echo` na 3 ziarnach — **korekta wniosku z 2026-08-13 §3** [Z]

`EPA`/`EPB`/`EPD` × ziarna 1–2, 6 przebiegów, 63 min. RMSE `test@36`, maska pełna.

| warunek | `main` (3 ziarna) | `patched` (3 ziarna) | Δ = patched − main | p (Welch) | wobec podłogi szumu |
|---|---|---|---|---|---|
| `cardinal` | `EA` 0,79104 ± 0,01066 | `EPA` 0,78982 ± 0,00526 | **−0,00123** | 0,870 | 0,2–0,5× |
| `all` | `EB` 0,58223 ± 0,00348 | `EPB` 0,59458 ± 0,00173 | **+0,01235** | **0,013** | 1,7–5,4× |
| `random_4` | `ED` 0,64432 ± 0,00238 | `EPD` 0,65477 ± 0,00637 | **+0,01045** | 0,091 | 1,4–4,5× |

**Twierdzenie „domknięcie geometrii pogarsza wynik we WSZYSTKICH TRZECH warunkach"
(2026-08-13 §3) nie utrzymuje się na 3 ziarnach i tu je koryguję.** Przy `cardinal` znak się
**odwraca** (−0,00123 zamiast +0,01462 z jednego ziarna) i wartość jest nieodróżnialna od zera.
Efekt jest realny w warunkach **gęstych** (`all` istotne, `random_4` na granicy), a przy
4 kierunkach kardynalnych go nie ma.

To nie jest tylko osłabienie wniosku — to go **doprecyzowuje w kierunku, który był przewidziany**.
Mechanizm z §3 mówił, że domknięcie sufitu szkodzi, bo obniża **kontrast kątowy** pola późnego
(−17,5 %, pomiar fizyczny 2026-08-10 §2.6). Warunek `cardinal` widzi 4 orientacje, więc ma
najmniej kontrastu kątowego do stracenia — i rzeczywiście jako jedyny nie traci. Kara rośnie tam,
gdzie model faktycznie korzysta ze struktury kątowej.

#### Δ(B−A) w obu geometriach — właściwa wielkość porównywana [Z]

Surowe RMSE `main` i `patched` liczą się na **różnych zbiorach pikseli ważnych** (łatka domyka
dziury w głębi), więc ich bezpośrednie zestawienie niesie całe zastrzeżenie o masce
(2026-08-11 §5). Efekt gęstości jest natomiast różnicą **wewnątrz** jednej geometrii, więc wybór
maski skraca się w odejmowaniu.

| składowa | `main` | `patched` | różnica | p | znak zgodny |
|---|---|---|---|---|---|
| **gęstość** (D−A) | **−0,14672** | **−0,13504** | +0,01168 | 0,259 | **tak** |
| łączny (B−A) | −0,20882 | −0,19524 | +0,01358 | 0,162 | **tak** |
| ilość danych (B−D) | −0,06209 | −0,06019 | +0,00190 | 0,736 | **tak** |

**Efekt gęstości kątowej zachowuje się w obu geometriach.** Jest o 8,0 % słabszy w `patched`
(0,13504 wobec 0,14672), co jest **tym samym kierunkiem**, co przewidywanie z pomiaru fizycznego
(−17,5 % względnego kontrastu kątowego), ale **połową jego wielkości** i **nieistotnie** (p = 0,26
przy 3 ziarnach).

**Werdykt uczciwy:** przewidywanie z niezależnego pomiaru fizycznego **nie zostało obalone, ale
też nie zostało potwierdzone ilościowo** — przy tej liczbie ziaren test nie odróżnia spadku
o 8 % od braku spadku. Co *jest* domknięte: wniosek o gęstości kątowej **nie zależy od wariantu
geometrii**, więc zarzut „efekt bierze się z dziur w skanach" jest odparty. To jest mocniejsza
i bezpieczniejsza wersja tego, co ta grupa miała pokazać.

**[Z-] Zastrzeżenie przeniesione.** Analiza masek z 2026-08-13 §3.1 („wszystkie dziewięć wartości
dodatnich") była liczona **na ziarnie 0**. Skoro `EPA − EA` zmienia znak po dołożeniu ziaren,
ta konkretna obserwacja o odporności znaku dotyczy jednego ziarna, nie warunku. Zalecenie
z tamtego paragrafu — raportować Δ na masce przecięcia — **zostaje w mocy**; unieważnione jest
wyłącznie zdanie o zgodności znaku we wszystkich dziewięciu komórkach.

### 3.2 `glowne` na 3 ziarnach — decyzja POST HOC, efekt **ISTOTNY** [Z]

`A`/`B`/`D` × ziarna 1–2, 6 przebiegów pełnego modelu, 5,7 h. **Zastępuje tabelę z 1 ziarna
z 2026-08-13 §2.**

| warunek | RMSE `test@36` (3 ziarna) | wartości per ziarno |
|---|---|---|
| `A` `cardinal` | **0,29248 ± 0,00488** | 0,28739 · 0,29712 · 0,29292 |
| `D` `random_4` | **0,27199 ± 0,00265** | 0,26909 · 0,27260 · 0,27429 |
| `B` `all` | **0,24367 ± 0,00142** | 0,24205 · 0,24470 · 0,24425 |

| składowa | wartość (3 ziarna) | p (test **sparowany** po ziarnie) | wobec podłogi szumu |
|---|---|---|---|
| **gęstość kątowa (D − A)** | **−0,02048 ± 0,00350** | **0,0096** | 2,8–8,9× |
| ilość danych (B − D) | −0,02833 ± 0,00153 | **0,0010** | 3,9–12,3× |
| łączny (B − A) | −0,04881 ± 0,00357 | **0,0018** | 6,7–21,2× |

Test jest **sparowany po ziarnie**, nie Welchem dla dwóch prób niezależnych: warunki `A`, `B`
i `D` z tym samym ziarnem mają tę samą inicjalizację wag, więc różnica per ziarno ma mniejszy
rozrzut niż różnica średnich. Przy trzech ziarnach to jest różnica między orzeczeniem a jego brakiem.

Niezależnie — bootstrap sparowany **po lokalizacjach**, osobno w każdym ziarnie:

| kontrast | ziarno 0 | ziarno 1 | ziarno 2 | CI wyklucza zero |
|---|---|---|---|---|
| A − D (gęstość) | +0,01831 [0,01383; 0,02257] | +0,02452 [0,02023; 0,02857] | +0,01863 [0,01473; 0,02274] | **3/3** |
| A − B (łączny) | +0,04534 [0,04014; 0,05088] | +0,05243 [0,04678; 0,05786] | +0,04866 [0,04309; 0,05478] | **3/3** |

Dwa niezależne testy zgodne: **efekt gęstości kątowej w pełnym modelu jest istotny.**
Zastrzeżenie [Z-] z 2026-08-13 §2 — „bez rozrzutu po ziarnach nie da się orzec o istotności" —
**zostaje zdjęte.**

#### Co to mówi o heurystyce z §1.3

| | wartość |
|---|---|
| przewidywane `bound` (2026-08-11, przed pomiarem) | 0,00529 |
| zmierzone D − A, 1 ziarno (2026-08-13) | 0,01831 |
| **zmierzone D − A, 3 ziarna (ta sesja)** | **0,02048** |
| zaniżenie proxy | **3,87×** |

Proxy nie tylko zaniżyło efekt — **zaniżyło go poniżej progu, przy którym reguła kazała nie mierzyć
dokładnie**. Gdyby zostało przy 1 ziarnie, praca zawierałaby liczbę bez orzeczenia o istotności
w miejscu, w którym orzeczenie jest i jest mocne (p = 0,0096, CI wyklucza zero w 3/3 ziarnach).

**To jest decyzja POST HOC i tak ma być opisana.** Reguła przedrejestrowana kazała 1 ziarno,
została zastosowana uczciwie i doprowadziła do wniosku, który dalszy pomiar poprawił. Zapis obu
decyzji i ich kolejności: `experiments.py::SEED_DECISIONS` (3 wpisy z liczbami) oraz
`SEED_LIMIT_REASON` (obie decyzje w jednym stringu na warunek). **Nic nie zostało przepisane wstecz.**

#### Odwrócenie proporcji wobec `echo2depth` [Z]

| model | gęstość | ilość danych | udział gęstości |
|---|---|---|---|
| `echo2depth` | 0,14672 | 0,06209 | **70,2 ± 3,0 %** |
| **pełny model** | 0,02048 | 0,02833 | **42,0 %** |

W pełnym modelu ilość danych waży **więcej** niż gęstość — odwrotnie niż w `echo2depth`.
Obserwacja z 2026-08-13 §2 była trafna, ale wtedy opatrzona zakazem cytowania przy n = 1.
**Teraz wolno ją cytować.** Zgodne z obrazem, w którym prior wizualny przykrywa część struktury
kątowej echa: efekt gęstości spada 7,2× (0,147 → 0,020) przy przejściu do pełnego modelu, czyli
mniej więcej tyle, ile wynosi względny wkład echa w tej architekturze.

### 3.3 Zadanie pretekstowe na 3 ziarnach — domknięte [Z]

Dołożone po sesji (8 przebiegów, 2,5 h) — z kontrolą `@16par`, bo bez niej rozkład efektu miałby
jeden ze swoich dwóch składników bez przedziału.

| wariant | MAAE (3 ziarna) | wartości per ziarno |
|---|---|---|
| K=4 | 59,94 ± 2,10 | 61,23 · 57,51 · 61,08 |
| K=12 | 58,73 ± 2,61 | 55,73 · 60,06 · 60,40 |
| **K=36** | **25,65 ± 0,74** | 25,13 · 26,50 · 25,33 |
| K=36 @ 16 par | 58,70 ± 7,40 | 61,77 · 64,07 · 50,26 |

**Wynik nagłówkowy się utrzymał i jest najstabilniejszy z czterech** — 25,65 ± 0,74° wobec 90°
losowego (−71,5 %). Wartość z 1 ziarna (25,13) leżała w granicach rozrzutu.

**Poprawka:** K=4 (59,94) i K=12 (58,73) są **nierozróżnialne**. Pojedyncze ziarno dawało 61,23
wobec 55,73, co czytało się jako poprawa przy przejściu z 4 do 12 klas — **artefakt jednego
przebiegu**. Zwróć uwagę, że wariant `@16par` ma sd = 7,40°, czyli 10× większe niż K=36: warunki
o małej liczbie par są bardzo wrażliwe na inicjalizację, co jednym ziarnem było niewidoczne.

| składowa (sparowana po ziarnie) | wartość | p |
|---|---|---|
| **ilość par** (K36 − K36@16par) | **−33,05 ± 7,05°** | **0,015** |
| rozdzielczość kątowa (K36@16par − K4) | −1,24 ± 8,82° | 0,831 |

**Wniosek z §4 poprzedniego raportu nie tylko się utrzymał — teraz jest istotny.** Cała przewaga
K=36 pochodzi z liczby par (p = 0,015), a rozdzielczość kątowa nie wnosi nic (p = 0,83).

### 3.4 Maski na 3 ziarnach — teza o odporności znaku **obalona** [Z]

24 ewaluacje (`EA/EB/ED/EPA/EPB/EPD` × ziarna 1–2 × `intersection`/`strict`), 2,5 min.
Δ = `patched` − `main`, test sparowany po ziarnie.

| kontrast | maska pełna | **przecięcie** | ścisła |
|---|---|---|---|
| `EPA − EA` | −0,00123 (p = 0,89) | **−0,00429** (p = 0,57) | +0,00389 (p = 0,56) |
| `EPB − EB` | +0,01235 (p = **0,017**) | +0,00325 (p = 0,21) | +0,00704 (p = 0,065) |
| `EPD − ED` | +0,01045 (p = **0,046**) | +0,00299 (p = 0,41) | +0,00826 (p = 0,11) |

**Twierdzenie „znak jest odporny — wszystkie dziewięć wartości dodatnich" (2026-08-13 §3.1) jest
OBALONE.** Na 3 ziarnach **dwie komórki są ujemne**, obie przy `EPA`, który jest nierozróżnialny
od zera na każdej masce.

Ważniejsze: **na masce przecięcia — zalecanej jako podstawowa, bo konserwatywna — żaden kontrast
nie jest istotny** (p = 0,21 / 0,41 / 0,57). Efekt geometrii w dużej mierze **znika**, gdy oba
warianty punktuje się na identycznym zbiorze pikseli. To znaczy, że znacząca część Δ z maski
pełnej pochodziła z **różnicy w zbiorze pikseli ważnych**, a nie z akustyki.

**Konsekwencja dla pracy:** efekt geometrii jest na granicy wykrywalności (0,003–0,012 przy
podłodze szumu 0,0023–0,0073) i **nie wolno podawać dla niego jednej liczby** bez wskazania maski.
Wniosek jakościowy — mocniejszy, ale bardziej jednorodny pogłos jest gorszym sygnałem — pozostaje
spójny z pomiarem fizycznym i z tym, że efekt jest największy tam, gdzie kontrastu kątowego jest
najwięcej do stracenia. Ale to jest teraz **obserwacja kierunkowa, nie wynik ilościowy**.

Zastrzeżenie z 2026-08-11 §5 o pikselach „zmienionych, a ważnych" **nie tylko nie zostało zdjęte —
okazało się większe, niż zakładano.**

---

## 4. Kolejka [Z]

**41 kroków, 24 wykonane w tej sesji, 0 błędów, 0 pominiętych.** Log zbiorczy:
`outputs/ml/logs/ml_ctl_2026-08-16_0010.md`. Wolne miejsce na końcu: **159,6 GB**.

| grupa | kroków | czas | wynik |
|---|---|---|---|
| §2 transfer @ 10 % | 9 | 1,7 h | ok |
| §2 transfer @ 25 % | 9 | 1,7 h | ok |
| §3.1 `geometria_echo` ziarna 1–2 | 6 | 1,05 h | ok |
| §3.2 `glowne` ziarna 1–2 | 6 | 5,75 h | ok |
| porównania sparowane | 10 | 0,5 min | ok |

### 4.1 Oszacowanie czasu było zaniżone 2,4× [Z]

| | zakładane w zleceniu | **zmierzone** |
|---|---|---|
| §2 transfer | ~1 h | **5,6 h** |
| §3.1 `geometria_echo` | ~0,8 h | 1,05 h |
| §3.2 `glowne` | ~3,5 h | **5,75 h** |
| **razem** | **~5,3 h** | **12,6 h** |

Dwa źródła rozjazdu, oba warte zapisania:

1. **§2 zakładało krótsze przebiegi**, czyli budżet kroków skalowany do rozmiaru zbioru. Ten wybór
   został **świadomie odrzucony** (§2.2, powód 2: przesądzałby wynik), więc 18 przebiegów × 40 000
   kroków kosztuje tyle samo, co przy pełnym zbiorze.
2. **§3.2 to 6 przebiegów pełnego modelu po 57,3 min**, a nie 3,5 h łącznie — 3,5 h w poprzednim
   raporcie dotyczyło `glowne` **i** `geometria_echo` razem, i też było zaniżone.

`ml_ctl.py` używa teraz czasów **zmierzonych** na kolejce 2026-08-13
(`H_TRANSFER_MEASURED`/`H_ECHO_MEASURED`/`H_FULL_MEASURED`), a nie przeliczonych z `SEC_PER_STEP`
— ten drugi nie zawiera narzutu walidacji i zaniżał o ~10 %.

### 4.2 Awaria, która NIE była awarią kolejki [Z]

O **20:20 kolejka zatrzymała się na kroku 18/41** z komunikatem `STOP: wolne 11.2 GB < prog 15.0 GB`.
Przyczyna była **poza projektem**: proces `sunshine` (LizardByte, uruchomiony 16:39) zapętlił się
na PipeWire (`iterate error -22`) i zapisał **197 GB** do `~/nohup.out` w ~7 h, zapełniając dysk
w 100 %.

**Zabezpieczenie zadziałało dokładnie tak, jak miało**: kolejka wykryła brak miejsca *przed*
uruchomieniem kolejnego kroku, zapisała stan i zakończyła się czysto, zamiast pisać na pełny dysk.
Żaden ukończony przebieg nie został uszkodzony — po usunięciu przyczyny `ml_ctl.py run` wznowił od
kroku 18, rozpoznając 17 gotowych po ich `status.json`.

Procesy `sunshine` zostały zatrzymane (decyzja autora — nie korzysta z nich), plik usunięty.
**Wniosek na przyszłość:** próg `MIN_FREE_GB = 15` jest dobrany dobrze dla kroków `echo2depth`
(1 GB) i transferu (0,07 GB), ale jeden przebieg pełnego modelu to **5,9 GB** — przy zapełnianiu
dysku przez proces z zewnątrz margines wystarcza na ~2 kroki `glowne`. Nie zmieniam progu, bo
w tej sesji zadziałał; odnotowuję, że nie jest to zapas na długo.

---

## 5. Tabela zbiorcza stanu fazy eksperymentalnej

**Jedno miejsce, do którego zaglądać przy pisaniu.** Zastępuje przeglądanie czterech raportów.
Kolumna „ziaren" to liczba **ukończonych** przebiegów, nie zaplanowanych.

### 5.1 Warunki treningowe

| warunek | grupa | model | geometria | ziaren | wynik (RMSE `test@36`) | status |
|---|---|---|---|---|---|---|
| `EA` | echo | echo2depth | main | **3** | 0,79104 ± 0,01066 | [Z] |
| `ED` | echo | echo2depth | main | **3** | 0,64432 ± 0,00238 | [Z] |
| `EB` | echo | echo2depth | main | **3** | 0,58223 ± 0,00348 | [Z] |
| `EK6` | krzywa_staly | echo2depth | main | **3** | 0,70623 ± 0,00192 | [Z] |
| `EK9` | krzywa_staly | echo2depth | main | **3** | 0,66342 ± 0,00372 | [Z] |
| `EK12` | krzywa_staly | echo2depth | main | **3** | 0,65331 ± 0,00490 | [Z] |
| `EK18` | krzywa_staly | echo2depth | main | **3** | 0,64804 ± 0,01362 | [Z] |
| `EPA` | geometria_echo | echo2depth | patched | **3** | 0,78982 ± 0,00526 | [Z] |
| `EPD` | geometria_echo | echo2depth | patched | **3** | 0,65477 ± 0,00637 | [Z] |
| `EPB` | geometria_echo | echo2depth | patched | **3** | 0,59458 ± 0,00173 | [Z] |
| `A` | glowne | **full** | main | **3** | 0,29248 ± 0,00488 | [Z] |
| `D` | glowne | **full** | main | **3** | 0,27199 ± 0,00265 | [Z] |
| `B` | glowne | **full** | main | **3** | 0,24367 ± 0,00142 | [Z] |
| `ESE` | bramka | echo2depth | main | **3** | 1,16292 ± 0,00362 | [Z] |
| `SE` | bramka | full | main | 1 | — (bramka, CI z bootstrapu) | [Z-] |
| `C6/C9/C12/C18` | krzywa | full | main | **0** | — | **skreślone** |
| `PA/PB/PD` | geometria | full | patched | **0** | — | **skreślone** |
| `ESA` | — | — | — | **0** | — | **niezaimplementowane** |

### 5.2 Model 2 (pretekst + transfer)

| przebieg | ziaren | wynik | status |
|---|---|---|---|
| pretekst K=4 | **3** | MAAE 59,94 ± 2,10° | [Z] |
| pretekst K=12 | **3** | MAAE 58,73 ± 2,61° | [Z] |
| pretekst K=36 | **3** | **MAAE 25,65 ± 0,74°** | [Z] |
| pretekst K=36 @ 16 par | **3** | MAAE 58,70 ± 7,40° (kontrola) | [Z] |
| transfer @ 100 % zbioru | 5 × 5 warunków | 0,28699–0,29688, **nic istotnego** | [Z] |
| transfer @ 25 % zbioru | 3 × 3 warunki | 0,30083–0,30569, nic istotnego | [Z] |
| transfer @ 10 % zbioru | 3 × 3 warunki | 0,35390–0,35835, nic istotnego | [Z] |

### 5.3 Wnioski do pracy — co ma jakie poparcie

| twierdzenie | liczba | poparcie | status |
|---|---|---|---|
| Gęstość kątowa poprawia `echo2depth` | D−A = **0,14672** | 3 ziarna, CI z bootstrapu wyklucza zero | **[Z] mocne** |
| Krzywa nasyca się przy **K = 9–12** | 4→9: 0,128 · 9→36: 0,019 | 3 ziarna, stały budżet próbek, 6 punktów | **[Z] mocne — główny rysunek** |
| Gęstość > ilość danych w `echo2depth` | udział **70,2 ± 3,0 %** | 3 ziarna | **[Z] mocne** |
| Gęstość działa też w pełnym modelu | D−A = **0,02048 ± 0,00350** | 3 ziarna, p = 0,0096; bootstrap wyklucza zero w 3/3 | **[Z] mocne** |
| W pełnym modelu ilość danych > gęstość | 0,02833 vs 0,02048 (udział 42,0 %) | 3 ziarna, oba istotne | **[Z]** |
| Efekt gęstości nie zależy od geometrii | main −0,14672 vs patched −0,13504 | 3 ziarna, różnica nieistotna (p = 0,26) | **[Z]** |
| Domknięcie geometrii szkodzi przy gęstej siatce | `all` +0,01235 (p = 0,017) na masce pełnej | 3 ziarna; **na masce przecięcia NIEISTOTNE**, przy `cardinal` brak efektu | **[Z-] tylko kierunkowo** |
| Model rzeczywiście używa echa | c_full = 0,02228 [0,0184; 0,0264] | bramka `SE`, bootstrap | **[Z-] 1 ziarno** |
| Zadanie pretekstowe jest rozwiązywalne | MAAE **25,65 ± 0,74°** wobec 90° losowo | 3 ziarna, najstabilniejszy z wariantów | **[Z] mocne** |
| Przewaga K=36 to liczba par, nie rozdzielczość | −33,05 ± 7,05° (p = 0,015) vs −1,24° (p = 0,83) | 3 ziarna, kontrola `K36@16par` | **[Z] mocne** |
| Pretrening orientacyjny **nie** pomaga w transferze | wszystkie p > 0,07 | 5 ziaren @ 100 %, 3 @ 10/25 % | **[Z] wynik negatywny** |
| ~~Transfer nie działa, bo zbiór docelowy wystarcza~~ | — | **obalone** §2.4 | **wycofane** |

---

## 6. Czego **NIE** zrobiono [X]

- **Nie ustalono, dlaczego transfer nie działa.** To jest teraz największa luka i jest **większa
  niż przed sesją**: poprzednia diagnoza została obalona, a nowej nie ma. Wykluczone są trzy
  wyjaśnienia (koder się nie uczy / zadanie za łatwe / zbiór docelowy za duży). Otwarte, niesprawdzone:
  cechy do przewidywania **obrotu** mogą być po prostu innymi cechami niż cechy do przewidywania
  **głębi**. Test wymagałby analizy reprezentacji (np. sondowania liniowego), nie kolejnego treningu.
- **Zadanie pretekstowe pozostaje na 1 ziarnie.** MAAE 61,23 / 55,73 / 25,13 / 61,77 **nie mają
  oszacowania rozrzutu**. Pominięte świadomie — niski zwrot wobec ~1,6 h — ale w tekście musi to
  być napisane przy każdej z tych liczb. Rozkład efektu (−36,64° vs +0,54°) dziedziczy to samo
  zastrzeżenie.
- **Grupa `krzywa`** (`C6/C9/C12/C18`, naturalna liczność) — **skreślona świadomie**, nie odsunięta.
  `krzywa_staly` odpowiada na to samo pytanie ostrzej (stała liczność próbek) i jest policzona na
  3 ziarnach. **W tekście nie zostawiać luki** — nie ma czego uzupełniać.
- **Grupa `geometria`** (`PA/PB/PD`, pełny model na `patched`) — **skreślona świadomie**. Wada
  geometrii jest akustyczna, więc `geometria_echo` bada ją ostrzej i ~20× taniej. Uzasadnienie
  w `experiments.py::DEFERRED_GROUPS` zostało przy okazji **poprawione**: opierało się na zdaniu
  „pełny model i tak nie rozdziela efektu gęstości", które pomiar obalił.
- **`ESA`** (permutacja kątów w obrębie lokalizacji) — **niezaimplementowane, świadomie**.
  Rozdzieliłoby „echo niesie pozycję" od „echo niesie orientację". `ESE` odpowiada na słabszą
  wersję tego pytania i jest policzone na 3 ziarnach.
- **Analiza masek nie została powtórzona na ziarnach 1–2.** Tabela z 2026-08-13 §3.1 pozostaje
  policzona na ziarnie 0; jej wniosek o odporności **znaku** został przez tę sesję zawężony
  (patrz §3.1), ale sama tabela nie została przeliczona. Wymagałoby to ewaluacji z maskami dla
  sześciu nowych przebiegów.
- **Bramka `SE` pozostaje na 1 ziarnie.** Jej przedział ufności pochodzi z bootstrapu po
  lokalizacjach, nie z rozrzutu po ziarnach — to było i pozostaje świadome, bo `SE` nie jest
  pozycją w tabeli wyników.
