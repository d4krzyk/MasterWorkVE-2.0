# Reguła interpretacyjna sondy głębi — ZAPISANA PRZED POMIAREM

Plik utworzony **2026-08-17, przed uruchomieniem kolejki**. Odpowiada §1.1 zlecenia sesji.
Jeśli wynik obali tę regułę, obalenie zostaje zapisane osobno — **reguły nie przepisuje się wstecz**.

## Wielkości

```
S_K36  = RMSE sondy głębi na zamrożonym enkoderze `pretext_K36`
S_K4   = RMSE sondy głębi na zamrożonym enkoderze `pretext_K4`
S_rand = RMSE sondy głębi na zamrożonym enkoderze LOSOWYM  (podłoga)
S_dpth = RMSE sondy głębi na zamrożonym enkoderze `depth_trained` (górna granica)
```

Wszystkie: 3 ziarna, enkoder zamrożony (`requires_grad=False` + `eval()`, weryfikowane sumą
kontrolną obejmującą bufory BatchNormu), uczony wyłącznie dekoder, dekoder **identyczny we
wszystkich warunkach** przy danym ziarnie (`d061b2ed80faa2ca` przy ziarnie 0).

## Werdykty

| wynik | wniosek |
|---|---|
| `S_K36 ≈ S_rand`, oba ≫ `S_dpth` | **cechy orientacyjne nie są cechami głębi** — odpowiedź na pytanie sesji |
| `S_K36` istotnie lepsze od `S_rand`, ale transfer i tak nie pomaga | cechy **są** użyteczne; problem w dynamice dostrajania — wymaga §1.3 |
| `S_K36 ≈ S_dpth` | reprezentacje równoważne; niepowodzenie transferu byłoby artefaktem protokołu |

## Kryterium istotności — obowiązuje łącznie

1. test Welcha po 3 ziarnach, próg `p < 0,05`;
2. **oraz** różnica większa niż górna granica podłogi szumu frameworka = **0,0073 RMSE**.

**Różnica poniżej podłogi szumu to brak różnicy, niezależnie od `p`.** Przy trzech ziarnach test
Welcha ma 2–4 stopnie swobody, więc może dać niskie `p` dla różnicy, która nie ma znaczenia
praktycznego; podłoga szumu jest zabezpieczeniem przed tym.

## Co dokładnie znaczy `≈`

`|różnica| ≤ 0,0073` (podłoga szumu). Powyżej tej wartości mówimy o różnicy i podajemy `p`.

## Zastrzeżenie zapisane z góry

`RGBDepthNet` jest U-Netem z **połączeniami skrótowymi**: `rgbdepth_conv1feature` (64 kanały
w pełnej rozdzielczości) trafia wprost do ostatniej warstwy dekodera. Nawet losowy enkoder podaje
więc dekoderowi użyteczne krawędzie i `S_rand` **nie będzie** bliskie poziomowi losowemu.
Dlatego wielkością rozstrzygającą jest **`S_K36 − S_rand`**, a nie bezwzględna wartość `S_K36`.

---

# Dopisek 2026-08-17: kontrola `K36@16par` — reguła zapisana PRZED pomiarem

Dopisane **po** zobaczeniu wyników pierwszej kolejki, ale **przed** uruchomieniem tej kontroli.

## Co rozdziela

Przewaga `pretext_K36` (63,4 % rozpiętości) nad `pretext_K4` (33,1 %) ma **dwa możliwe źródła**:

- **jakość pretreningu** — K=36 rozwiązuje zadanie znacznie lepiej (MAAE 25,65° wobec 59,94°);
- **liczba par** — K=36 widział 1 296 par na lokalizację, K=4 tylko 16, czyli **81× więcej**.

Warunek `K36@16par` ma **gęstą siatkę 36 orientacji przy budżecie par równym K=4** (16 par)
i osiąga MAAE **58,70°**, czyli praktycznie tyle samo co K=4 (59,94°).

## Przewidywania — rozłączne

| wynik sondy dla `K36@16par` | wniosek |
|---|---|
| ≈ `pretext_K4` (~33 %) | o jakości cech decyduje **jakość rozwiązania zadania pretekstowego** (równoważnie: liczba par). Sama gęstość siatki kątowej w danych **nie wystarcza** |
| ≈ `pretext_K36` (~63 %) | decyduje **gęstość kątowa samych danych echa**, niezależnie od tego, jak dobrze rozwiązane jest zadanie. Mocniejsza wersja tezy z postera |
| pośrednio | oba czynniki wnoszą; podać rozbicie |

`≈` oznacza różnicę nieprzekraczającą podłogi szumu (0,0073 RMSE); powyżej — podajemy `p` z testu
Welcha po 3 ziarnach.

## Zastrzeżenie

`K36@16par` jest jednocześnie **kontrolą z §4 raportu 2026-08-13**, gdzie pokazał, że cała
przewaga K=36 w samym zadaniu pretekstowym pochodzi z liczby par (−33,05°, p = 0,015), a nie
z rozdzielczości kątowej (−1,24°, p = 0,83). Jeśli sonda głębi da ten sam wzorzec, oba pomiary
mówią to samo dwoma niezależnymi drogami. Jeśli da inny — to jest wynik wymagający wyjaśnienia,
nie do przemilczenia.
