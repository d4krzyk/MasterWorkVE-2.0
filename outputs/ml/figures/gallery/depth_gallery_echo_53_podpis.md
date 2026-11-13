# Galeria echo 5.3: fakty do podpisu

- **Wybór próbek:** reguła, nie ręcznie. W każdej scenie testowej (apartment_2, frl_apartment_5, office_4) próbka o medianie RMSE modelu EB (wersja główna, 3 wiersze) i z 90. percentyla (wersja dodatkowa, 6 wierszy). Tabela i kontrola: `depth_gallery_echo_53_wybor.md`.
- **Modele:** sieci tylko-echo, ziarno 0, najlepszy checkpoint wg walidacji na 36 orientacjach; tryb eval, AMP, maska głębia > 0. Wszystkie oceniane z **prawdziwym echem**, także ESE (jak w `eval.json`: `shuffled_echo = False`).
- **Trening:** EA na 4 orientacjach kardynalnych (0/90/180/270°); ED na 4 orientacjach losowanych z 36; EB na 36 orientacjach; ESE na 36 orientacjach z echem z innej lokalizacji (permutowanym).
- **Skala głębi:** wspólna 0–10 m, wartości powyżej obcięte; skala błędu 0–3 m. Pikseli > 10 m w test@36: 0,019 % (18365/96853180); dla progu 10,29 m 0,014 % (`depth_range_stats.md`). Szary = piksele bez prawdy.
- **RMSE test@36 z `eval.json`** (średnia z 3 ziaren ± odch. std. z próby; pula pikseli całego zbioru):
  - EA: 0,7910 ± 0,0107 m (ziarna: 0,7790, 0,7991, 0,7951)
  - ED: 0,6443 ± 0,0024 m (ziarna: 0,6470, 0,6423, 0,6437)
  - EB: 0,5822 ± 0,0035 m (ziarna: 0,5831, 0,5852, 0,5784)
  - ESE: 1,1629 ± 0,0036 m (ziarna: 1,1647, 1,1653, 1,1588)
- Liczba pod mapą = RMSE tej jednej próbki [m], nie wynik na zbiorze.
