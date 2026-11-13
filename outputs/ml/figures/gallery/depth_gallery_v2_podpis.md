# Galeria v2: fakty do podpisu

- **Wybór próbek:** reguła, nie ręcznie. W każdej z trzech scen testowych (apartment_2, frl_apartment_5, office_4) próbka o medianie RMSE („typowy”) i z 90. percentyla („trudny”), wg RMSE modelu B na tabeli per próbka (test@36). Wybrane próbki i kontrola: `depth_gallery_v2_wybor.md`.
- **Modele:** wszystkie ziarno 0, najlepszy checkpoint wg walidacji na 36 orientacjach; tryb eval, AMP, ta sama maska (głębia > 0) i preprocessing co w ewaluacji.
- **Skala głębi:** wspólna 0–10 m dla wszystkich map głębi, wartości powyżej 10 m obcięte do koloru maksimum. Pikseli z głębią > 10 m w test@36: 0,019 % ważnych pikseli (18365/96853180); dla progu 10,29 m (`depth_range_stats.md`) 0,014 %. Skala błędu: 0–3 m (też obcięta). Szary = piksele bez prawdy (głębia 0).
- **RMSE test@36 z `eval.json`** (średnia z 3 ziaren, ± odch. std. z próby; RMSE puli pikseli całego zbioru, nie tych 6 próbek):
  - obraz+echo, 36 orientacji (B_seed0): 0,2437 ± 0,0014 m (ziarna: 0,2421, 0,2447, 0,2443)
  - obraz+echo, 4 orientacje (A_seed0): 0,2925 ± 0,0049 m (ziarna: 0,2874, 0,2971, 0,2929)
  - samo echo, 36 orientacji (EB_seed0): 0,5822 ± 0,0035 m (ziarna: 0,5831, 0,5852, 0,5784)
  - samo echo, 4 orientacje (EA_seed0): 0,7910 ± 0,0107 m (ziarna: 0,7790, 0,7991, 0,7951)
  - sam obraz, 36 orientacji, RGB-only (RGBONLY_B_seed0): 0,2871 ± 0,0034 m (ziarna: 0,2907, 0,2838, 0,2868)
- **Liczby parametrów:** pełny model (obraz+echo) 316 918 781 ; RGB-only 16 658 561 ; tylko-echo 8 984 073.
- **RGB-only vs pełny:** RGB-only (sama `RGBDepthNet`, krok 7) uczono na tych samych parach (lokalizacja, kąt) co B, tym samym budżetem kroków; różni się od pełnego modelu brakiem echa, materiału i mechanizmu uwagi (a więc nie jest czystym pomiarem wkładu echa).
- Liczba pod mapą predykcji = RMSE tej jednej próbki [m]; nie jest to wynik na zbiorze.
