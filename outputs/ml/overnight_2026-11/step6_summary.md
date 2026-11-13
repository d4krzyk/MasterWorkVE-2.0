# Krok 6: ED z różnym `angle_seed` (training seed 0, 40 000 kroków, AMP)

| przebieg | angle_seed | RMSE test@36 |
|---|---|---|
| ED_seed0 (oryginał, sierpień) | 0 | 0.64695 |
| ED_seed0_nf0 (powtórzenie) | 0 | 0.65143 |
| ED_seed0_nf1 (powtórzenie) | 0 | 0.64986 |
| ED_seed0_nf2 (powtórzenie) | 0 | 0.65477 |
| ED_seed0_nf3 (powtórzenie) | 0 | 0.65331 |
| ED_seed0_nf4 (powtórzenie) | 0 | 0.65288 |
| ED_seed0_as1 | 1 | 0.64461 |
| ED_seed0_as2 | 2 | 0.66028 |
| ED_seed0_as3 | 3 | 0.64627 |

- powtórzenia `nf` (angle_seed 0): średnia 0.65245, sd 0.00188, rozstęp 0.00491.
- `as1..3` (n = 3): średnia 0.65039, sd 0.00861, rozstęp 0.01568; średnia `as` − średnia `nf` = -0.00207; oryginał − średnia `as` = -0.00344.
- czy wynik mieści się w zakresie powtórzeń `nf`: [('as1', False), ('as2', False), ('as3', False)]

Interpretacja (hipotezy, nie wnioski): jeśli `as1..3` mają rozrzut podobny do `nf`, to losowanie 4 orientacji na lokalizację (angle_seed) nie wnosi wykrywalnej dodatkowej zmienności ponad niedeterminizm GPU; jeśli rozstęp `as` jest wyraźnie większy — angle_seed jest osobnym źródłem rozrzutu.
