 
# Ustalenia do pracy (zweryfikowane z surowymi plikami, listopad 2026)
Źródło prawdy: STAN_WYNIKOW.md i surowe pliki (outputs/ml/eval/*/eval.json, outputs/probe_census/_all_scenes.csv).

## Generator
- Szum σ₁ (census, 1740 lok.): 0,0253–0,1345, mediana 0,0583; maks. ok. 2,1× Δ=0,0644. Stare „0,03–0,16" to nie σ₁.
- Średnie N: 12,08 (census), 12,13 (gotowy zbiór main). Stałe N o tej gwarancji: 54 (100% lok.), 35 (99%), 25 (95%); N=21 pokrywa 91,21%.
- „2,14×" = 21/9,83 z 12 pozycji sprzed census. Z census: 54/12,13 ≈ 4,45×.
- mp3d: 89,1% dopasowane, 1,9% remis, 9,0% domyślne (przeliczone z surowych danych 2026-11-13; wcześniej 88,6%, suma 99,5%).
- RT60: 7 z 8 scen szczelnych, powód pominięcia office_4 nie jest nigdzie zapisany.

## Protokół ML
- Trening: 40 000 kroków, batch 32, lr 1e-4, wd 5e-4, Adam (0,9; 0,999), LogDepthLoss na pikselach gt≠0, max_depth 14,104, AMP włączone (43/43 przebiegi głębi).
- Podział lokalizacji 1374/183/183; odcisk e0bf7547668d9e0a we wszystkich 79 eval.json (79 = 43 przebiegi + 36 katalogów mask-*; bez katalogów _nf i plików compare_*); lokalizacji testowych: apartment_2 71, frl_apartment_5 74, office_4 38.
- RMSE w pracy: po pikselach. RMSE per obraz (jak Parida): A 0,25480, D 0,23598, B 0,21361 (średnie z 3 ziaren), tylko do porównań z literaturą.
- Bootstrap: 1000 losowań (EA vs ED), 2000 (A vs D/B, SE vs B). CI dla B−A w sieci tylko-echo bez wiarygodnego źródła (200 losowań albo brak zapisu).
- Podłoga szumu GPU: 0,00732 (jedna para, warunek A, 2000 kroków). Dolna granica 0,0023 bez źródła. Do zastąpienia wynikiem pomiaru noise_floor_ED_40k.

## Wyniki (test@36, 3 ziarna)
- Tylko-echo: EA 0,79104, ED 0,64432, EB 0,58223, ESE 1,16292; D−A = −0,14672 (−18,5%); udział gęstości 70,2 ± 3,0%.
- Krzywa vs K=36 (Welch): K=9 +0,01910 (p=0,0032), K=12 +0,00899 (p=0,0675), K=18 +0,00372 (p=0,6846).
- Model pełny: A 0,29248, D 0,27199, B 0,24367; D−A = −0,02048; test@4: 0,28422 / 0,27657 / 0,24739; office_4 D−A = −0,01630 ± 0,02188 (p=0,326).
- Pretekst K=4: top-1 49,58% (47,92 / 52,08 / 48,74); MAAE K=36 25,65°. Transfer: brak istotnego efektu.
- Patched − main: EPA−EA −0,00123, EPD−ED +0,01045, EPB−EB +0,01235.
- Ziarna A/D/B: 1→3 post hoc (D−A z ziarna 0 = −0,01831; średnia z 3 = −0,02048).

## Architektura i przebiegi
- Model pełny = RGBDepthNet + SimpleAudioDepthNet + MaterialPropertyNet (wagi losowe, bez straty materiałowej) + attentionNet. Tylko-echo = SimpleAudioDepthNet.
- Przebiegi wykonane: 43 (głębia) + 12 (pretekst) + 43 (transfer) + 27 (sondy) = 125. „66" = plan 22×3 z experiments.json (budzet.razem.przebiegow), odtwarzalny. „84" nieodtwarzalne.
- final_results_2026-08-13.json, echo_ablation.json, echo_3seeds.json powstały ad hoc, żaden program w repo ich nie zapisuje.

## Rozstrzygnięte
- Błędne przypisania materiałów: 24,5% = dwa rażące błędy (sufit→Acoustic Tile 11,14% + podłoga→Carpet 13,40%); 29,55% = trzy (z roletami→Glass 5,01%). Praca używa trzech, więc „ok. 30%" jest zgodne (REPLICA_MATERIALS.md §3.1–3.3).
- Przyspieszenie BilinearEinsum: 38,7× sama warstwa (743,4 → 19,2 ms, RAPORT_SESJI_2026-08-05 §3.9); 16,15× pętla treningowa (1,5391 → 0,0953 s/krok, determinism_check.json; dolne oszacowanie, zawiera narzut diagnostyczny); 19,5× iteracja pełnego modelu (1513,5 → 77,57 ms). Do pracy: 16,15× z przypisem o 38,7×.

## Kontrole z 2026-11-13 (STAN_WYNIKOW.md §11)
- Holm w rodzinach: żaden wniosek o istotności się nie zmienia; `EPB−EB` p Holm = 0,038. Konserwatywnie na 17 testów: `D−A` model pełny 0,096, `EPB−EB` 0,11.
- Bootstrap EA vs EB, n_boot = 2000: CI ziarno 0 [+0,16936; +0,22596], ziarno 1 [+0,18549; +0,24552], ziarno 2 [+0,19016; +0,24446]; stare CI (n_boot = 200) było węższe.
- Piksele z głębią > 10,29 m: test@36 0,014 %, trening B 0,018 %; > 8 m: 0,143 %. Argument „73 % zakresu głębi" (§10.2) nie przekłada się na udział pikseli.
- Diagnostyka gałęzi: stała α → RMSE +0,41–0,45 m; cechy materiału zero/przetasowane → +0,46–0,53 m; gałęzie osobno 1,4 m (obraz) i 4,3–4,6 m (audio).
- Niedeterminizm GPU (ED, 40 000 kroków): sd 0,00188, rozstęp 0,00491; oryginalny ED_seed0 poza zakresem 5 powtórzeń (−2,9 sd); ziarna 1 i 2: −0,0007 i +0,0049; ponowna ewaluacja oryginalnych wag |Δ| ≤ 1,1·10⁻⁷.
- Krzywa K: kroki powyżej K = 12 to 2,1× / 1,1× / 0,8× rozstępu 0,00491, czyli w zakresie niedeterminizmu GPU.
- Zweryfikowane i skorygowane: „żadna > 6" dla `snr_final` nie jest prawdą w `main` (6 próbek, maks. 6,21); „43 poprawione kategorie" = 41 kategorii nazwanych z domyślnego na konkretny materiał (51 ze zmienionym przypisaniem).

