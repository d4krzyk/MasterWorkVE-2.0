 
# Ustalenia do pracy (zweryfikowane z surowymi plikami, listopad 2026)
Źródło prawdy: STAN_WYNIKOW.md i surowe pliki (outputs/ml/eval/*/eval.json, outputs/probe_census/_all_scenes.csv).

## Generator
- Szum σ₁ (census, 1740 lok.): 0,0253–0,1345, mediana 0,0583; maks. ok. 2,1× Δ=0,0644. Stare „0,03–0,16" to nie σ₁.
- Średnie N: 12,08 (census), 12,13 (gotowy zbiór main). Stałe N o tej gwarancji: 54 (100% lok.), 35 (99%), 25 (95%); N=21 pokrywa 91,21%.
- „2,14×" = 21/9,83 z 12 pozycji sprzed census. Z census: 54/12,13 ≈ 4,45×.
- mp3d: 88,6% dopasowane, 1,9% remis, 9,0% domyślne.
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
