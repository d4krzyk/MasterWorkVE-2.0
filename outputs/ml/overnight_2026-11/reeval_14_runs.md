# Ponowna ewaluacja 14 przebiegów trenowanych 2026-08-11 (stan kodu sprzed commita e5eff1a1)

Metoda: `runs/<id>_reeval` z kopią `config.json` i dowiązaniami do `best_*.pth` oryginału (bez kopiowania wag), `evaluate.py --test-set test@36`, wynik w `eval/<id>_reeval/`; oryginały nietknięte. `ED_seed0/1/2` policzone wcześniej (ten sam sposób); tutaj dołączone z istniejących plików.

| przebieg | RMSE test@36 oryginał | RMSE test@36 re-ewaluacja | różnica | max \|Δ\| per scena | trening (start UTC) | eval oryginału | best_step | ESE: shuffled_echo (oryg./re) |
|---|---|---|---|---|---|---|---|---|
| SE_seed0 | 0.26433380 | 0.26433378 | -2.63e-08 | 9.7e-08 | 2026-08-11 19:05 | 08-11 23:21 | 38000 | False/False |
| B_seed0 | 0.24205105 | 0.24205106 | +7.14e-09 | 1.0e-07 | 2026-08-11 20:01 | 08-11 23:21 | 38000 | False/False |
| EA_seed0 | 0.77895299 | 0.77895297 | -2.12e-08 | 1.2e-07 | 2026-08-11 20:58 | 08-12 13:30 | 2000 | False/False |
| EA_seed1 | 0.79910485 | 0.79910505 | +1.96e-07 | 3.1e-07 | 2026-08-11 21:40 | 08-12 13:30 | 4000 | False/False |
| EA_seed2 | 0.79507569 | 0.79507568 | -1.19e-08 | 7.4e-08 | 2026-08-11 22:21 | 08-12 13:30 | 2000 | False/False |
| ED_seed0 | 0.64695049 | 0.64695050 | +1.48e-08 | 1.5e-07 | 2026-08-11 21:18 | 08-12 13:30 | 9000 | False/False |
| ED_seed1 | 0.64232286 | 0.64232275 | -1.13e-07 | 1.9e-07 | 2026-08-11 22:00 | 08-12 13:30 | 12000 | False/False |
| ED_seed2 | 0.64368941 | 0.64368939 | -2.66e-08 | 2.6e-07 | 2026-08-11 22:42 | 08-12 13:31 | 7000 | False/False |
| EB_seed0 | 0.58311155 | 0.58311158 | +3.55e-08 | 5.0e-08 | 2026-08-11 21:08 | 08-12 13:30 | 36000 | False/False |
| EB_seed1 | 0.58518575 | 0.58518582 | +7.24e-08 | 1.7e-07 | 2026-08-11 21:50 | 08-12 13:30 | 25000 | False/False |
| EB_seed2 | 0.57838601 | 0.57838577 | -2.41e-07 | 2.8e-07 | 2026-08-11 22:31 | 08-12 13:31 | 38000 | False/False |
| ESE_seed0 | 1.16467736 | 1.16467732 | -3.59e-08 | 7.8e-08 | 2026-08-11 21:29 | 08-12 13:30 | 6000 | False/False |
| ESE_seed1 | 1.16533418 | 1.16533403 | -1.51e-07 | 2.7e-07 | 2026-08-11 22:11 | 08-12 13:30 | 15000 | False/False |
| ESE_seed2 | 1.15875979 | 1.15875971 | -8.00e-08 | 1.2e-07 | 2026-08-11 22:52 | 08-12 13:31 | 5000 | False/False |

**Maksymalna różnica |Δ RMSE test@36| w 14 przebiegach: 2.41e-07** (oczekiwane ≈ 1e-7).

Wniosek: ewaluacja wag z 11.08 obecnym kodem i sterownikiem (580.178.04) odtwarza zapisane wyniki; rozbieżność ED_seed0 vs powtórzenia `nf` nie wynika z ewaluacji.
