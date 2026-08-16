"""Macierz eksperymentow -- Blok 4. Definicja warunkow, NIE uruchamianie.

ZASADA NADRZEDNA: STALA LICZBA KROKOW GRADIENTU, NIE EPOK.

Warunek `cardinal` ma 5 496 probek treningowych, `all` ma 49 464 -- dokladnie
9x wiecej. Przy stalej liczbie EPOK ten drugi dostalby 9x wiecej krokow
optymalizacji i wygralby z tego powodu. Roznica RMSE bylaby wtedy nierozroznialna
od efektu dluzszego treningu, a caly wniosek o gestosci katowej -- niewazny.
Dlatego kazdy warunek dostaje `TOTAL_STEPS` krokow, niezaleznie od tego, ile razy
w tym czasie obejdzie swoj zbior.

Konsekwencja, ktora trzeba wypunktowac w pracy: warunek `cardinal` zobaczy swoj
zbior ~233 razy, a `all` ~26 razy. Rozna liczba powtorzen tej samej probki to
rozne ryzyko przeuczenia -- dlatego kazdy warunek ma wlasna walidacje i wlasny
wybor checkpointu po najlepszym RMSE walidacyjnym, a nie po ostatnim kroku.

DLACZEGO WARUNEK D (`random_4`). Bez niego porownanie A (4 katy) z B (36 katow)
myli dwie rzeczy: gestosc katowa i rozmiar zbioru. D ma DOKLADNIE tyle probek co
A (5 496), ale katy sa losowane per lokalizacja, wiec model widzi w sumie
wszystkie 36 orientacji -- tylko nie wszystkie z kazdego punktu. Roznica D-A
izoluje sama roznorodnosc katowa przy stalej liczbie probek; roznica B-D izoluje
sama ilosc danych. Bez D nie da sie orzec, ktora z tych dwoch rzeczy dziala.

DLACZEGO ECHO2DEPTH. W pelnym modelu obraz RGB niesie wiekszosc informacji o
glebi, a echo dokłada niewiele -- efekt gestosci katowej moze zginac pod priorem
wizualnym. Sama galaz audio (`net_audiodepth`, bez fuzji i bez materialu) jest
najczystszym testem hipotezy: cala informacja o glebi pochodzi wtedy z echa,
wiec kazda poprawa jest przypisana echu bez dyskusji.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..dataset import angles as angles_mod
from .. import paths
from ..dataset.echo_h5_dataset import expected_n_samples
from ..dataset.splits import load_splits

# Budzet optymalizacji wspolny dla WSZYSTKICH warunkow.
TOTAL_STEPS = 40_000
BATCH_SIZE = 32

# Minimum 3 ziarna na warunek. Pojedynczy przebieg nie pozwala orzec o roznicy
# 2-3 % w RMSE -- to mieści sie w rozrzucie samej inicjalizacji wag, ktory przy
# tej architekturze (bilinear 512x512x512, inicjalizacja N(0, 0.02)) nie jest maly.
SEEDS = (0, 1, 2)

# Model: 'full' = pelny AudioVisualModel Paridy (RGB + echo + material + uwaga),
#        'echo2depth' = sama galaz audio.
MODEL_FULL = "full"
MODEL_ECHO = "echo2depth"

# Liczby parametrow ZMIERZONE 2026-08-10 (`sum(p.numel() ...)` na modelach
# zbudowanych przez `ModelBuilder` Paridy z audio_shape=[2,257,166]), a nie
# oszacowane. Trzymane jako stale, bo policzenie ich wymaga zbudowania modeli
# (kilka sekund i import torcha), a `exp_ctl.py plan` ma dzialac natychmiast i
# bez GPU. Weryfikacja: `verify_param_counts()`.
PARAM_COUNTS = {
    MODEL_FULL: 316_918_781,      # rgbdepth 16 658 561 + audio 8 984 073
    MODEL_ECHO: 8_984_073,        # + attention 279 581 505 + material 11 694 642
}

# Zmierzone s/krok przy batchu 32 i --fast-bilinear (raport 2026-08-05 §3.9 dla
# `full`; mikrobenchmark dla `echo2depth`). Bez --fast-bilinear `full` rosnie
# ~19,5x -- patrz `exp_ctl.sec_per_step()`.
SEC_PER_STEP = {MODEL_FULL: 0.07757, MODEL_ECHO: 0.0116}


def verify_param_counts() -> dict:
    """Sprawdza, czy `PARAM_COUNTS` nadal zgadza sie z faktycznymi modelami.

    Osobna funkcja, a nie automatyczne liczenie przy kazdym imporcie: budowanie
    modeli kosztuje kilka sekund, a te liczby zmieniaja sie tylko wtedy, gdy
    zmieni sie architektura referencyjna -- czyli nigdy, bo jej nie wolno ruszac.
    """
    import torch  # noqa: F401  (import lokalny: `plan` ma dzialac bez torcha w hot path)
    paths.add_parida_to_syspath()
    from models.models import ModelBuilder
    b = ModelBuilder()
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    aud = b.build_audiodepth(audio_shape=[2, 257, 166])
    full = n(aud) + n(b.build_rgbdepth()) + n(b.build_attention()) + n(b.build_material_property())
    got = {MODEL_FULL: full, MODEL_ECHO: n(aud)}
    return {"measured": got, "constants": dict(PARAM_COUNTS), "ok": got == PARAM_COUNTS}


@dataclass(frozen=True)
class Condition:
    id: str
    angle_subset: str
    model: str = MODEL_FULL
    geometry: str = "main"
    angle_seed: int = 0
    isolates: str = ""
    group: str = ""
    # Kontrola permutacyjna echa (Blok 1.2). Nie None -> spektrogram brany z
    # innej probki zbioru. Patrz `EchoH5Dataset._echo_permutation`.
    shuffle_echo_seed: int | None = None

    def n_train_samples(self, splits=None) -> int:
        splits = splits or load_splits(variant=self.geometry)
        return expected_n_samples(splits, "train", self.angle_subset)

    def epochs_equivalent(self, splits=None, steps: int = TOTAL_STEPS,
                          batch_size: int = BATCH_SIZE) -> float:
        """Ile razy warunek obejdzie swoj zbior w `steps` krokach."""
        n = self.n_train_samples(splits)
        return steps * batch_size / n


# --------------------------------------------------------------- macierz A/B/D

CONDITIONS: tuple[Condition, ...] = (
    Condition("A", "cardinal", isolates="baseline VisualEchoes (Gao 2020): 4 kierunki kardynalne",
              group="glowne"),
    Condition("B", "all", isolates="efekt laczny gestosci i ilosci danych (36 orientacji)",
              group="glowne"),
    Condition("D", "random_4", isolates="SAMA gestosc katowa przy licznosci probek rownej A",
              group="glowne"),

    # Krzywa nasycenia. Kazdy punkt to podzbior TYCH SAMYCH renderow, wiec
    # roznice miedzy punktami nie moga pochodzic z szumu generatora.
    Condition("C6", "every_6", isolates="krzywa nasycenia: 6 orientacji", group="krzywa"),
    Condition("C9", "every_4", isolates="krzywa nasycenia: 9 orientacji", group="krzywa"),
    Condition("C12", "every_3", isolates="krzywa nasycenia: 12 orientacji", group="krzywa"),
    Condition("C18", "every_2", isolates="krzywa nasycenia: 18 orientacji", group="krzywa"),

    # ECHO2DEPTH -- ta sama siatka katow co warunki glowne, ale bez obrazu.
    Condition("EA", "cardinal", model=MODEL_ECHO,
              isolates="echo2depth, 4 kierunki: hipoteza bez priora wizualnego", group="echo"),
    Condition("EB", "all", model=MODEL_ECHO,
              isolates="echo2depth, 36 orientacji: hipoteza bez priora wizualnego", group="echo"),
    Condition("ED", "random_4", model=MODEL_ECHO,
              isolates="echo2depth, sama gestosc przy licznosci rownej EA", group="echo"),

    # ---------------- BRAMKA WYKONALNOSCI (Blok 1.2) ----------------
    # Warunek, ktory MUSI pojsc jako pierwszy pelny przebieg w ogole. Bez niego
    # wynik zerowy calej macierzy jest nieinterpretowalny: "gestosc katowa nie
    # niesie informacji" i "model w ogole nie uzywa echa" daja te same liczby.
    #   RMSE(SE) - RMSE(B)  =  CALKOWITY wklad echa do pelnego modelu
    #                       =  GORNE OGRANICZENIE na efekt gestosci katowej
    # Punkt odniesienia: u Gao echo daje 7,5 % (0,374 -> 0,346). U Paridy
    # marginalny wklad echa nie jest nigdzie raportowany -- trzeba go zmierzyc.
    Condition("SE", "all", shuffle_echo_seed=20260810,
              isolates="BRAMKA: echo z losowo innej probki -> calkowity wklad echa (vs B)",
              group="bramka"),
    Condition("ESE", "all", model=MODEL_ECHO, shuffle_echo_seed=20260810,
              isolates="BRAMKA: echo2depth z permutowanym echem -> podloga zadania (vs EB)",
              group="bramka"),

    # ---------------- KRZYWA PRZY STALYM BUDZECIE PROBEK (3.1) ----------------
    # Krzywa `every_N` rosnie jednoczesnie po gestosci katowej I po rozmiarze
    # zbioru (8 244 -> 24 732 probek), wiec w duzej mierze pokazuje nasycenie po
    # rozmiarze zbioru -- zjawisko znane i nieciekawe. Ta krzywa trzyma licznosc
    # STALA (5 496 w kazdym punkcie) i zmienia wylacznie roznorodnosc katowa.
    # Koncami sa dokladnie EA (= random_4_of_4) i ED (= random_4_of_36).
    Condition("EK6", "random_4_of_6", model=MODEL_ECHO,
              isolates="staly budzet 4 probek/lok., katy z siatki 6 orientacji", group="krzywa_staly"),
    Condition("EK9", "random_4_of_9", model=MODEL_ECHO,
              isolates="staly budzet 4 probek/lok., katy z siatki 9 orientacji", group="krzywa_staly"),
    Condition("EK12", "random_4_of_12", model=MODEL_ECHO,
              isolates="staly budzet 4 probek/lok., katy z siatki 12 orientacji", group="krzywa_staly"),
    Condition("EK18", "random_4_of_18", model=MODEL_ECHO,
              isolates="staly budzet 4 probek/lok., katy z siatki 18 orientacji", group="krzywa_staly"),

    # Wariant geometrii -- osobna os, poza glowna macierza. Uruchamiac dopiero,
    # gdy os gestosci jest zamknieta; z zastrzezeniem z GENERATOR_PARAMS.md §4.5
    # (sceny zalatane sa mierzalnie bardziej wyidealizowane niz skany).
    Condition("PB", "all", geometry="patched",
              isolates="wplyw domkniecia dziur w geometrii, przy 36 orientacjach", group="geometria"),
    Condition("PA", "cardinal", geometry="patched",
              isolates="wplyw domkniecia dziur, przy 4 kierunkach", group="geometria"),
    # PD BEZ NIEGO replikacja `patched` odtwarza dokladnie te dwuznacznosc, ktora
    # warunek D naprawil w `main`: dostaje sie Delta laczne, nie rozlozone na
    # skladowa gestosci i skladowa ilosci danych.
    Condition("PD", "random_4", geometry="patched",
              isolates="patched: SAMA gestosc przy licznosci rownej PA", group="geometria"),

    # ---------------- REPLIKACJA `patched` NA ECHO2DEPTH (3.3) ----------------
    # Wada geometrii jest wada AKUSTYCZNA: sufit dokłada do obrazu kilka procent
    # pikseli, a do echa cala strukture pogloosu -- `geometry_check.py` zmierzyl
    # +46,3 % energii POZNEJ przy +1,3 % energii calkowitej. Sygnal jest wiec
    # tam, gdzie faktycznie gryzie, a 9 przebiegow echo2depth kosztuje mniej niz
    # jeden przebieg PA na pelnym modelu.
    Condition("EPA", "cardinal", model=MODEL_ECHO, geometry="patched",
              isolates="echo2depth patched, 4 kierunki", group="geometria_echo"),
    Condition("EPB", "all", model=MODEL_ECHO, geometry="patched",
              isolates="echo2depth patched, 36 orientacji", group="geometria_echo"),
    Condition("EPD", "random_4", model=MODEL_ECHO, geometry="patched",
              isolates="echo2depth patched, sama gestosc przy licznosci rownej EPA",
              group="geometria_echo"),
)

CONDITIONS_BY_ID = {c.id: c for c in CONDITIONS}

# Grupy uruchamiane razem, W KOLEJNOSCI URUCHAMIANIA, nie alfabetycznie.
# `bramka` idzie PIERWSZA i to nie jest kwestia gustu: jesli calkowity wklad
# echa okaze sie rzedu 0,005 RMSE, cala macierz na pelnym modelu jest niezdolna
# wykryc efekt gestosci i ciezar dowodu trzeba przeniesc na `echo` i Model 2.
# Lepiej wiedziec to po godzinie niz po dwudziestu czterech.
GROUPS = ("bramka", "echo", "glowne", "krzywa", "krzywa_staly", "geometria_echo", "geometria")


# =====================================================================
# PLAN FAKTYCZNY wobec PRZESTRZENI PROJEKTOWEJ
# =====================================================================
#
# `CONDITIONS x SEEDS` to 66 przebiegow -- ale to jest PRZESTRZEN PROJEKTOWA,
# nie plan. Po drodze zapadly decyzje, ktore czesc z nich odwolaly albo odsunely.
# Bez zapisania ich tutaj `exp_ctl.py status` pokazuje "14/66", co czyta sie jako
# "zrobione 21 %", podczas gdy wobec faktycznego planu jest to 45 %.
#
# Rozroznienie, ktore trzeba utrzymac: sa DWA rodzaje niezaplanowania.
#   * odwolane regula  -- decyzja zapadla na podstawie pomiaru, nie wroci bez
#                         nowego pomiaru (glowne, ziarna 1-2);
#   * odsuniete        -- warunek jest zdefiniowany i gotowy, tylko ma nizszy
#                         priorytet; wraca, jesli wyniki tego zazadaja.

# =====================================================================
# HISTORIA DECYZJI O LICZBIE ZIAREN -- CZYTAC PRZED `PLANNED_SEEDS`
# =====================================================================
#
# Ten blok jest zapisem CHRONOLOGICZNYM, nie stanem koncowym. Praca ma
# pokazywac, ze regula byla zapisana PRZED pomiarem, zadzialala, i ze zmiana
# nastapila z podanego powodu -- to jest mocniejsze niz udawanie, ze od poczatku
# planowano 3 ziarna. Dlatego pierwotna regula ZOSTAJE tutaj razem ze swoim
# uzasadnieniem i wynikiem, ktory dala.
#
# --- DECYZJA 1 (2026-08-11 §2, PRZEDREJESTROWANA) -------------------------
# Regula zapisana PRZED zmierzeniem czegokolwiek na grupie `glowne`:
#     bound = f_ang x c_full;  jesli bound < 0,015 -> 1 ziarno, nie 3.
# Pomiar bramki dal c_full = 0,02228 (calkowity wklad echa do pelnego modelu).
# Udzial gestosci katowej w efekcie lacznym zmierzony na `echo2depth` wynosil
# f_ang = 0,2373, wiec:
#     bound = 0,2373 x 0,02228 = 0,00529   ->  0,72-2,28x podlogi szumu
# czyli PONIZEJ progu. Regula zastosowana bez negocjacji: A/B/D dostaly po
# 1 ziarnie, z jawnym zastrzezeniem, ze nie beda mialy orzeczenia o istotnosci.
# TO BYLO PROCEDURALNIE POPRAWNE i tak ma zostac opisane.
#
# --- CO SIE OKAZALO (2026-08-13 §2, POMIAR) -------------------------------
# Zmierzone D - A w pelnym modelu = 0,01831, a nie przewidziane 0,00529.
# Heurystyka ZANIZYLA efekt 3,46x. Zawiodla nie regula, tylko jej przeslanka:
# zalozenie, ze udzial gestosci `f_ang` przenosi sie MULTIPLIKATYWNIE miedzy
# architekturami (echo2depth -> pelny model). Zalozenie okazalo sie falszywe,
# przy czym w strone KONSERWATYWNA -- prawdziwy efekt jest 2,5-8x ponad podloga
# szumu (0,0023-0,0073), a nie 0,72-2,28x, jak przewidywalo proxy.
#
# --- DECYZJA 2 (2026-08-15 §3.2, POST HOC) --------------------------------
# Autor zatwierdza dolozenie ziaren 1-2 dla A/B/D. Jest to decyzja POST HOC:
# zapadla PO zobaczeniu danych i wylacznie dlatego, ze proxy sie pomylilo.
# Nie unieważnia decyzji 1 -- unieważnia tylko jej przeslanke. Bez tego zapisu
# tabela z 3 ziarnami wygladalaby na plan od poczatku, czym nie byla.
# Przy okazji EPA/EPB/EPD tez ida na 3 ziarna (2026-08-15 §3.1): ich zadanie
# ("domkniecie maski scislej") zostalo wykonane przy 1 ziarnie, ale wynik
# okazal sie kontrintuicyjny i zgodny z niezaleznym pomiarem fizycznym, wiec
# zaslugue na przedzial ufnosci.
SEED_DECISIONS: tuple[dict, ...] = (
    {"data": "2026-08-11", "sekcja": "§2", "warunki": ("A", "B", "D"),
     "ziaren": 1, "typ": "przedrejestrowana",
     "regula": "bound = f_ang x c_full < 0,015 -> 1 ziarno",
     "liczby": {"c_full": 0.02228, "f_ang": 0.2373, "bound": 0.00529,
                "prog": 0.015, "podloga_szumu": (0.0023, 0.0073)},
     "powod": "przewidywany efekt ponizej progu -- 3 ziarna kupowalyby precyzje "
              "dla liczby, ktora i tak nie odpowie na pytanie pracy"},
    {"data": "2026-08-15", "sekcja": "§3.2", "warunki": ("A", "B", "D"),
     "ziaren": 3, "typ": "post hoc",
     "regula": "brak -- decyzja autora po zobaczeniu pomiaru",
     "liczby": {"D_minus_A_zmierzone": 0.01831, "bound_przewidziany": 0.00529,
                "zanizenie_x": 3.46, "krotnosc_podlogi_szumu": (2.5, 8.0)},
     "powod": "proxy `f_ang` zanizylo efekt 3,46x (ustalenie empiryczne z "
              "2026-08-13 §2); przeslanka o multiplikatywnosci f_ang miedzy "
              "architekturami okazala sie falszywa"},
    {"data": "2026-08-15", "sekcja": "§3.1", "warunki": ("EPA", "EPB", "EPD"),
     "ziaren": 3, "typ": "post hoc",
     "regula": "brak -- decyzja autora po zobaczeniu pomiaru",
     "liczby": {"patched_minus_main": (0.01462, 0.01014, 0.01507)},
     "powod": "wynik kontrintuicyjny (`patched` GORSZY mimo +46 % energii poglosu) "
              "i zgodny z niezaleznym pomiarem fizycznym (-17,5 % kontrastu "
              "katowego) -- potwierdzone przewidywanie zasluguje na przedzial ufnosci"},
)

# Ile ziaren faktycznie zaplanowano dla danego warunku (domyslnie: wszystkie SEEDS).
# STAN PO DECYZJI 2. Chronologia jest w `SEED_DECISIONS` powyzej -- nie kasowac
# jej przy kolejnych zmianach, tylko dopisywac.
PLANNED_SEEDS: dict[str, int] = {
    # A/B/D: 1 -> 3 (decyzja 2, post hoc). Pierwotne "1" i jego uzasadnienie
    # sa w SEED_DECISIONS[0]; nie usuwac.
    "A": 3, "B": 3, "D": 3,
    # Bramka pelnego modelu -- nie jest pozycja w tabeli pracy, a jej przedzial
    # ufnosci pochodzi z bootstrapu po lokalizacjach, nie z rozrzutu po ziarnach.
    # BEZ ZMIAN: decyzja 2 dotyczy warunkow tabeli wynikow, nie bramki.
    "SE": 1,
    # EPA/EPB/EPD: 1 -> 3 (decyzja 2026-08-15 §3.1). Pierwotny powod ("wystarczy
    # 1 model `patched` do domkniecia maski scislej") byl trafny wobec pytania,
    # ktore wtedy zadawano -- pytanie sie zmienilo, nie odpowiedz.
    "EPA": 3, "EPB": 3, "EPD": 3,
}

# Powod, dla ktorego warunek ma MNIEJ ziaren niz `SEEDS`. Pokazuje OBIE decyzje
# i ich kolejnosc, a nie tylko stan koncowy -- inaczej z kodu nie da sie
# odtworzyc, ze regula w ogole byla.
SEED_LIMIT_REASON: dict[str, str] = {
    "A": "2026-08-11 §2: 1 ziarno (bound = 0,00529 < prog 0,015) -> "
         "2026-08-15 §3.2: 3 ziarna, POST HOC (proxy f_ang zanizylo efekt 3,46x; "
         "zmierzone D-A = 0,01831, czyli 2,5-8x podlogi szumu)",
    "B": "2026-08-11 §2: 1 ziarno (bound = 0,00529 < prog 0,015) -> "
         "2026-08-15 §3.2: 3 ziarna, POST HOC (proxy f_ang zanizylo efekt 3,46x)",
    "D": "2026-08-11 §2: 1 ziarno (bound = 0,00529 < prog 0,015) -> "
         "2026-08-15 §3.2: 3 ziarna, POST HOC (proxy f_ang zanizylo efekt 3,46x)",
    "SE": "bramka, nie pozycja w tabeli pracy (bez zmian od 2026-08-11)",
    "EPA": "2026-08-11 §5: 1 ziarno (wystarczy 1 model `patched` do maski scislej) -> "
           "2026-08-15 §3.1: 3 ziarna (wynik kontrintuicyjny, potwierdza pomiar fizyczny)",
    "EPB": "2026-08-11 §5: 1 ziarno (wystarczy 1 model `patched` do maski scislej) -> "
           "2026-08-15 §3.1: 3 ziarna (wynik kontrintuicyjny, potwierdza pomiar fizyczny)",
    "EPD": "2026-08-11 §5: 1 ziarno (wystarczy 1 model `patched` do maski scislej) -> "
           "2026-08-15 §3.1: 3 ziarna (wynik kontrintuicyjny, potwierdza pomiar fizyczny)",
}

# Grupy odsuniete w calosci -- z powodem. NIE sa skasowane: warunki sa
# zdefiniowane i gotowe do uruchomienia, jesli wyniki tego zazadaja.
DEFERRED_GROUPS: dict[str, str] = {
    "krzywa": "rosnie po gestosci I po rozmiarze zbioru naraz (2026-08-10 §5.1) "
              "-- zastapiona przez `krzywa_staly` o stalej licznosci; "
              "przy okazji najdrozsza grupa macierzy (10,4 h, 70,8 GB)",
    # POPRAWIONE 2026-08-15. Poprzednia wersja konczyla sie zdaniem "po degradacji
    # pelny model i tak nie rozdziela efektu" -- to twierdzenie jest FALSZYWE:
    # pomiar z 2026-08-13 §2 dal D - A = 0,01831, czyli 2,5-8x podlogi szumu,
    # wiec pelny model efekt gestosci JEDNAK rozdziela. Grupa zostaje odsunieta,
    # ale z powodu kosztu i ostrosci pytania, a NIE z powodu niezdolnosci modelu.
    "geometria": "wada geometrii jest AKUSTYCZNA (+46,3 % energii poznej wobec "
                 "+1,3 % calkowitej), wiec `geometria_echo` bada to ostrzej i ~20x "
                 "taniej; PA/PB/PD kosztuja ~2,6 h i odpowiadaja na to samo pytanie "
                 "slabiej -- odsuniete dla kosztu, nie dla niezdolnosci pelnego modelu",
}


def plan_status(cond_id: str, seed: int) -> tuple[bool, str]:
    """Czy ten przebieg jest w PLANIE, a jesli nie -- dlaczego.

    Zwraca `(zaplanowany, powod)`. Powod jest pusty dla zaplanowanych.
    """
    cond = CONDITIONS_BY_ID.get(cond_id)
    if cond is None:
        return False, "nieznany warunek"
    if cond.group in DEFERRED_GROUPS:
        return False, DEFERRED_GROUPS[cond.group]
    limit = PLANNED_SEEDS.get(cond_id, len(SEEDS))
    if seed >= limit:
        return False, SEED_LIMIT_REASON.get(cond_id, f"zaplanowano {limit} ziarno/-a")
    return True, ""


def plan_summary() -> dict:
    """Ile przebiegow jest w planie, a ile w przestrzeni projektowej."""
    total = planned = 0
    powody: dict[str, int] = {}
    for c in CONDITIONS:
        for s in SEEDS:
            total += 1
            ok, why = plan_status(c.id, s)
            if ok:
                planned += 1
            else:
                powody[why] = powody.get(why, 0) + 1
    return {"przestrzen_projektowa": total, "w_planie": planned,
            "niezaplanowane": total - planned, "powody": powody}


@dataclass
class RunSpec:
    """Pojedynczy przebieg: warunek + ziarno. To jest jednostka kolejkowania."""

    condition: str
    seed: int
    total_steps: int = TOTAL_STEPS
    batch_size: int = BATCH_SIZE
    num_workers: int = 8
    amp: bool = True
    lr: float = 1e-4
    weight_decay: float = 5e-4
    beta1: float = 0.9
    optimizer: str = "adam"
    validation_freq: int = 1000
    display_freq: int = 100
    edge_threshold_m: float = 0.10
    extra: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.condition}_seed{self.seed}"

    def run_dir(self) -> Path:
        return paths.RUNS_DIR / self.run_id


def default_matrix(groups=("glowne", "echo", "krzywa"), seeds=SEEDS, **kw) -> list[RunSpec]:
    out = []
    for c in CONDITIONS:
        if c.group not in groups:
            continue
        for s in seeds:
            out.append(RunSpec(condition=c.id, seed=s, **kw))
    return out


def matrix_summary(groups=GROUPS, seeds=SEEDS, steps=TOTAL_STEPS,
                   batch_size=BATCH_SIZE, s_per_step: float | None = None) -> dict:
    """Podsumowanie macierzy: licznosci, rownowaznik epok, szacowany czas."""
    rows = []
    for c in CONDITIONS:
        if c.group not in groups:
            continue
        splits = load_splits(variant=c.geometry)
        n = c.n_train_samples(splits)
        rows.append({
            "id": c.id,
            "grupa": c.group,
            "angle_subset": c.angle_subset,
            "katow_na_lokalizacje": angles_mod.angles_per_location(c.angle_subset),
            "model": c.model,
            "geometria": c.geometry,
            "probek_train": n,
            "probek_val": expected_n_samples(splits, "val", c.angle_subset),
            "probek_test": expected_n_samples(splits, "test", c.angle_subset),
            "rownowaznik_epok": round(steps * batch_size / n, 1),
            "izoluje": c.isolates,
            "przebiegow": len(seeds),
        })
    total_runs = sum(r["przebiegow"] for r in rows)
    out = {
        "krokow_na_przebieg": steps,
        "batch_size": batch_size,
        "ziarna": list(seeds),
        "warunkow": len(rows),
        "przebiegow_razem": total_runs,
        "warunki": rows,
    }
    if s_per_step:
        h = steps * s_per_step / 3600
        out["godzin_na_przebieg"] = round(h, 2)
        out["godzin_cala_macierz"] = round(h * total_runs, 1)
        out["dni_cala_macierz"] = round(h * total_runs / 24, 2)
    return out


def disk_budget(param_counts: dict[str, int]) -> dict:
    """Ile miejsca zajmuje przebieg -- OSOBNO same wagi i wagi ze stanem Adama.

    Raport 2026-08-05 podawal 1,3 GB na przebieg, liczac 317 M x 4 B. To sa
    SAME `state_dict`y. `train_condition.save_checkpoint()` zapisuje jednak
    rowniez `optimizer.state_dict()`, a Adam trzyma DWA momenty na kazdy
    parametr (`exp_avg`, `exp_avg_sq`) -- czyli checkpoint wznowieniowy jest
    ~3x wiekszy od samych wag. Skoro `train_condition.py` ma realnie wznawiac
    po SIGTERM, to jest liczba, ktora obowiazuje.

    Na przebieg zapisywane sa TRZY rzeczy (od 2026-08-11):
      * `best_<net>.pth`   -- wagi najlepsze wg `val@36` (1x parametry) -- PODSTAWOWE
      * `best4_<net>.pth`  -- wagi najlepsze wg `val@4`  (1x parametry) -- kolumna odpornosci
      * `checkpoint.pt`    -- wagi + 2 momenty Adama + scaler (3x parametry)

    Drugi komplet wag to skutek decyzji o walidacji na pelnych 36 katach
    (`train_condition.VAL_ANGLE_SUBSET`): `val@4` liczy sie z tych samych
    predykcji, ale MOZE wskazac inny krok, wiec jego wagi trzeba zapisac osobno.
    Gdy oba kryteria wskazuja ten sam krok, drugi plik nie powstaje
    (`status.json: best_step_same`), wiec ponizsze 5x to gorne oszacowanie.
    """
    b = 4  # float32
    out = {}
    for model, n in param_counts.items():
        weights_gb = n * b / 1024 ** 3
        out[model] = {
            "parameters": n,
            "best_weights_GB": round(weights_gb, 3),
            "best_weights_val4_GB": round(weights_gb, 3),
            "checkpoint_weights_only_GB": round(weights_gb, 3),
            "checkpoint_with_adam_GB": round(3 * weights_gb, 3),
            # 1x best@36 + 1x best@4 + 3x checkpoint = 5x parametry
            "per_run_total_GB": round(5 * weights_gb, 3),
        }
    return out


def matrix_disk_and_time(param_counts: dict[str, int], s_per_step: dict[str, float],
                         groups=GROUPS, seeds=SEEDS, steps=TOTAL_STEPS) -> dict:
    """Laczny czas i dysk macierzy, rozbite na grupy i typy modelu."""
    per_model = disk_budget(param_counts)
    rows = []
    for c in CONDITIONS:
        if c.group not in groups:
            continue
        h = steps * s_per_step[c.model] / 3600
        rows.append({
            "id": c.id, "grupa": c.group, "model": c.model,
            "przebiegow": len(seeds),
            "godzin_na_przebieg": round(h, 3),
            "godzin_razem": round(h * len(seeds), 2),
            "GB_razem_z_adamem": round(per_model[c.model]["per_run_total_GB"] * len(seeds), 1),
            "GB_razem_same_wagi": round(per_model[c.model]["best_weights_GB"] * len(seeds), 1),
        })
    by_group: dict[str, dict] = {}
    for r in rows:
        g = by_group.setdefault(r["grupa"], {"przebiegow": 0, "godzin": 0.0, "GB": 0.0})
        g["przebiegow"] += r["przebiegow"]
        g["godzin"] += r["godzin_razem"]
        g["GB"] += r["GB_razem_z_adamem"]
    for g in by_group.values():
        g["godzin"] = round(g["godzin"], 2)
        g["GB"] = round(g["GB"], 1)
    return {
        "na_model": per_model,
        "s_na_krok": s_per_step,
        "warunki": rows,
        "grupy": by_group,
        "razem": {
            "przebiegow": sum(r["przebiegow"] for r in rows),
            "godzin": round(sum(r["godzin_razem"] for r in rows), 1),
            "dni": round(sum(r["godzin_razem"] for r in rows) / 24, 2),
            "GB_z_adamem": round(sum(r["GB_razem_z_adamem"] for r in rows), 1),
            "GB_same_wagi": round(sum(r["GB_razem_same_wagi"] for r in rows), 1),
        },
    }


def config_path() -> Path:
    return paths.ML_OUTPUTS / "experiments.json"


def dump_config(path: Path | None = None, s_per_step: float | None = None,
                param_counts: dict[str, int] | None = None,
                s_per_step_by_model: dict[str, float] | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Budzet liczy sie ZAWSZE, ze stalych zmierzonych. Wczesniej byl opcjonalny
    # i `exp_ctl.py plan` cicho go kasowal z pliku przy kazdym wywolaniu.
    payload = {
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "seeds": list(SEEDS),
        "zasada": "stala liczba krokow gradientu we wszystkich warunkach, NIE stala liczba epok",
        "kolejnosc_grup": list(GROUPS),
        "conditions": [asdict(c) for c in CONDITIONS],
        "summary": matrix_summary(s_per_step=s_per_step),
        "budzet": matrix_disk_and_time(param_counts or PARAM_COUNTS,
                                       s_per_step_by_model or SEC_PER_STEP),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
