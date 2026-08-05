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

from . import angles as angles_mod
from . import paths
from .echo_h5_dataset import expected_n_samples
from .splits import load_splits

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


@dataclass(frozen=True)
class Condition:
    id: str
    angle_subset: str
    model: str = MODEL_FULL
    geometry: str = "main"
    angle_seed: int = 0
    isolates: str = ""
    group: str = ""

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

    # Wariant geometrii -- osobna os, poza glowna macierza. Uruchamiac dopiero,
    # gdy os gestosci jest zamknieta; z zastrzezeniem z GENERATOR_PARAMS.md §4.5
    # (sceny zalatane sa mierzalnie bardziej wyidealizowane niz skany).
    Condition("PB", "all", geometry="patched",
              isolates="wplyw domkniecia dziur w geometrii, przy 36 orientacjach", group="geometria"),
    Condition("PA", "cardinal", geometry="patched",
              isolates="wplyw domkniecia dziur, przy 4 kierunkach", group="geometria"),
)

CONDITIONS_BY_ID = {c.id: c for c in CONDITIONS}

# Grupy uruchamiane razem, w kolejnosci priorytetu naukowego. `glowne` odpowiada
# na pytanie pracy; bez nich reszta nie ma czego doprecyzowac.
GROUPS = ("glowne", "echo", "krzywa", "geometria")


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


def config_path() -> Path:
    return paths.ML_OUTPUTS / "experiments.json"


def dump_config(path: Path | None = None, s_per_step: float | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "seeds": list(SEEDS),
        "zasada": "stala liczba krokow gradientu we wszystkich warunkach, NIE stala liczba epok",
        "conditions": [asdict(c) for c in CONDITIONS],
        "summary": matrix_summary(s_per_step=s_per_step),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
