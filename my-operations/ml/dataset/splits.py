"""Podzial train / val / test -- PO LOKALIZACJACH, nie po probkach.

DLACZEGO po lokalizacjach. Jedna lokalizacja daje 36 probek roznionych wylacznie
orientacja agenta: pozycja jest ta sama, wiec geometria widziana z niej jest w
duzej mierze ta sama scena z innego kata. Losowy podzial po probkach wpuscilby
te sama lokalizacje jednoczesnie do walidacji i do testu (w innych orientacjach),
a to jest wyciek: model oceniany bylby na pozycji, ktorej sasiedztwo widzial przy
wyborze checkpointu. Przy 36 orientacjach ten wyciek jest 9x grozniejszy niz w
oryginalnym ukladzie 4-kierunkowym Gao, wiec nie da sie go zignorowac.

DLACZEGO plik na dysku. Splity musza byc identyczne miedzy warunkiem A i B, i
miedzy ziarnami -- inaczej roznica RMSE mieszalaby efekt gestosci katowej z
efektem innego zbioru testowego. Podzial jest wiec liczony RAZ, zapisany do JSON
i pozniej tylko wczytywany; ziarno sieci (`--seed`) nie ma na niego wplywu.

DLACZEGO ten sam plik dla obu wariantow geometrii. Sprawdzone: `locations/loc_id`
i `locations/position` sa bit-identyczne miedzy `main` a `patched` dla wszystkich
10 wspolnych scen (warianty roznia sie wylacznie siatka, nie zbiorem lokalizacji),
wiec jeden podzial obsluguje oba i porownanie main-vs-patched idzie na tych samych
lokalizacjach.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .. import paths

# Podzial scen dokladnie jak w `options/base_options.py` Paridy dla Repliki.
# Przepisany tu, a nie zaimportowany, bo import wymagalby uruchomienia
# argparse'a Paridy -- ale MUSI sie zgadzac co do znaku, inaczej wyniki nie sa
# porownywalne z opublikowanymi.
TRAIN_SCENES = [
    "apartment_0", "apartment_1",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2", "frl_apartment_3",
    "frl_apartment_4", "office_0", "office_1", "office_2", "office_3",
    "hotel_0", "room_0", "room_1", "room_2",
]
# W base_options.py `scenes['val']` NIE JEST NIGDY USTAWIANE, mimo ze train.py go
# uzywa (`opt.mode = 'val'` -> KeyError). Gao dzieli te 3 sceny held-out na pol
# miedzy walidacje i test; tu robimy to samo, ale na poziomie lokalizacji.
HELDOUT_SCENES = ["apartment_2", "frl_apartment_5", "office_4"]

ALL_SCENES = TRAIN_SCENES + HELDOUT_SCENES

# Ziarno podzialu lokalizacji. Celowo osobna stala, nie `--seed` przebiegu:
# podzial ma byc niezmiennikiem calej macierzy eksperymentow.
SPLIT_SEED = 20260805

SPLIT_VERSION = "1.0.0"


@dataclass(frozen=True)
class Splits:
    """scene -> posortowana lista location_id, osobno dla kazdego podzbioru."""

    train: dict[str, list[int]]
    val: dict[str, list[int]]
    test: dict[str, list[int]]
    meta: dict

    def scenes(self, mode: str) -> list[str]:
        return sorted(getattr(self, mode).keys())

    def locations(self, mode: str) -> dict[str, list[int]]:
        return getattr(self, mode)

    def n_locations(self, mode: str) -> int:
        return sum(len(v) for v in getattr(self, mode).values())


def _read_location_ids(scene: str, variant: str) -> list[int]:
    p = paths.scene_h5(scene, variant)
    if not p.exists():
        raise FileNotFoundError(f"brak pliku HDF5 dla sceny {scene!r} (wariant {variant!r}): {p}")
    with h5py.File(p, "r") as f:
        if not bool(f.attrs.get("complete", False)):
            raise RuntimeError(f"{p} ma complete=False -- podzial na niekompletnym zbiorze nie ma sensu")
        return sorted(int(x) for x in f["locations/loc_id"][:])


def build_splits(variant: str = "main") -> Splits:
    """Liczy podzial od zera na podstawie zawartosci plikow HDF5."""
    train: dict[str, list[int]] = {}
    for scene in TRAIN_SCENES:
        train[scene] = _read_location_ids(scene, variant)

    val: dict[str, list[int]] = {}
    test: dict[str, list[int]] = {}
    # Kazda scena held-out dzielona osobno na pol. DLACZEGO osobno, a nie na
    # wspolnej puli: inaczej losowanie moglo by dac walidacje zdominowana przez
    # jedna scene, a wtedy wybor checkpointu premiowalby model dobry akurat na
    # niej. Dzielac w obrebie sceny, obie polowki maja ten sam sklad scen.
    for scene in HELDOUT_SCENES:
        loc_ids = _read_location_ids(scene, variant)
        # Ziarno zalezne od nazwy sceny -> permutacja sceny A nie zmienia sie,
        # gdy dolozy sie/usunie scene B. Podzial jest wiec stabilny lokalnie.
        rng = np.random.default_rng(_scene_seed(scene))
        perm = rng.permutation(len(loc_ids))
        half = len(loc_ids) // 2
        val_idx = sorted(perm[:half].tolist())
        test_idx = sorted(perm[half:].tolist())
        val[scene] = [loc_ids[i] for i in val_idx]
        test[scene] = [loc_ids[i] for i in test_idx]

    meta = {
        "split_version": SPLIT_VERSION,
        "split_seed": SPLIT_SEED,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "built_from_variant": variant,
        "train_scenes": TRAIN_SCENES,
        "heldout_scenes": HELDOUT_SCENES,
        "rule": "sceny train wg base_options.py Paridy; 3 sceny held-out dzielone "
                "po lokalizacjach 50/50 na val i test (nieparzysta liczba -> test dostaje +1)",
        "n_locations": {
            "train": sum(len(v) for v in train.values()),
            "val": sum(len(v) for v in val.values()),
            "test": sum(len(v) for v in test.values()),
        },
    }
    meta["location_fingerprint"] = _fingerprint(train, val, test)
    return Splits(train=train, val=val, test=test, meta=meta)


def _scene_seed(scene: str) -> int:
    h = hashlib.sha256(f"{SPLIT_SEED}:{scene}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _fingerprint(*dicts: dict[str, list[int]]) -> str:
    """Skrot calego podzialu -- pozwala pozniej udowodnic, ze dwa przebiegi
    naprawde uzyly tego samego zbioru testowego."""
    h = hashlib.sha256()
    for d in dicts:
        for scene in sorted(d):
            h.update(scene.encode())
            h.update(np.asarray(d[scene], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def split_path(name: str = "replica_locations") -> Path:
    return paths.SPLITS_DIR / f"{name}.json"


def save_splits(splits: Splits, path: Path | None = None) -> Path:
    path = path or split_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": splits.meta,
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_splits(path: Path | None = None, *, variant: str = "main", autobuild: bool = True) -> Splits:
    """Wczytuje podzial; przy braku pliku liczy go i zapisuje (raz)."""
    path = path or split_path()
    if not path.exists():
        if not autobuild:
            raise FileNotFoundError(f"brak pliku podzialu: {path}")
        splits = build_splits(variant)
        save_splits(splits, path)
        return splits
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Splits(
        train={k: [int(i) for i in v] for k, v in payload["train"].items()},
        val={k: [int(i) for i in v] for k, v in payload["val"].items()},
        test={k: [int(i) for i in v] for k, v in payload["test"].items()},
        meta=payload["meta"],
    )
