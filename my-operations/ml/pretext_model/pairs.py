"""Zbior par (widok z orientacji i, echo z orientacji j) z tej samej lokalizacji.

ETYKIETA TO PRZESUNIECIE WZGLEDNE (j - i), NIE ORIENTACJA BEZWZGLEDNA. U Gao
przy K=4 klasy to: `^` to samo, `>` obrot w prawo o 90 stopni, `v` przeciwnie,
`<` w lewo o 90 stopni. Orientacja bezwzgledna bylaby zadaniem innym i latwiejszym
(scena ma stale kierunki), a przede wszystkim nie tym, ktore Gao raportuje.

DLACZEGO PODSIATKA K ORIENTACJI, A NIE WSZYSTKIE 36 Z DOWOLNYM K. Klasy powstaja
z roznic katow, wiec zeby kazda roznica wpadala dokladnie w jedna klase, oba katy
musza pochodzic z tej samej rownomiernej podsiatki K orientacji. Dla K = 4 sa to
kierunki kardynalne -- czyli DOKLADNIE uklad Gao. Liczba par na lokalizacje
wynosi wtedy K^2 (16 / 144 / 1 296), co zgadza sie z tabela w `__init__.py`.

DLACZEGO OPAKOWANIE NA `EchoH5Dataset`, A NIE WLASNY CZYTNIK HDF5. Cala logika,
ktora latwo zepsuc -- bezpieczenstwo wobec `fork()`, dopasowanie wierszy po
(location_id, angle_deg), przygotowanie obrazu wierne `process_image()` Paridy,
sprawdzanie flagi `written` -- jest juz napisana i zweryfikowana 8 testami
(`echo_data.py --verify-loader`). Powtorzenie jej tutaj oznaczaloby drugie
miejsce, w ktorym te same bledy moga wystapic osobno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data as data

from ..dataset import angles as angles_mod
from ..dataset.echo_h5_dataset import DatasetConfig, EchoH5Dataset
from ..dataset.splits import Splits


@dataclass(frozen=True)
class PairConfig:
    variant: str = "main"
    mode: str = "train"
    n_classes: int = 36              # K; musi rownomiernie dzielic 36
    pairs_per_location: int | None = None   # None -> wszystkie K^2
    pair_seed: int = 0               # uzywane wylacznie przy podprobkowaniu
    augment: bool | None = None

    def angle_subset(self) -> str:
        if self.n_classes == angles_mod.N_ANGLES:
            return "all"
        return f"every_{angles_mod.N_ANGLES // self.n_classes}"


class OrientationPairDataset(data.Dataset):
    def __init__(self, cfg: PairConfig, splits: Splits | None = None):
        if angles_mod.N_ANGLES % cfg.n_classes != 0:
            raise ValueError(f"K={cfg.n_classes} nie dzieli {angles_mod.N_ANGLES} bez reszty")
        self.cfg = cfg
        self.K = cfg.n_classes
        self.step_deg = 360 // self.K

        self.base = EchoH5Dataset(
            DatasetConfig(variant=cfg.variant, mode=cfg.mode,
                          angle_subset=cfg.angle_subset(), augment=cfg.augment),
            splits=splits)

        n = len(self.base)
        if n % self.K != 0:
            raise RuntimeError(f"{n} probek nie dzieli sie przez K={self.K}")
        self.n_locations = n // self.K

        # ZALOZENIE O UKLADZIE INDEKSU, SPRAWDZONE, A NIE PRZYJETE: `EchoH5Dataset`
        # buduje indeks w kolejnosci scena -> lokalizacja -> kat rosnaco, wiec
        # K kolejnych pozycji to jedna lokalizacja. Gdyby to sie kiedys zmienilo,
        # pary laczylyby rozne lokalizacje i zadanie stalo by sie bez sensu --
        # cicho, bo nadal by sie uczylo.
        loc = self.base.index_loc.reshape(self.n_locations, self.K)
        sc = self.base.index_scene.reshape(self.n_locations, self.K)
        ang = self.base.index_angle.reshape(self.n_locations, self.K)
        if not (np.all(loc == loc[:, :1]) and np.all(sc == sc[:, :1])):
            raise RuntimeError("uklad indeksu EchoH5Dataset nie grupuje K katow na lokalizacje")
        if not np.all(np.diff(ang.astype(np.int64), axis=1) > 0):
            raise RuntimeError("katy w obrebie lokalizacji nie sa rosnaco posortowane")
        self._loc_scene = sc[:, 0]
        self._loc_id = loc[:, 0]
        self._angles = ang[0].astype(np.int64)

        # Podprobkowanie (kontrola 4.4): stale `m` par na lokalizacje, losowane
        # STRATYFIKOWANIE PO LOKALIZACJI. Gdyby losowac globalnie, czesc
        # lokalizacji dostalaby wiecej par niz inne i porownanie z K=4 (ktore ma
        # dokladnie 16 na kazdej) mieszaloby liczbe par z ich rozkladem.
        self._pairs: np.ndarray | None = None
        if cfg.pairs_per_location is not None:
            m = int(cfg.pairs_per_location)
            if not 1 <= m <= self.K * self.K:
                raise ValueError(f"pairs_per_location={m} poza zakresem 1..{self.K ** 2}")
            rng = np.random.default_rng(cfg.pair_seed)
            picks = np.empty((self.n_locations, m), dtype=np.int32)
            for l in range(self.n_locations):
                picks[l] = rng.choice(self.K * self.K, size=m, replace=False)
            self._pairs = picks
        self.pairs_per_location = (cfg.pairs_per_location
                                   if cfg.pairs_per_location is not None else self.K * self.K)

    def __len__(self) -> int:
        return int(self.n_locations * self.pairs_per_location)

    def _decode(self, index: int) -> tuple[int, int, int]:
        l, r = divmod(index, self.pairs_per_location)
        code = int(self._pairs[l, r]) if self._pairs is not None else r
        i, j = divmod(code, self.K)
        return l, i, j

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        l, i, j = self._decode(index)
        base_i = l * self.K + i
        base_j = l * self.K + j

        img = self.base[base_i]["img"]        # widok z orientacji i
        audio = self.base.get_audio(base_j)   # echo z orientacji j

        cls = (j - i) % self.K
        return {
            "img": img,
            "audio": audio,
            "label": torch.tensor(cls, dtype=torch.int64),
            "shift_deg": torch.tensor(cls * self.step_deg, dtype=torch.int64),
            "scene_idx": torch.tensor(int(self._loc_scene[l]), dtype=torch.int64),
            "location_id": torch.tensor(int(self._loc_id[l]), dtype=torch.int64),
            "angle_i_deg": torch.tensor(int(self._angles[i]), dtype=torch.int64),
            "angle_j_deg": torch.tensor(int(self._angles[j]), dtype=torch.int64),
        }

    def close(self) -> None:
        self.base.close()

    @property
    def scenes(self) -> list[str]:
        return self.base.scenes

    def summary(self) -> dict:
        return {
            "K": self.K,
            "step_deg": self.step_deg,
            "variant": self.cfg.variant,
            "mode": self.cfg.mode,
            "angle_subset": self.cfg.angle_subset(),
            "n_locations": self.n_locations,
            "pairs_per_location": self.pairs_per_location,
            "n_pairs": len(self),
            "subsampled": self._pairs is not None,
            "pair_seed": self.cfg.pair_seed,
            "chance_top1": 1.0 / self.K,
            "chance_maae_deg": 90.0,
        }


def build_pair_loader(cfg: PairConfig, *, batch_size: int, num_workers: int,
                      splits: Splits | None = None, shuffle: bool | None = None,
                      drop_last: bool | None = None,
                      generator: torch.Generator | None = None):
    ds = OrientationPairDataset(cfg, splits=splits)
    if shuffle is None:
        shuffle = cfg.mode == "train"
    if drop_last is None:
        drop_last = cfg.mode == "train"
    kwargs: dict = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        pin_memory=True, drop_last=drop_last)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    if generator is not None:
        kwargs["generator"] = generator
    return data.DataLoader(ds, **kwargs), ds
