"""Dataset czytajacy echa/RGB/glebie prosto z HDF5 -- zamiennik `AudioVisualDataset`.

Oryginal (`beyond-image-to-depth/data_loader/audio_visual_dataset.py`) zostaje
NIETKNIETY; ta klasa stoi obok, zeby dalo sie oba warianty porownac i zeby
kazda roznica wyniku dala sie przypisac do zmiany danych, a nie do zmiany kodu
wspolnego.

Trzy rzeczy robi inaczej niz oryginal -- kazda z powodu, nie dla wygody:

1. NIE liczy STFT w `__getitem__`. Spektrogramy sa juz policzone w zbiorze,
   dokladnie tym samym potokiem: `librosa.stft(n_fft=512, win_length=64)`, przy
   czym `hop_length` librosy domyslnie wynosi `win_length // 4 = 16` -- czyli
   generator (`echo_core/spectrogram.py`, hop podany jawnie jako 16) produkuje
   BIT-ZGODNE wejscie z `generate_spectrogram(..., winl=64)` Paridy. Ksztalt
   (2, 257, 166) zgadza sie z `opt.audio_shape` dla Repliki. Liczenie tego
   ponownie w gorącej petli byloby czystym marnotrawstwem CPU.

2. NIE laduje calego zbioru do RAM. Stary pickle mial 913 MB i miescil sie w
   pamieci; ten zbior ma ~26 GiB w dwoch wariantach (15 + 11 GiB), wiec czytany
   jest leniwie, po jednej probce, z chunkow HDF5 1:1 odpowiadajacych probce.

3. Indeksuje po (scene, location_id, angle_deg), a nie po plaskiej liscie
   kluczy pickle'a -- bo filtr gestosci katowej i podzial po lokalizacjach
   wymagaja obu wspolrzednych osobno.

PULAPKA, KTORA TRZEBA ZNAC: h5py nie przezywa `fork()`. Otwarty uchwyt
odziedziczony przez workera DataLoadera daje ciche uszkodzenie danych albo
zawis, BEZ komunikatu bledu. Dlatego uchwyty otwierane sa leniwie i dodatkowo
pilnowany jest PID (patrz `_handles()`): po forku slownik uchwytow jest
PORZUCANY, nie zamykany -- zamkniecie odziedziczonego uchwytu dotyka wspolnego
stanu biblioteki HDF5 i jest samo w sobie niebezpieczne.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.utils.data as data

from . import angles as angles_mod
from . import paths
from .splits import Splits, load_splits

# Statystyki ImageNet -- te same, ktorych uzywa `AudioVisualDataset`, bo
# `MaterialPropertyNet` startuje z wag ResNetu pretrenowanego na ImageNecie.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Stala z `base_options.py` dla Repliki. Model mnozy sigmoid przez nia, wiec
# glebie powyzej tej wartosci sa z definicji nieosiagalne dla predykcji.
MAX_DEPTH_REPLICA = 14.104

EXPECTED_SPEC_SHAPE = (2, 257, 166)
EXPECTED_IMAGE_SHAPE = (128, 128)


@dataclass(frozen=True)
class DatasetConfig:
    """Wszystko, co decyduje o TOZSAMOSCI zbioru probek.

    Rozdzielone od parametrow treningu celowo: dwa przebiegi z ta sama
    `DatasetConfig` widza dokladnie te same probki, niezaleznie od ziarna sieci.
    """

    variant: str = "main"            # main | patched
    mode: str = "train"              # train | val | test
    angle_subset: str = "all"
    angle_seed: int = 0              # uzywane wylacznie przez random_K
    image_transform: bool = True
    augment: bool | None = None      # None -> augmentacja tylko w train (jak u Paridy)
    return_meta: bool = True

    def effective_augment(self) -> bool:
        if self.augment is not None:
            return self.augment
        return self.mode == "train"


class EchoH5Dataset(data.Dataset):
    def __init__(self, cfg: DatasetConfig, splits: Splits | None = None):
        if cfg.variant not in paths.DATASET_DIRS:
            raise ValueError(f"nieznany wariant {cfg.variant!r}, oczekiwano {list(paths.DATASET_DIRS)}")
        if cfg.mode not in ("train", "val", "test"):
            raise ValueError(f"nieznany tryb {cfg.mode!r}")
        angles_mod.parse(cfg.angle_subset)  # walidacja wczesnie, nie po godzinie treningu

        self.cfg = cfg
        self.splits = splits if splits is not None else load_splits(variant=cfg.variant)
        self.max_depth = MAX_DEPTH_REPLICA

        self._build_index()

        # Uchwyty HDF5 -- ZAWSZE None po konstrukcji. Gdyby cokolwiek otworzylo
        # plik tutaj, workery DataLoadera odziedziczylyby ten uchwyt przez fork.
        self._files: dict[str, h5py.File] | None = None
        self._pid: int | None = None

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    # ------------------------------------------------------------------ indeks

    def _build_index(self) -> None:
        """Buduje (scene_idx, row) dla kazdej probki, czytajac WYLACZNIE male,
        nieskompresowane tablice metadanych (`location_id`, `angle_deg`).

        Zaden bajt echa/RGB/glebi nie jest tu dotykany, wiec zbudowanie indeksu
        dla calego zbioru kosztuje ulamek sekundy i mozna to robic w petli po
        wszystkich wariantach filtra bez obciazania dysku.
        """
        cfg = self.cfg
        wanted: dict[str, list[int]] = self.splits.locations(cfg.mode)

        self.scenes: list[str] = sorted(wanted.keys())
        self.scene_paths: list[str] = [str(paths.scene_h5(s, cfg.variant)) for s in self.scenes]

        scene_idx_all: list[np.ndarray] = []
        row_all: list[np.ndarray] = []
        loc_all: list[np.ndarray] = []
        ang_all: list[np.ndarray] = []
        self.per_scene_counts: dict[str, int] = {}
        self.per_scene_locations: dict[str, int] = {}

        for si, scene in enumerate(self.scenes):
            loc_ids = wanted[scene]
            if not loc_ids:
                self.per_scene_counts[scene] = 0
                self.per_scene_locations[scene] = 0
                continue

            with h5py.File(self.scene_paths[si], "r") as f:
                sample_loc = f["location_id"][:].astype(np.int64)
                sample_ang = f["angle_deg"][:].astype(np.int64)
                written = f["written"][:].astype(bool)

            # Mapa (location_id, angle) -> wiersz. Zbudowana raz na scene;
            # kodowanie w jeden int64 pozwala uzyc szybkiego `np.isin`
            # zamiast petli po ~7600 probkach.
            key = sample_loc * 1000 + sample_ang

            rows_for_scene: list[np.ndarray] = []
            for loc_id in loc_ids:
                sel = angles_mod.select_angles(
                    cfg.angle_subset, scene=scene, location_id=int(loc_id), seed=cfg.angle_seed
                )
                rows_for_scene.append(loc_id * 1000 + sel)
            wanted_keys = np.concatenate(rows_for_scene)

            order = np.argsort(key, kind="stable")
            key_sorted = key[order]
            pos = np.searchsorted(key_sorted, wanted_keys)
            bad = (pos >= key_sorted.size) | (key_sorted[np.minimum(pos, key_sorted.size - 1)] != wanted_keys)
            if bad.any():
                missing = wanted_keys[bad]
                raise RuntimeError(
                    f"{scene}: brak {missing.size} probek (loc*1000+kat), np. {missing[:5].tolist()} "
                    f"-- plik HDF5 nie zawiera wszystkich orientacji dla wybranych lokalizacji"
                )
            rows = order[pos]

            # `written` to flaga generatora: probka faktycznie zapisana na dysk.
            # Wszystkie pliki maja complete=True, ale sprawdzamy jawnie, bo cicho
            # wpuszczona niezapisana probka bylaby zerowym echem i zerowa glebia
            # -- czyli nauczylaby siec bledu, ktorego nikt by nie zauwazyl.
            if not written[rows].all():
                bad = int((~written[rows]).sum())
                raise RuntimeError(f"{scene}: {bad} wybranych probek ma written=0")

            scene_idx_all.append(np.full(rows.size, si, dtype=np.int16))
            row_all.append(rows.astype(np.int32))
            # Lokalizacje i katy trzymamy w indeksie, a nie doczytujemy z HDF5
            # przy kazdej probce: to oszczedza dwa male odczyty na `__getitem__`,
            # a przy 62 640 probkach na epoke te odczyty widac w profilu.
            loc_all.append((wanted_keys // 1000).astype(np.int32))
            ang_all.append((wanted_keys % 1000).astype(np.int16))
            self.per_scene_counts[scene] = int(rows.size)
            self.per_scene_locations[scene] = len(loc_ids)

        if scene_idx_all:
            self.index_scene = np.concatenate(scene_idx_all)
            self.index_row = np.concatenate(row_all)
            self.index_loc = np.concatenate(loc_all)
            self.index_angle = np.concatenate(ang_all)
        else:
            self.index_scene = np.empty(0, dtype=np.int16)
            self.index_row = np.empty(0, dtype=np.int32)
            self.index_loc = np.empty(0, dtype=np.int32)
            self.index_angle = np.empty(0, dtype=np.int16)

    # ------------------------------------------------------------- uchwyty h5

    def _handles(self) -> dict[str, h5py.File]:
        pid = os.getpid()
        if self._files is None or self._pid != pid:
            # Po forku PORZUCAMY odziedziczone uchwyty bez zamykania -- patrz
            # komentarz w naglowku modulu.
            self._files = {}
            self._pid = pid
        return self._files

    def _file(self, scene_idx: int) -> h5py.File:
        handles = self._handles()
        path = self.scene_paths[scene_idx]
        f = handles.get(path)
        if f is None:
            f = h5py.File(path, "r")
            handles[path] = f
        return f

    def close(self) -> None:
        if self._files and self._pid == os.getpid():
            for f in self._files.values():
                f.close()
        self._files = None
        self._pid = None

    # ------------------------------------------------------------------ dostep

    def __len__(self) -> int:
        return int(self.index_scene.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        si = int(self.index_scene[index])
        row = int(self.index_row[index])
        f = self._file(si)

        # Echo: float16 na dysku (polowa miejsca przy bledzie wzglednym ~5e-4
        # na wartosciach rzedu 10) -> float32 w pamieci, bo siec liczy w float32.
        echo = np.asarray(f["echo"][row], dtype=np.float32)
        audio = torch.from_numpy(echo)

        rgb = f["rgb"][row]           # (128, 128, 4) uint8, RGBA
        depth = f["depth"][row]       # (128, 128) float32, SUROWE METRY

        img = self._prepare_image(rgb)

        # Glebia zostaje w metrach. Model mnozy swoj sigmoid przez
        # `opt.max_depth`, wiec predykcja jest w metrach i `depth_gt` musi byc
        # w tych samych jednostkach -- normalizacja tutaj rozjechalaby skale.
        depth_t = torch.from_numpy(np.asarray(depth, dtype=np.float32)).unsqueeze(0)

        out = {"img": img, "depth": depth_t, "audio": audio}

        if self.cfg.return_meta:
            # Tylko tensory liczbowe: `DataParallel` rozprasza slownik wsadu po
            # urzadzeniach i przewrocilby sie na stringu.
            out["scene_idx"] = torch.tensor(si, dtype=torch.int64)
            out["location_id"] = torch.tensor(int(self.index_loc[index]), dtype=torch.int64)
            out["angle_deg"] = torch.tensor(int(self.index_angle[index]), dtype=torch.int64)
        return out

    def _prepare_image(self, rgb: np.ndarray) -> torch.Tensor:
        """RGBA uint8 -> znormalizowany tensor (3, 128, 128).

        Kanal alfa jest odrzucany: generator zapisuje RGBA, bo taki bufor zwraca
        habitat-sim, ale alfa jest stale 255 (sprawdzone na calym zbiorze), a
        siec Paridy przyjmuje 3 kanaly (`RGBDepthNet(input_nc=3)`).
        """
        rgb3 = rgb[..., :3]

        if not self.cfg.image_transform:
            return torch.from_numpy(np.ascontiguousarray(rgb3)).permute(2, 0, 1).float()

        if self.cfg.effective_augment():
            # Sciezka wierna oryginalowi: te same trzy operacje PIL w tej samej
            # kolejnosci i tych samych zakresach co `process_image()` Paridy.
            from PIL import Image, ImageEnhance

            img = Image.fromarray(rgb3, mode="RGB")
            img = ImageEnhance.Brightness(img).enhance(random.random() * 0.6 + 0.7)
            img = ImageEnhance.Color(img).enhance(random.random() * 0.6 + 0.7)
            img = ImageEnhance.Contrast(img).enhance(random.random() * 0.6 + 0.7)
            t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
        else:
            # Bez augmentacji `transforms.ToTensor()` to dokladnie uint8/255 z
            # permutacja osi, wiec pominiecie PIL jest tu BIT-IDENTYCZNE, nie
            # przyblizone -- a oszczedza konwersje tam i z powrotem.
            t = torch.from_numpy(np.ascontiguousarray(rgb3)).permute(2, 0, 1).float().div_(255.0)

        return t.sub_(self._mean).div_(self._std)

    # ------------------------------------------------------------------- opis

    def name(self) -> str:
        return "EchoH5Dataset"

    def summary(self) -> dict:
        return {
            "name": self.name(),
            "variant": self.cfg.variant,
            "mode": self.cfg.mode,
            "angle_subset": self.cfg.angle_subset,
            "angle_seed": self.cfg.angle_seed,
            "angles_per_location": angles_mod.angles_per_location(self.cfg.angle_subset),
            "n_scenes": len(self.scenes),
            "n_locations": sum(self.per_scene_locations.values()),
            "n_samples": len(self),
            "augment": self.cfg.effective_augment(),
            "per_scene_samples": dict(self.per_scene_counts),
        }


def expected_n_samples(splits: Splits, mode: str, angle_subset: str) -> int:
    """Ile probek POWINNO wyjsc -- liczone z samego podzialu, bez otwierania HDF5.

    Sluzy do porownania z faktyczna dlugoscia datasetu w `--verify-loader`:
    zgodnosc tych dwoch liczb dowodzi, ze filtr katow nie gubi ani nie dubluje
    probek.
    """
    per_loc = angles_mod.angles_per_location(angle_subset)
    return splits.n_locations(mode) * per_loc


def build_dataloader(
    cfg: DatasetConfig,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool | None = None,
    splits: Splits | None = None,
    prefetch_factor: int | None = None,
    pin_memory: bool = True,
    drop_last: bool | None = None,
    persistent_workers: bool | None = None,
    generator: torch.Generator | None = None,
) -> tuple[data.DataLoader, EchoH5Dataset]:
    """Fabryka DataLoadera z domyslnymi ustawieniami bezpiecznymi dla h5py."""
    ds = EchoH5Dataset(cfg, splits=splits)
    if shuffle is None:
        shuffle = cfg.mode == "train"
    if drop_last is None:
        drop_last = cfg.mode == "train"
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    if generator is not None:
        kwargs["generator"] = generator
    return data.DataLoader(ds, **kwargs), ds
