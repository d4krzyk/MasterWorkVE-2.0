#!/usr/bin/env python3
"""Generator datasetu ech 36-orientacyjnych dla Visual Echoes 2.0.

Implementuje specyfikacje z `my-operations/docs/GENERATOR_PARAMS.md` (zamrozona
2026-07-26, poprawka wysokosci `y` 2026-07-28). Dokument jest zrodlem prawdy —
kazda stala w bloku PARAMETRY nizej ma tam odwolanie do eksperymentu, ktory ja
rozstrzygnal. Nie zmieniac ich bez zmiany dokumentu.

Uruchomienie (env `habitat` aktywne):

    conda activate habitat
    python my-operations/generate_echo_dataset.py --dry-run --scene office_1
    python my-operations/generate_echo_dataset.py --scene office_1
    python my-operations/generate_echo_dataset.py --scene office_1 --resume
    python my-operations/generate_echo_dataset.py --verify office_1
    python my-operations/generate_echo_dataset.py --status

`--status`, `--verify` i `--dry-run` NIE dotykaja GPU ani nie tworza Simulatora —
mozna je odpalac w drugim terminalu w trakcie generacji.

Jedna scena = jeden proces OS = jeden dlugo zyjacy Simulator. Powyzej ~30
konstrukcji Simulatora w jednym procesie karta zawiesza sie sprzetowo (procedura
odzysku przez PCI FLR w CLAUDE.md), dlatego generator NIGDY nie konstruuje
Simulatora w petli.
"""

import argparse
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# SCIEZKI — jeden blok, zeby kolejna zmiana struktury repo wymagala edycji
# w jednym miejscu. Wszystko wyprowadzone z polozenia tego pliku, wiec skrypt
# dziala niezaleznie od katalogu roboczego (inaczej niz smoke-tests/).
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
MY_OPS = SCRIPT_PATH.parent
REPO_ROOT = MY_OPS.parent

HABITAT_SIM_PY = REPO_ROOT / "habitat-sim/src_python"
SCENE_ROOT = REPO_ROOT / "sound-spaces/data/scene_datasets/replica"
METADATA_ROOT = MY_OPS / "metadata/replica"
LOCATIONS_PKL = REPO_ROOT / "scenes_ve_metadata_locations/scene_observations_128.pkl"
MATERIAL_CONFIG = MY_OPS / "replica_material_config.json"
CHIRP_PATH = MY_OPS / "sweep_audio/3ms_sweep.wav"
SPEC_DOC = MY_OPS / "docs/GENERATOR_PARAMS.md"

OUT_ROOT = REPO_ROOT / "outputs/echoes_36deg"  # gitignored (patrz .gitignore)

# `import habitat_sim` wymaga src_python na sciezce; robimy to tutaj, zeby
# skrypt dzialal takze bez recznie ustawionego PYTHONPATH (istotne dla
# --status/--verify odpalanych ad hoc w drugim terminalu).
if str(HABITAT_SIM_PY) not in sys.path:
    sys.path.insert(0, str(HABITAT_SIM_PY))
if str(MY_OPS) not in sys.path:
    sys.path.insert(0, str(MY_OPS))


def scene_mesh(scene):
    return SCENE_ROOT / scene / "habitat/mesh_semantic.ply"


def points_txt(scene):
    return METADATA_ROOT / scene / "points.txt"


def graph_pkl(scene):
    return METADATA_ROOT / scene / "graph.pkl"


def scene_h5(scene):
    return OUT_ROOT / f"{scene}.h5"


def scene_log(scene):
    return OUT_ROOT / f"{scene}.log"


def scene_decisions(scene):
    return OUT_ROOT / f"{scene}_decisions.jsonl"


def scene_progress(scene):
    # Sidecar czytany przez --status w trakcie generacji: plik HDF5 jest wtedy
    # otwarty do zapisu przez inny proces, a maly JSON zapisywany atomowo
    # (tmp + rename) zawsze da sie przeczytac bez ryzyka.
    return OUT_ROOT / f"{scene}.progress.json"


# ---------------------------------------------------------------------------
# PARAMETRY — GENERATOR_PARAMS.md §1, §2, §3.2. NIE ZMIENIAC bez dokumentu.
# ---------------------------------------------------------------------------
SCRIPT_VERSION = "1.0.0"

INDIRECT_RAY_COUNT = 500       # §1, e2_bias_orientation / e2_rays_vs_renders
THREAD_COUNT = 1               # §1, e2_thread_budget_confirm (watki DZIELA budzet promieni)
SENSOR_HEIGHT = 1.25           # §1, listener_height + PKL_FORMAT.md (kamera i audio w jednym punkcie)
AVERAGING_DOMAIN = "mag"       # §1, e3_averaging_domain: estymata = (1/N) * suma |STFT|

N_PROBE = 8                    # §3.3 pkt 1: sonda 8 renderow przy 0 stopni, podzial 4+4
N_MIN, N_MAX = 6, 40           # §3.2 (N_MAX podniesione z 24 do 40, rewizja 2026-07-26)
TARGET_SNR = 3.5               # §3.2
SIGNAL_10DEG = 0.0644          # §3.2, mediana z noise_floor_scenes (zakres 0.0639-0.0662)

ANGLES_DEG = tuple(range(0, 360, 10))   # 36 orientacji co 10 stopni, §2
N_ANGLES = len(ANGLES_DEG)

S_PER_RENDER_SPEC = 0.2606     # §4, srednia wazona po 18 scenach
MEAN_N_SPEC = 9.83             # §3.1, srednia po 12 zmierzonych pozycjach

CAMERA_RESOLUTION = (128, 128)  # PKL_FORMAT.md
CAMERA_HFOV = 90.0              # PKL_FORMAT.md (kontrola negatywna przy 70 st.: RGB RMSE 33.6)

# Kolejnosc scen — GENERATOR_PARAMS.md §4.2. Nie alfabetyczna: najpierw scena
# walidacyjna, potem komplet held-out (zeby dataloader mogl powstawac rownolegle
# do generacji), potem po jednej treningowej z kazdej rodziny, potem reszta
# rosnaco po liczbie lokalizacji.
SCENE_ORDER = (
    "office_1",                                        # walidacyjna generatora
    "apartment_2", "frl_apartment_5", "office_4",      # held-out
    "room_0", "office_0", "hotel_0",                   # po jednej z kazdej rodziny
    "room_1", "office_2", "room_2", "office_3",
    "frl_apartment_2", "frl_apartment_4", "frl_apartment_0", "frl_apartment_1",
    "frl_apartment_3", "apartment_1", "apartment_0",
)
HELD_OUT = ("apartment_2", "frl_apartment_5", "office_4")

# Bajty na probke bez kompresji: echo float16 + rgb uint8 + depth float32.
BYTES_PER_SAMPLE = 2 * 257 * 166 * 2 + 128 * 128 * 4 + 128 * 128 * 4


# ---------------------------------------------------------------------------
# Przerwanie — SIGINT/SIGTERM konczy BIEZACA probke, zapisuje ja i zamyka plik.
# Przerwanie w srodku N-powtorzen jednej probki jest zabronione (GENERATOR_PARAMS
# §4: checkpoint na granicy PROBKI jest bezpieczny, e1_checkpoint_boundary_merge,
# Wilcoxon p=0.949 — ale granica probki, nie granica renderu).
# ---------------------------------------------------------------------------
_INTERRUPTED = False


def _install_signal_handlers(log):
    def handler(signum, _frame):
        global _INTERRUPTED
        name = signal.Signals(signum).name
        if _INTERRUPTED:
            log.warning("%s po raz drugi — przerywam natychmiast (plik moze byc niekompletny)", name)
            raise KeyboardInterrupt(name)
        _INTERRUPTED = True
        log.warning("%s — koncze biezaca probke, zapisuje i zamykam plik czysto. "
                    "Wznowienie: --scene <scena> --resume", name)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------
# Logowanie
# ---------------------------------------------------------------------------
def setup_logging(scene=None):
    import logging

    log = logging.getLogger("gen")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    if scene is not None:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(scene_log(scene), encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Lokalizacje: zbior z pkl, wspolrzedne z points.txt, wysokosc z graph.pkl
# ---------------------------------------------------------------------------
def load_scene_locations(scene):
    """-> (loc_ids: list[int], positions: dict[int, np.ndarray(3, float32)])

    Zrodla, zgodnie z GENERATOR_PARAMS.md §2 i PKL_FORMAT.md:
      * ZBIOR lokalizacji — klucze `scene_observations_128.pkl`, nie caly
        `points.txt`. Tylko ten zbior odpowiada probkowaniu z pracy Gao
        (VisualEchoes, ECCV 2020), wiec tylko on daje porownywalnosc.
      * x, z — z `points.txt`: x = a, z = -b.
      * y — z `graph.pkl` (`node["point"][1]`, pelna precyzja float32), a dla
        8 lokalizacji z calego zbioru, ktorych w grafie nie ma (room_0: 102,
        103, 111, 112, 120, 121; room_1: 45, 51) — stala sceny, bo `y` jest
        stale w obrebie sceny.

    DLACZEGO NIE `pathfinder.snap_point()`: zwraca on wysokosc powierzchni
    navmesha, ktora lezy ~0.21 m NAD `y` z grafu (mediana po 1740 lokalizacjach,
    maksimum 0.49 m; navmesh Repliki nie ma zapisanych NavMeshSettings, wiec
    recast odtwarza go z domyslna kwantyzacja). Zmierzone na office_1 przez
    porownanie piksel-po-pikselu z pkl: `y` z grafu daje RGB RMSE 0.0125 i
    99.98 % pikseli bit-identycznych, `snap_point` — RGB RMSE 50.05 i 36 %.
    Patrz GENERATOR_PARAMS.md §2 (poprawka 2026-07-28) i §5 ograniczenie 8.
    """
    import pandas as pd
    import pickle

    with open(LOCATIONS_PKL, "rb") as f:
        observations = pickle.load(f)
    if scene not in observations:
        raise KeyError(f"scena {scene!r} nie wystepuje w {LOCATIONS_PKL.name}")
    loc_ids = sorted({int(k[0]) for k in observations[scene].keys()})
    del observations  # 913 MB — nie trzymamy tego przez cala generacje

    with open(graph_pkl(scene), "rb") as f:
        graph = pickle.load(f)
    node_y = {int(n): float(d["point"][1]) for n, d in graph.nodes(data=True)}
    if not node_y:
        raise RuntimeError(f"graph.pkl sceny {scene} nie ma zadnego wezla z 'point'")
    y_scene = float(np.median(list(node_y.values())))

    points = pd.read_csv(points_txt(scene), sep="\t", header=None, names=["id", "a", "b", "c"])
    by_id = {int(r.id): (float(r.a), float(r.b)) for r in points.itertuples()}

    positions = {}
    for lid in loc_ids:
        if lid not in by_id:
            raise KeyError(f"{scene}: location_id={lid} z pkl nie ma odpowiednika w points.txt")
        a, b = by_id[lid]
        positions[lid] = np.array([a, node_y.get(lid, y_scene), -b], dtype=np.float32)
    return loc_ids, positions


# ---------------------------------------------------------------------------
# Estymator szumu — dokladnie ten sam podzial na dwie rozlaczne polowki,
# ktorego uzywa kazdy eksperyment w projekcie (diagnose_rlr_noise.py:2584-2591,
# 2688-2690) i ktory definiuje GENERATOR_PARAMS.md §3.2.
# ---------------------------------------------------------------------------
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mean_f32(specs):
    """Srednia po renderach w float32.

    GENERATOR_PARAMS.md §4.1, regula bezwzgledna: usredniac w float32, rzutowac
    na float16 dopiero na gotowym wyniku. Akumulacja w float16 kumulowalaby blad
    zaokraglenia przy kazdym z N dodawan i zweryfikowane liczby o bledzie
    kwantyzacji (7-8e-5 RMSE, 112-240x ponizej podlogi szumu) przestalyby
    obowiazywac.
    """
    return np.mean(np.stack(specs, axis=0), axis=0, dtype=np.float32)


def sigma_1_from_specs(specs):
    """Szum POJEDYNCZEGO renderu, oszacowany z podzialu `specs` na dwie polowki.

        sigma_1 = RMSE(A, B) / sqrt(2) * sqrt(h)

    gdzie A, B to srednie z h = len(specs)//2 rozlacznych renderow kazda.
    Dzielenie przez sqrt(2) bierze sie stad, ze RMSE dwoch niezaleznych estymat
    o szumie sigma_h wynosi sqrt(2)*sigma_h; mnozenie przez sqrt(h) przelicza
    szum estymaty z h renderow na szum pojedynczego renderu.

    Przy nieparzystym len(specs) ostatni render nie wchodzi do PODZIALU (bo
    polowki musza byc rownoliczne), ale wchodzi do estymaty koncowej — patrz
    snr_from_specs().
    """
    n = len(specs)
    h = n // 2
    if h < 1:
        raise ValueError(f"potrzeba >= 2 renderow do podzialu na polowki, jest {n}")
    a = _mean_f32(specs[:h])
    b = _mean_f32(specs[h:2 * h])
    return _rmse(a, b) / np.sqrt(2.0) * np.sqrt(h), h


def snr_from_specs(specs, signal=SIGNAL_10DEG):
    """SNR estymaty zbudowanej ze WSZYSTKICH `len(specs)` renderow.

    Estymata koncowa usrednia n renderow, wiec jej szum to sigma_n =
    sigma_1/sqrt(n), i to on stoi w mianowniku:

        snr = SIGNAL_10DEG / sigma_n = SIGNAL_10DEG * sqrt(n) / sigma_1

    UWAGA — odstepstwo od doslownego zapisu w GENERATOR_PARAMS.md §3.4.1.
    Dokument podaje tam `snr = SIGNAL_10DEG / (RMSE(A,B)/sqrt(2))`. To wyrazenie
    jest o czynnik sqrt(2) za male: RMSE(A,B)/sqrt(2) to szum estymaty z POLOWY
    renderow (sigma_{n/2}), a nie z wszystkich n. Przy dokladnie trafionym N
    dawaloby snr = 3.5/sqrt(2) = 2.47, czyli ponizej progu ZAWSZE — petla
    weryfikacyjna z §3.4 dorenderowywalaby kazda probke, co przeczy zdaniu tego
    samego paragrafu, ze "kosztuje tylko dorenderowanie nielicznych przypadkow".
    Wersja uzyta tutaj jest samouzgodniona z regula na N z §3.2: jesli
    n = (3.5*sigma_1/SIGNAL)^2, to sigma_n = SIGNAL/3.5, czyli snr = 3.5 dokladnie.
    Jest to tez ta sama dekompozycja, ktora zapisano w komentarzach
    diagnose_rlr_noise.py:1764-1765 i 2026-2027 ("estymata full ma 2N renderow,
    wiec jej szum to sigma_N/sqrt(2) = noise/2").
    """
    n = len(specs)
    sigma_1, _h = sigma_1_from_specs(specs)
    sigma_n = sigma_1 / np.sqrt(n)
    snr = float(signal / sigma_n) if sigma_n > 0 else float("inf")
    return snr, float(sigma_1)


def plan_n(sigma_1):
    """sigma_1 -> (n_raw, n_planned, clamped) wg GENERATOR_PARAMS.md §3.2."""
    n_raw = int(np.ceil((TARGET_SNR * sigma_1 / SIGNAL_10DEG) ** 2)) if sigma_1 > 0 else N_MIN
    n_planned = int(min(max(n_raw, N_MIN), N_MAX))
    clamped = "min" if n_raw < N_MIN else ("max" if n_raw > N_MAX else "")
    return n_raw, n_planned, clamped


# ---------------------------------------------------------------------------
# Magazyn HDF5
# ---------------------------------------------------------------------------
SPEC_SHAPE = (2, 257, 166)
CLAMPED_DTYPE = "S4"  # b"", b"min", b"max" — §3.4.1 definiuje `clamped` jako str

SAMPLE_FIELDS = [
    # (nazwa, ksztalt bez pierwszego wymiaru, dtype, kompresja)
    ("echo", SPEC_SHAPE, "float16", True),
    ("rgb", (128, 128, 4), "uint8", True),
    ("depth", (128, 128), "float32", True),
    ("location_id", (), "int32", False),
    ("angle_deg", (), "int16", False),
    ("position", (3,), "float32", False),
    ("snr_probe", (), "float32", False),
    ("snr_final", (), "float32", False),
    ("n_total", (), "int16", False),
    ("n_rendered_extra", (), "int16", False),
    # Ponizsze cztery sa stale w obrebie lokalizacji, ale §3.4.1 definiuje je
    # jako pola PROBKI — trzymamy je per probka, zeby dataset dalo sie czytac
    # bez laczenia z tabela lokalizacji, i dodatkowo per lokalizacja w /locations.
    ("n_planned", (), "int16", False),
    ("n_raw", (), "int32", False),
    ("sigma_1_probe", (), "float32", False),
    ("n_probe", (), "int16", False),
    ("clamped", (), CLAMPED_DTYPE, False),
    ("written", (), "uint8", False),
]

LOCATION_FIELDS = [
    ("loc_id", (), "int32"),
    ("sigma_1_probe", (), "float32"),
    ("n_raw", (), "int32"),
    ("n_planned", (), "int16"),
    ("clamped", (), CLAMPED_DTYPE),
    ("position", (3,), "float32"),
    ("probe_seconds", (), "float32"),
    ("seconds", (), "float32"),
    ("decided", (), "uint8"),
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_dir):
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return out.stdout.strip()
        return f"<git rev-parse zwrocilo {out.returncode}: {out.stderr.strip()}>"
    except (OSError, subprocess.SubprocessError) as e:
        return f"<niedostepne: {type(e).__name__}: {e}>"


def build_file_attrs(scene, loc_ids):
    """Komplet atrybutow pliku — pelna reprodukowalnosc bez zagladania do repo."""
    import habitat_sim

    return {
        # co i czym
        "scene": scene,
        "dataset": "replica",
        "script": SCRIPT_PATH.name,
        "script_version": SCRIPT_VERSION,
        "spec_document": str(SPEC_DOC.relative_to(REPO_ROOT)),
        # GENERATOR_PARAMS.md §1 + §2
        "indirect_ray_count": INDIRECT_RAY_COUNT,
        "thread_count": THREAD_COUNT,
        "sample_rate": 44100,
        "enable_materials": True,
        "channel_type": "Binaural",
        "channel_count": 2,
        "listener_height": SENSOR_HEIGHT,
        "camera_height": SENSOR_HEIGHT,
        "camera_resolution": list(CAMERA_RESOLUTION),
        "camera_hfov_deg": CAMERA_HFOV,
        "scene_id": str(scene_mesh(scene).relative_to(REPO_ROOT)),
        "load_semantic_mesh": True,
        "create_renderer": True,
        "enable_physics": False,
        "gpu_device_id": 0,
        "averaging_domain": AVERAGING_DOMAIN,
        "material_config": str(MATERIAL_CONFIG.relative_to(REPO_ROOT)),
        "material_config_sha256": _sha256(MATERIAL_CONFIG),
        # potok spektrogramu
        "chirp": str(CHIRP_PATH.relative_to(REPO_ROOT)),
        "chirp_sha256": _sha256(CHIRP_PATH),
        "echo_ms": 60,
        "echo_samples": 2646,
        "stft_n_fft": 512,
        "stft_win_length": 64,
        "stft_hop_length": 16,
        "spectrogram_shape": list(SPEC_SHAPE),
        # regula adaptacyjna, §3.2
        "signal_10deg": SIGNAL_10DEG,
        "target_snr": TARGET_SNR,
        "n_min": N_MIN,
        "n_max": N_MAX,
        "n_probe": N_PROBE,
        "angles_deg": list(ANGLES_DEG),
        # format zapisu, §4.1
        "echo_dtype": "float16",
        "echo_accumulation_dtype": "float32",
        "depth_dtype": "float32 (surowe metry, bez klipowania)",
        "rgb_dtype": "uint8 RGBA",
        # zrodla pozycji
        "locations_source": str(LOCATIONS_PKL.relative_to(REPO_ROOT)),
        "n_locations": len(loc_ids),
        "n_samples_expected": len(loc_ids) * N_ANGLES,
        "y_source": ("graph.pkl node['point'][1]; stala sceny dla lokalizacji spoza grafu "
                     "(PKL_FORMAT.md, GENERATOR_PARAMS.md §2 poprawka 2026-07-28)"),
        "xz_source": "points.txt: x = a, z = -b",
        # srodowisko
        "habitat_sim_commit": _git_commit(REPO_ROOT / "habitat-sim"),
        "rlr_audio_propagation_commit": _git_commit(REPO_ROOT / "habitat-sim/src/deps/rlr-audio-propagation"),
        "habitat_sim_version": str(getattr(habitat_sim, "__version__", "<brak>")),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        # 1 od 2026-07-28 (wczesniej 2 — zbedna, nieodczytywana symulacja).
        # Pliki z wartoscia 2 powstaly stara sciezka; rownowaznosc statystyczna
        # obu potwierdzona pomiarem, patrz GENERATOR_PARAMS.md §4.3.
        "audio_sims_per_render": 1,
        "snr_definition": "SIGNAL_10DEG * sqrt(n) / sigma_1, sigma_1 z podzialu na polowki",
    }


class DatasetStore:
    """Plik HDF5 jednej sceny. Rozmiar ustalony z gory (n_lok x 36 probek).

    Chunkowanie po pierwszym wymiarze — niezapisane chunki nie zajmuja miejsca
    na dysku, wiec plik czesciowy nie marnuje przestrzeni, a indeks probki
    (i_lok * 36 + i_kat) jest deterministyczny, co upraszcza wznawianie.
    """

    def __init__(self, path, scene, loc_ids, positions, attrs=None, mode="w", flush_every=36):
        import h5py

        self.path = Path(path)
        self.scene = scene
        self.loc_ids = list(loc_ids)
        self.n_loc = len(self.loc_ids)
        self.n_samples = self.n_loc * N_ANGLES
        self.loc_index = {lid: i for i, lid in enumerate(self.loc_ids)}
        self.flush_every = flush_every
        self._since_flush = 0
        self._last_sidecar = 0.0

        if mode == "w":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = h5py.File(self.path, "w")
            for name, shape, dtype, compress in SAMPLE_FIELDS:
                kw = dict(chunks=(1,) + shape) if compress else {}
                if compress:
                    kw["compression"] = "gzip"
                    kw["compression_opts"] = 4
                self.f.create_dataset(name, shape=(self.n_samples,) + shape, dtype=dtype, **kw)
            grp = self.f.create_group("locations")
            for name, shape, dtype in LOCATION_FIELDS:
                grp.create_dataset(name, shape=(self.n_loc,) + shape, dtype=dtype)
            grp["loc_id"][:] = np.array(self.loc_ids, dtype=np.int32)
            grp["position"][:] = np.stack([positions[l] for l in self.loc_ids]).astype(np.float32)
            for k, v in (attrs or {}).items():
                self.f.attrs[k] = v
            self.f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            self.f.attrs["runs"] = json.dumps([])
            self.f.attrs["render_seconds_total"] = 0.0
            self.f.attrs["renders_total"] = 0
        else:
            self.f = h5py.File(self.path, "r+")
            self._check_compatible(attrs or {})
            # Nie blad, ale musi byc widoczne: plik zaczety stara sciezka (2
            # symulacje audio na render) dopisywany nowa (1). Rownowaznosc obu
            # potwierdzona pomiarowo (GENERATOR_PARAMS.md §4.3), wiec dane
            # pozostaja jednorodne rozkladowo — ale fakt ma zostac w logu.
            old = int(self.f.attrs.get("audio_sims_per_render", 0))
            new = int((attrs or {}).get("audio_sims_per_render", old))
            if old and old != new:
                self._path_mismatch = (old, new)
            else:
                self._path_mismatch = None

        self.written = self.f["written"][:].astype(bool)
        self.decided = self.f["locations/decided"][:].astype(bool)

    def _check_compatible(self, attrs):
        """Wznowienie na pliku wygenerowanym innymi parametrami byloby cichym
        zanieczyszczeniem datasetu — sprawdzamy jawnie i przerywamy."""
        critical = ("scene", "indirect_ray_count", "thread_count", "listener_height",
                    "signal_10deg", "target_snr", "n_min", "n_max", "n_probe",
                    "material_config_sha256", "chirp_sha256", "n_locations", "y_source")
        problems = []
        for key in critical:
            if key not in attrs:
                continue
            if key not in self.f.attrs:
                problems.append(f"{key}: brak w pliku")
                continue
            old, new = self.f.attrs[key], attrs[key]
            if isinstance(old, bytes):
                old = old.decode()
            if isinstance(old, np.generic):
                old = old.item()
            if old != new:
                problems.append(f"{key}: w pliku {old!r}, teraz {new!r}")
        if problems:
            raise RuntimeError(
                "Plik istniejacy powstal z innymi parametrami niz obecne — wznowienie "
                "zmieszaloby dwa rozne zbiory:\n  " + "\n  ".join(problems))
        if self.f["written"].shape[0] != self.n_samples:
            raise RuntimeError(
                f"Plik ma {self.f['written'].shape[0]} miejsc na probki, oczekiwano {self.n_samples}")

    # -- odczyt stanu -------------------------------------------------------
    def sample_index(self, loc_id, angle_deg):
        return self.loc_index[loc_id] * N_ANGLES + ANGLES_DEG.index(int(angle_deg))

    def location_done(self, loc_id):
        i = self.loc_index[loc_id]
        return bool(self.written[i * N_ANGLES:(i + 1) * N_ANGLES].all())

    def missing_angles(self, loc_id):
        i = self.loc_index[loc_id]
        block = self.written[i * N_ANGLES:(i + 1) * N_ANGLES]
        return [ANGLES_DEG[k] for k in range(N_ANGLES) if not block[k]]

    def get_decision(self, loc_id):
        """Decyzja o N zapadla wczesniej (przed przerwaniem) -> odtwarzamy ja.

        DLACZEGO: ponowne odpalenie sondy po wznowieniu daloby inne sigma_1
        (inny stan RNG), a wiec potencjalnie inne n_planned dla pozostalych
        orientacji tej samej lokalizacji. Zlamaloby to zalozenie, ze wszystkie
        36 orientacji jednej lokalizacji ma identyczne N (§3.3 pkt 2), i to
        w sposob skorelowany z momentem awarii.
        """
        i = self.loc_index[loc_id]
        if not self.decided[i]:
            return None
        g = self.f["locations"]
        return {
            "sigma_1_probe": float(g["sigma_1_probe"][i]),
            "n_raw": int(g["n_raw"][i]),
            "n_planned": int(g["n_planned"][i]),
            "clamped": g["clamped"][i].decode(),
        }

    # -- zapis --------------------------------------------------------------
    def put_decision(self, loc_id, sigma_1, n_raw, n_planned, clamped, probe_seconds):
        i = self.loc_index[loc_id]
        g = self.f["locations"]
        g["sigma_1_probe"][i] = np.float32(sigma_1)
        g["n_raw"][i] = np.int32(n_raw)
        g["n_planned"][i] = np.int16(n_planned)
        g["clamped"][i] = clamped.encode()
        g["probe_seconds"][i] = np.float32(probe_seconds)
        g["decided"][i] = 1
        self.decided[i] = True

    def put_location_time(self, loc_id, seconds):
        self.f["locations/seconds"][self.loc_index[loc_id]] = np.float32(seconds)

    def put_sample(self, loc_id, angle_deg, echo_f32, rgb, depth, position, meta):
        idx = self.sample_index(loc_id, angle_deg)
        f = self.f
        # Rzutowanie na float16 DOPIERO tutaj, na gotowej sredniej (§4.1).
        f["echo"][idx] = echo_f32.astype(np.float16)
        f["rgb"][idx] = rgb
        f["depth"][idx] = depth
        f["location_id"][idx] = np.int32(loc_id)
        f["angle_deg"][idx] = np.int16(angle_deg)
        f["position"][idx] = position.astype(np.float32)
        for key in ("snr_probe", "snr_final", "sigma_1_probe"):
            f[key][idx] = np.float32(meta[key])
        for key in ("n_total", "n_rendered_extra", "n_planned", "n_probe"):
            f[key][idx] = np.int16(meta[key])
        f["n_raw"][idx] = np.int32(meta["n_raw"])
        f["clamped"][idx] = meta["clamped"].encode()
        # `written` na koncu: awaria w polowie zapisu zostawia flage 0, wiec
        # wznowienie po prostu nadpisze te probke od nowa.
        f["written"][idx] = 1
        self.written[idx] = True
        self._since_flush += 1
        return idx

    def maybe_flush(self, force=False):
        if force or self._since_flush >= self.flush_every:
            self.f.flush()
            self._since_flush = 0
            return True
        return False

    def n_written(self):
        return int(self.written.sum())

    def write_progress_sidecar(self, extra=None):
        payload = {
            "scene": self.scene,
            "n_samples_expected": self.n_samples,
            "n_written": self.n_written(),
            "n_locations": self.n_loc,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
        }
        payload.update(extra or {})
        tmp = scene_progress(self.scene).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(scene_progress(self.scene))
        self._last_sidecar = time.time()

    def sidecar_heartbeat(self, min_interval=30.0):
        """Sidecar sluzy --status za sygnal zycia, wiec nie moze sie odswiezac
        raz na lokalizacje: przy N=29 jedna lokalizacja to ~5 min, czyli dluzej
        niz prog swiezosci. Zapis maleńkiego JSON-a co 30 s jest darmowy."""
        if time.time() - self._last_sidecar >= min_interval:
            self.write_progress_sidecar()

    def close(self, final_attrs=None):
        for k, v in (final_attrs or {}).items():
            self.f.attrs[k] = v
        self.f.flush()
        self.f.close()


# ---------------------------------------------------------------------------
# Renderowanie
# ---------------------------------------------------------------------------
class Renderer:
    """Jeden dlugo zyjacy Simulator + jeden render = jedna probka RIR.

    Sciezka renderowania jest CELOWO wywolaniem `test_rlr_audio.phase3_echolocation`,
    a nie wlasna kopia: to dokladnie ta funkcja, ktora wygenerowala cala
    charakterystyke szumu (diagnose_rlr_noise.py wola ja przez render_raw()).
    Przepisanie jej tutaj groziloby cicha zmiana kolejnosci wywolan audio,
    a wiec i sekwencji RNG, wzgledem ktorej skalibrowano SIGNAL_10DEG i rozklad N.
    """

    def __init__(self, scene, log):
        import test_rlr_audio as tra

        self.tra = tra
        self.log = log
        self.scene = scene

        class _Args:
            pass

        args = _Args()
        args.scene = str(scene_mesh(scene))
        args.sensor_height = SENSOR_HEIGHT
        args.material_config = str(MATERIAL_CONFIG)
        args.out_dir = str(OUT_ROOT / "_rlr_scratch")
        args.indirect_ray_count = INDIRECT_RAY_COUNT
        args.thread_count = THREAD_COUNT
        args.gpu_device_id = 0

        t0 = time.perf_counter()
        self.sim = tra.build_simulator(args)
        log.info("Simulator zbudowany w %.1f s (%s, %d promieni, %d watek)",
                 time.perf_counter() - t0, scene, INDIRECT_RAY_COUNT, THREAD_COUNT)

        import librosa
        self.chirp, _sr = librosa.load(str(CHIRP_PATH), sr=tra.SAMPLE_RATE, mono=True)

        # setAudioMaterialsJSON() musi paść PRZED pierwszym runSimulation (to ono
        # wola loadSemanticMesh, ktore zamyka baze materialow). Kazde kolejne
        # wywolanie jest w AudioSensor.cpp:173-182 no-opem konczacym sie
        # ostrzezeniem w logu, wiec podajemy config tylko przy pierwszym
        # renderze — zachowanie symulatora identyczne, log krotszy o ~600 tys. linii.
        self._materials_pending = True
        self.n_renders = 0
        self.render_seconds = 0.0

    def render(self, position, angle_deg):
        """-> (spec float32 (2,257,166), rgb uint8, depth float32)"""
        mc = str(MATERIAL_CONFIG) if self._materials_pending else None
        t0 = time.perf_counter()
        # run_simulation=False: symulacje akustyczna uruchamia raz
        # get_sensor_observations() (przez Sensor._get_audio_observation()) i to
        # jej wynik trafia do obserwacji. Jawne runSimulation() liczylo druga,
        # nieodczytywana — polowa czasu renderu szla do kosza. Rownowaznosc obu
        # sciezek zweryfikowana pomiarowo 2026-07-28 na dwoch skrajnych pozycjach
        # (najglosniejsza i najcichsza zmierzona): wszystkie roznice ponizej
        # 1.3 SE, N z reguly 30->31 i 4->4. Szczegoly: GENERATOR_PARAMS.md §4.3.
        obs, _listener, _rot = self.tra.phase3_echolocation(
            self.sim, position, float(angle_deg), mc, run_simulation=False)
        self._materials_pending = False

        rir = np.transpose(np.array(obs["audio_sensor"]))
        if rir.size == 0 or not np.any(rir):
            raise RuntimeError(
                f"RIR to same zera dla pozycji {position} / kata {angle_deg} — symulacja akustyczna "
                "nie zwrocila echa (patrz test_rlr_audio.phase4_validate_rir)")
        _echo, spec = self.tra.render_spectrogram(rir, self.chirp)
        if spec.shape != SPEC_SHAPE:
            raise RuntimeError(f"spektrogram ma ksztalt {spec.shape}, oczekiwano {SPEC_SHAPE}")

        # np.flip() w Sensor.get_observation() zwraca WIDOK na bufor sensora,
        # ktory kolejny render nadpisze — stad jawna kopia.
        rgb = np.array(obs["rgb"], dtype=np.uint8, copy=True)
        depth = np.array(obs["depth"], dtype=np.float32, copy=True)

        self.n_renders += 1
        self.render_seconds += time.perf_counter() - t0
        return spec.astype(np.float32, copy=False), rgb, depth

    def close(self):
        try:
            self.sim.close()
        finally:
            self.sim = None


# ---------------------------------------------------------------------------
# Generacja
# ---------------------------------------------------------------------------
def _fmt_hms(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def generate(scene, limit=None, resume=False, force=False, flush_every=36):
    log = setup_logging(scene)
    _install_signal_handlers(log)

    loc_ids, positions = load_scene_locations(scene)
    if limit is not None:
        loc_ids = loc_ids[:limit]
        log.warning("--limit %d: generuje tylko %d z %d lokalizacji. Plik pozostanie NIEKOMPLETNY "
                    "(rozmiar ustalony na pelna scene), --verify zglosi brak probek.",
                    limit, len(loc_ids), len(positions))

    all_loc_ids = sorted(positions)  # rozmiar pliku zawsze na pelna scene
    attrs = build_file_attrs(scene, all_loc_ids)

    path = scene_h5(scene)
    if path.exists() and not (resume or force):
        raise SystemExit(
            f"{path} juz istnieje. Uzyj --resume, zeby dopisac brakujace probki, "
            f"albo --force, zeby wygenerowac od zera (NADPISZE istniejace dane).")
    mode = "r+" if (path.exists() and resume) else "w"
    if mode == "w" and path.exists():
        log.warning("--force: nadpisuje %s", path)

    store = DatasetStore(path, scene, all_loc_ids, positions, attrs=attrs, mode=mode,
                         flush_every=flush_every)
    log.info("Plik: %s (tryb %s), zapisanych probek na starcie: %d/%d",
             path, mode, store.n_written(), store.n_samples)
    if getattr(store, "_path_mismatch", None):
        old, new = store._path_mismatch
        log.warning("Plik zaczeto przy audio_sims_per_render=%d, dopisujemy przy %d. "
                    "Rownowaznosc obu sciezek potwierdzona pomiarem (GENERATOR_PARAMS.md §4.3), "
                    "ale scena bedzie mieszana — rozwaz regeneracje od zera (--force).", old, new)
    # Od razu, zeby --status w drugim terminalu widzial scene jako "W TOKU"
    # jeszcze zanim skonczy sie pierwsza lokalizacja.
    store.write_progress_sidecar({"running": True})

    t_start = time.time()
    renderer = None
    n_locations_done = 0
    n_list = []
    extra_renders_total = 0
    exit_code = 0
    interrupted = False

    decisions_fh = open(scene_decisions(scene), "a", encoding="utf-8")
    try:
        todo = [l for l in loc_ids if not store.location_done(l)]
        log.info("Do zrobienia: %d lokalizacji (%d juz kompletnych)",
                 len(todo), len(loc_ids) - len(todo))
        if not todo:
            log.info("Nic do zrobienia — wszystkie zadane lokalizacje sa kompletne.")
        else:
            renderer = Renderer(scene, log)

        for pos_in_todo, loc_id in enumerate(todo, start=1):
            if _INTERRUPTED:
                interrupted = True
                break
            t_loc = time.time()
            position = positions[loc_id]
            missing = store.missing_angles(loc_id)

            # --- decyzja o N: sonda albo odtworzenie z pliku ------------------
            decision = store.get_decision(loc_id) if resume else None
            probe_specs = None
            probe_seconds = 0.0
            if decision is None:
                t_probe = time.time()
                probe_specs = [renderer.render(position, 0.0)[0] for _ in range(N_PROBE)]
                probe_seconds = time.time() - t_probe
                sigma_1, _h = sigma_1_from_specs(probe_specs)
                n_raw, n_planned, clamped = plan_n(sigma_1)
                store.put_decision(loc_id, sigma_1, n_raw, n_planned, clamped, probe_seconds)
            else:
                sigma_1 = decision["sigma_1_probe"]
                n_raw, n_planned, clamped = (decision["n_raw"], decision["n_planned"],
                                             decision["clamped"])
                log.info("  lok %d: decyzja odtworzona z pliku (N=%d) — sonda nie jest powtarzana",
                         loc_id, n_planned)

            # --- 36 orientacji ------------------------------------------------
            loc_extra = 0
            for angle in ANGLES_DEG:
                if angle not in missing:
                    continue

                if angle == 0 and probe_specs is not None:
                    # KRYTYCZNE — jednorodnosc szumu miedzy orientacjami.
                    #
                    # Przy N < 8 sonda daje wiecej renderow niz potrzeba dla
                    # orientacji 0 stopni. Nadmiarowe ODRZUCAMY — kazda z 36
                    # orientacji uzywa dokladnie N renderow.
                    #
                    # Powod nie jest kosztowy (strata to 0.06 h na caly zbior),
                    # tylko metodologiczny: przy N=6 i uzyciu wszystkich 8
                    # renderow sondy orientacja 0 stopni mialaby szum nizszy
                    # o sqrt(6/8) = 13 %. W zmierzonym rozkladzie N dotyczy to
                    # ~1/3 lokalizacji. A 0 stopni jest jedna z czterech
                    # orientacji bazowych: w warunku 4-kierunkowym to 1 z 4
                    # probek (25 %), w 36-kierunkowym 1 z 36 (2.8 %) — czyli
                    # SREDNI POZIOM SZUMU roznilby sie systematycznie miedzy
                    # warunkami ablacji, skorelowany dokladnie ze zmienna
                    # eksperymentalna. Darmowe do unikniecia, wiec unikamy.
                    specs = list(probe_specs[:n_planned])
                    n_probe_used = len(specs)
                    first_rgb = first_depth = None
                    while len(specs) < n_planned:
                        s, r, d = renderer.render(position, angle)
                        specs.append(s)
                        if first_rgb is None:
                            first_rgb, first_depth = r, d
                    if first_rgb is None:
                        # Wszystkie N renderow pochodzi z sondy, ktora zapisywala
                        # tylko spektrogramy — dorenderowujemy sam obraz.
                        # Rendering wizualny jest deterministyczny (PKL_FORMAT.md),
                        # wiec ten render jest identyczny z kazdym innym przy tej
                        # samej pozie; kosztuje 0.2 ms (audio dominuje).
                        _s, first_rgb, first_depth = renderer.render(position, angle)
                else:
                    specs = []
                    n_probe_used = 0
                    first_rgb = first_depth = None
                    for _ in range(n_planned):
                        s, r, d = renderer.render(position, angle)
                        specs.append(s)
                        if first_rgb is None:
                            first_rgb, first_depth = r, d

                # --- weryfikacja po fakcie (§3.4) ----------------------------
                snr_probe, _ = snr_from_specs(specs)
                snr_final = snr_probe
                extra = 0
                guard = 0
                while snr_final < TARGET_SNR and len(specs) < N_MAX:
                    guard += 1
                    if guard > 8:
                        raise RuntimeError(
                            f"petla weryfikacyjna nie zbiegla po 8 iteracjach "
                            f"(lok {loc_id}, kat {angle}, n={len(specs)}, snr={snr_final:.3f})")
                    # Nie dokladamy po jednym renderze: przeliczamy WYMAGANE n
                    # z aktualnego (dokladniejszego niz 8-renderowa sonda)
                    # oszacowania sigma_1 i skaczemy tam od razu. Dokladanie po
                    # jednym i sprawdzanie po kazdym byloby optional stopping —
                    # zatrzymywaloby sie dokladnie wtedy, gdy oszacowanie szumu
                    # akurat wypadnie nisko, czyli z obciazeniem w dol.
                    sigma_now, _ = sigma_1_from_specs(specs)
                    need = int(np.ceil((TARGET_SNR * sigma_now / SIGNAL_10DEG) ** 2))
                    need = int(min(max(need, len(specs) + 1), N_MAX))
                    for _ in range(need - len(specs)):
                        specs.append(renderer.render(position, angle)[0])
                        extra += 1
                    snr_final, _ = snr_from_specs(specs)

                loc_extra += extra
                extra_renders_total += extra

                echo = _mean_f32(specs)   # estymata = mean(|STFT|), domena "mag" (§1, §3.3 pkt 3)
                store.put_sample(
                    loc_id, angle, echo, first_rgb, first_depth, position,
                    {"snr_probe": snr_probe, "snr_final": snr_final,
                     "sigma_1_probe": sigma_1, "n_raw": n_raw, "n_planned": n_planned,
                     "n_total": len(specs), "n_rendered_extra": extra,
                     "n_probe": n_probe_used, "clamped": clamped})
                if store.maybe_flush():
                    store.write_progress_sidecar()
                else:
                    store.sidecar_heartbeat()

                if _INTERRUPTED:
                    interrupted = True
                    break

            store.put_location_time(loc_id, time.time() - t_loc)
            store.maybe_flush(force=True)
            store.write_progress_sidecar()
            # Lokalizacja przerwana sygnalem ma zapisane tylko czesc katow — nie
            # liczy sie do sredniego N ani do tempa, bo zaklamalaby oba (jej czas
            # jest urwany w losowym miejscu).
            loc_complete = store.location_done(loc_id)
            if loc_complete:
                n_list.append(n_planned)
                n_locations_done += 1

            decisions_fh.write(json.dumps({
                "scene": scene, "loc_id": int(loc_id),
                "sigma_1_probe": round(float(sigma_1), 6),
                "n_raw": int(n_raw), "n_planned": int(n_planned), "clamped": clamped,
                "n_rendered_extra": int(loc_extra),
                "seconds": round(time.time() - t_loc, 2),
                "probe_seconds": round(probe_seconds, 2),
                "complete": bool(loc_complete),
                "utc": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            decisions_fh.flush()

            if loc_complete:
                per_loc = (time.time() - t_start) / n_locations_done
                eta = _fmt_hms(per_loc * (len(todo) - pos_in_todo))
                log.info("lok %3d/%-3d id=%-4d sigma1=%.5f N_raw=%-3d N=%-3d%s +%-3d dorend. | "
                         "sr.N %.1f | %5.1f s/lok | ETA %s",
                         pos_in_todo, len(todo), loc_id, sigma_1, n_raw, n_planned,
                         f" [{clamped}]" if clamped else "      ", loc_extra,
                         float(np.mean(n_list)), time.time() - t_loc, eta)
            else:
                log.warning("lok %3d/%-3d id=%-4d N=%-3d PRZERWANA — zapisano %d z %d katow, "
                            "decyzja o N zachowana w pliku; --resume dokonczy reszte",
                            pos_in_todo, len(todo), loc_id, n_planned,
                            N_ANGLES - len(store.missing_angles(loc_id)), N_ANGLES)

            if interrupted or _INTERRUPTED:
                interrupted = True
                break

    except KeyboardInterrupt:
        log.error("Przerwane drugim sygnalem — plik moze zawierac niedokonczona probke "
                  "(flaga `written` chroni przed jej odczytem; --resume ja nadpisze)")
        exit_code = 130
    except BaseException:
        log.error("BLAD generacji — pelny traceback ponizej. Plik zostanie zamkniety, "
                  "dotychczasowe probki sa zapisane; wznowienie: --resume")
        log.error("%s", traceback.format_exc())
        exit_code = 1
    finally:
        decisions_fh.close()
        seconds_render = renderer.render_seconds if renderer else 0.0
        n_renders = renderer.n_renders if renderer else 0
        if renderer is not None:
            renderer.close()

        prev_sec = float(store.f.attrs.get("render_seconds_total", 0.0))
        prev_ren = int(store.f.attrs.get("renders_total", 0))
        runs = json.loads(store.f.attrs.get("runs", "[]"))
        runs.append({
            "started_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(),
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(time.time() - t_start, 1),
            "renders": n_renders,
            "locations": n_locations_done,
            "interrupted": bool(interrupted),
            "limit": limit,
            "hostname": socket.gethostname(),
            "exit_code": exit_code,
        })
        s_per_render = ((prev_sec + seconds_render) / (prev_ren + n_renders)
                        if (prev_ren + n_renders) else 0.0)
        store.close({
            "runs": json.dumps(runs),
            "render_seconds_total": prev_sec + seconds_render,
            "renders_total": prev_ren + n_renders,
            "seconds_per_render": s_per_render,
            "last_update_utc": datetime.now(timezone.utc).isoformat(),
            "complete": bool(store.n_written() == store.n_samples),
        })
        store.write_progress_sidecar({"complete": bool(store.n_written() == store.n_samples),
                                      "seconds_per_render": s_per_render,
                                      "interrupted": bool(interrupted)})

        wall = time.time() - t_start
        log.info("=" * 78)
        log.info("scena %s | lokalizacji w tym przebiegu %d | renderow %d | %.4f s/render",
                 scene, n_locations_done, n_renders, s_per_render)
        log.info("probek w pliku %d/%d (%.1f %%) | dorenderowanych renderow %d | czas %s",
                 store.n_written(), store.n_samples,
                 100.0 * store.n_written() / max(store.n_samples, 1), extra_renders_total,
                 _fmt_hms(wall))
        if n_list:
            log.info("N wybrane: mediana %d, srednia %.2f, zakres %d-%d",
                     int(np.median(n_list)), float(np.mean(n_list)), min(n_list), max(n_list))
        if interrupted:
            log.info("PRZERWANE czysto na granicy probki. Wznowienie:")
            log.info("  python %s --scene %s --resume", SCRIPT_PATH.name, scene)
        log.info("=" * 78)

    # Czyste przerwanie to nie blad — kod 0, zgodnie z wymaganiem odpornosci.
    return exit_code


# ---------------------------------------------------------------------------
# --dry-run (bez GPU)
# ---------------------------------------------------------------------------
def dry_run(scene, limit=None):
    log = setup_logging()
    loc_ids, positions = load_scene_locations(scene)
    n_all = len(loc_ids)
    if limit is not None:
        loc_ids = loc_ids[:limit]
    n_loc = len(loc_ids)
    n_samples = n_loc * N_ANGLES

    # Renderow na lokalizacje: sonda (N_PROBE) + 36*N, minus te rendery sondy,
    # ktore zostana wykorzystane przy 0 stopni (min(N_PROBE, N)).
    n_mean = MEAN_N_SPEC
    renders_per_loc = N_PROBE + N_ANGLES * n_mean - min(N_PROBE, n_mean)
    renders = n_loc * renders_per_loc
    seconds = renders * S_PER_RENDER_SPEC
    raw_bytes = n_samples * BYTES_PER_SAMPLE

    y_values = np.array([positions[l][1] for l in loc_ids])

    print(f"\n=== --dry-run: {scene} ===")
    print(f"  plik wyjsciowy            {scene_h5(scene)}")
    print(f"  istnieje                  {'TAK' if scene_h5(scene).exists() else 'nie'}")
    print(f"  scena (mesh)              {scene_mesh(scene)}")
    print(f"    istnieje                {'TAK' if scene_mesh(scene).exists() else 'NIE — BLAD'}")
    print(f"  navmesh                   {'TAK' if (SCENE_ROOT / scene / 'habitat/mesh_semantic.navmesh').exists() else 'NIE — BLAD'}")
    print(f"  lokalizacje (pkl)         {n_loc}" + (f" z {n_all} (--limit)" if limit else ""))
    print(f"  orientacje                {N_ANGLES} (co 10 st.)")
    print(f"  PROBKI                    {n_samples}")
    print(f"  y agenta (z graph.pkl)    {y_values[0]:.6f}"
          f"{'' if np.ptp(y_values) == 0 else f' (UWAGA: niestale, rozrzut {np.ptp(y_values):.6f})'}")
    print(f"  pozycja pierwsza          {positions[loc_ids[0]]}")
    print()
    print(f"  zakladane srednie N       {n_mean} (GENERATOR_PARAMS.md §3.1)")
    print(f"  renderow / lokalizacje    {renders_per_loc:.1f}  (sonda {N_PROBE} + 36xN - odzysk)")
    print(f"  RENDEROW LACZNIE          {renders:,.0f}".replace(",", " "))
    print(f"  tempo (spec)              {S_PER_RENDER_SPEC} s/render")
    print(f"  CZAS SZACOWANY            {_fmt_hms(seconds)}  ({seconds/3600:.2f} h)")
    print(f"    + dorenderowanie §3.4   nieliczne, nieuwzglednione")
    print()
    print(f"  bajtow / probke           {BYTES_PER_SAMPLE:,}".replace(",", " ") +
          "  (echo 170 648 + rgb 65 536 + depth 65 536)")
    print(f"  ROZMIAR bez kompresji     {raw_bytes/2**30:.2f} GiB")
    print(f"    po gzip -4              mniej; rzeczywisty rozmiar w --status")
    free = os.statvfs(OUT_ROOT.parent if OUT_ROOT.parent.exists() else REPO_ROOT)
    print(f"  wolne na dysku            {free.f_bavail * free.f_frsize / 2**30:.1f} GiB")
    print()
    print("  wejscia:")
    for label, p in (("chirp", CHIRP_PATH), ("materialy", MATERIAL_CONFIG),
                     ("lokalizacje", LOCATIONS_PKL), ("points.txt", points_txt(scene)),
                     ("graph.pkl", graph_pkl(scene))):
        print(f"    {label:<12} {'OK ' if p.exists() else 'BRAK'} {p}")
    return 0


# ---------------------------------------------------------------------------
# --verify (bez GPU)
# ---------------------------------------------------------------------------
def _open_readonly(path):
    import h5py
    # locking=False pozwala czytac plik, ktory inny proces trzyma otwarty do
    # zapisu — dzieki temu --verify/--status dzialaja w drugim terminalu.
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


def _hist(values, bins=None, width=46):
    values = np.asarray(values)
    if values.size == 0:
        return ["    (brak danych)"]
    if bins is None:
        uniq = np.unique(values)
        if uniq.size <= 20 and np.all(uniq == uniq.astype(int)):
            counts = [(int(u), int((values == u).sum())) for u in uniq]
            top = max(c for _, c in counts)
            return [f"    {u:>6} | {'#' * max(1, int(width * c / top)):<{width}} {c}"
                    for u, c in counts]
        bins = 12
    counts, edges = np.histogram(values, bins=bins)
    top = max(counts.max(), 1)
    return [f"    {edges[i]:>7.3f}-{edges[i+1]:<7.3f} | "
            f"{'#' * max(0, int(width * counts[i] / top)):<{width}} {counts[i]}"
            for i in range(len(counts))]


def verify(scene, n_plots=3, seed=0):
    print(f"\n{'=' * 78}\n  WERYFIKACJA: {scene}\n{'=' * 78}")
    path = scene_h5(scene)
    if not path.exists():
        print(f"\n  WERDYKT: FAIL — plik nie istnieje: {path}")
        return 1

    failures, warnings_ = [], []
    f = _open_readonly(path)
    try:
        written = f["written"][:].astype(bool)
        n_slots = written.size
        n_written = int(written.sum())
        idx = np.flatnonzero(written)

        loc_id = f["location_id"][:][idx]
        angle = f["angle_deg"][:][idx]
        n_planned = f["n_planned"][:][idx].astype(int)
        n_total = f["n_total"][:][idx].astype(int)
        n_extra = f["n_rendered_extra"][:][idx].astype(int)
        n_raw = f["n_raw"][:][idx].astype(int)
        snr_probe = f["snr_probe"][:][idx].astype(float)
        snr_final = f["snr_final"][:][idx].astype(float)
        sigma_1 = f["sigma_1_probe"][:][idx].astype(float)
        clamped = np.array([c.decode() for c in f["clamped"][:][idx]])

        target_snr = float(f.attrs.get("target_snr", TARGET_SNR))
        n_max = int(f.attrs.get("n_max", N_MAX))
        n_loc_expected = int(f.attrs.get("n_locations", n_slots // N_ANGLES))
        expected_samples = n_loc_expected * N_ANGLES

        # ---------------- KOMPLETNOSC -----------------------------------
        print("\n--- KOMPLETNOSC ---")
        print(f"  probek zapisanych         {n_written} / {expected_samples} "
              f"({100.0*n_written/max(expected_samples,1):.2f} %)")
        if n_written != expected_samples:
            failures.append(f"kompletnosc: {n_written} probek zamiast {expected_samples} "
                            f"({n_loc_expected} lokalizacji x {N_ANGLES})")

        locs_present = np.unique(loc_id)
        print(f"  lokalizacji obecnych      {locs_present.size} / {n_loc_expected}")
        bad_angles = []
        for lid in locs_present:
            angs = angle[loc_id == lid]
            if angs.size != N_ANGLES or set(angs.tolist()) != set(ANGLES_DEG):
                bad_angles.append((int(lid), angs.size, len(set(angs.tolist()))))
        if bad_angles:
            failures.append(f"{len(bad_angles)} lokalizacji nie ma kompletu 36 unikalnych katow "
                            f"(np. {bad_angles[:5]})")
        else:
            print(f"  komplet 36 katow          OK dla wszystkich {locs_present.size} lokalizacji")
        if np.unique(np.stack([loc_id, angle]), axis=1).shape[1] != loc_id.size:
            failures.append("wystepuja zduplikowane pary (location_id, angle_deg)")
        else:
            print("  duplikaty (lok, kat)      brak")

        try:
            expected_locs = set(load_scene_locations(scene)[0])
            got = set(int(x) for x in locs_present)
            if got - expected_locs:
                failures.append(f"location_id spoza scene_observations_128.pkl: "
                                f"{sorted(got - expected_locs)[:10]}")
            missing_locs = expected_locs - got
            if missing_locs:
                failures.append(f"brakuje {len(missing_locs)} lokalizacji z pkl: "
                                f"{sorted(missing_locs)[:10]}")
            else:
                print(f"  zgodnosc z pkl            OK ({len(expected_locs)} lokalizacji)")
        except Exception as e:   # brak pkl nie jest bledem DANYCH, tylko srodowiska
            warnings_.append(f"nie udalo sie sprawdzic zbioru lokalizacji wzgledem pkl: {e}")

        # ---------------- INTEGRALNOSC ----------------------------------
        print("\n--- INTEGRALNOSC ---")
        expected_dtypes = {"echo": "float16", "rgb": "uint8", "depth": "float32",
                           "location_id": "int32", "angle_deg": "int16",
                           "position": "float32", "snr_probe": "float32",
                           "snr_final": "float32", "n_total": "int16",
                           "n_rendered_extra": "int16", "n_planned": "int16",
                           "n_raw": "int32", "sigma_1_probe": "float32"}
        expected_shapes = {"echo": SPEC_SHAPE, "rgb": (128, 128, 4), "depth": (128, 128),
                           "position": (3,)}
        for name, dt in expected_dtypes.items():
            if str(f[name].dtype) != dt:
                failures.append(f"dtype {name}: {f[name].dtype}, oczekiwano {dt}")
        for name, sh in expected_shapes.items():
            if tuple(f[name].shape[1:]) != sh:
                failures.append(f"ksztalt {name}: {f[name].shape[1:]}, oczekiwano {sh}")
        if not failures:
            print(f"  ksztalty i dtype          OK (echo {f['echo'].shape}, "
                  f"rgb {f['rgb'].shape}, depth {f['depth'].shape})")

        # Skan w kawalkach — caly `echo` nie miesci sie wygodnie w RAM dla duzych scen.
        echo_max = 0.0
        n_nan = n_inf = n_zero = n_depth_bad = 0
        depth_min, depth_max = np.inf, -np.inf
        CH = 256
        for start in range(0, n_slots, CH):
            sl = slice(start, min(start + CH, n_slots))
            mask = written[sl]
            if not mask.any():
                continue
            e = f["echo"][sl][mask].astype(np.float32)
            d = f["depth"][sl][mask]
            n_nan += int(np.isnan(e).sum() + np.isnan(d).sum())
            n_inf += int(np.isinf(e).sum() + np.isinf(d).sum())
            n_zero += int((~e.any(axis=(1, 2, 3))).sum())
            echo_max = max(echo_max, float(e.max()))
            depth_min = min(depth_min, float(d.min()))
            depth_max = max(depth_max, float(d.max()))
            n_depth_bad += int((d < 0).sum())

        print(f"  NaN / Inf                 {n_nan} / {n_inf}")
        if n_nan or n_inf:
            failures.append(f"NaN={n_nan}, Inf={n_inf} w echo/depth")
        print(f"  probki z samych zer        {n_zero}")
        if n_zero:
            failures.append(f"{n_zero} probek `echo` zlozonych z samych zer")
        print(f"  echo max                  {echo_max:.4f}  (float16 max 65504, zapas "
              f"{65504/max(echo_max,1e-9):.0f}x)")
        if echo_max >= 65504:
            failures.append(f"echo max {echo_max} osiaga granice float16 — nastapilo obciecie")
        print(f"  depth zakres              {depth_min:.3f} - {depth_max:.3f} m")
        if n_depth_bad:
            failures.append(f"{n_depth_bad} pikseli depth < 0")
        if depth_max > 30.0:
            warnings_.append(f"depth max {depth_max:.2f} m > 30 m — nietypowe dla Repliki "
                             f"(referencja z PKL_FORMAT.md: 12.66 m)")

        # ---------------- POPRAWNOSC METODOLOGICZNA ---------------------
        print("\n--- POPRAWNOSC METODOLOGICZNA ---")
        # N_MAX jest twardym limitem CALKOWITEJ liczby renderow probki (§5 ogr. 6:
        # probki obciete moga nie osiagnac SNR 3.5 i gwarancja ich nie obejmuje),
        # wiec gwarancje sprawdzamy tam, gdzie limit NIE zadzialal.
        capped = n_total >= n_max
        guaranteed = ~capped
        below = guaranteed & (snr_final < target_snr)
        ok_all = int((snr_final >= target_snr).sum())
        print(f"  snr_final >= {target_snr}          {ok_all} / {len(snr_final)} probek "
              f"(min {snr_final.min():.3f})")
        if below.any():
            failures.append(f"{int(below.sum())} probek nieobcietych przez N_MAX ma "
                            f"snr_final < {target_snr} (min {snr_final[below].min():.3f})")

        # Probki, ktorym petla weryfikacyjna dobila do N_MAX. Gwarancja jakosci
        # ich nie obejmuje (§5 ograniczenie 6), wiec musza byc WYMIENIONE, a nie
        # tylko wylaczone z mianownika — inaczej raport milczaco gubi przypadki,
        # ktore maja trafic do pracy.
        if capped.any():
            cap_ok = capped & (snr_final >= target_snr)
            cap_bad = capped & (snr_final < target_snr)
            print(f"  przy limicie N_MAX={n_max}       {int(capped.sum())} probek "
                  f"({int(cap_ok.sum())} osiagnelo prog mimo limitu, "
                  f"{int(cap_bad.sum())} nie)")
            for i in np.flatnonzero(capped)[:20]:
                print(f"      lok {loc_id[i]:<5} kat {angle[i]:<4} n_planned={n_planned[i]:<3} "
                      f"+{n_extra[i]:<3} = {n_total[i]:<3} snr_probe={snr_probe[i]:.3f} "
                      f"snr_final={snr_final[i]:.3f}")
            if cap_bad.any():
                warnings_.append(
                    f"{int(cap_bad.sum())} probek dobilo do N_MAX={n_max} i NIE osiagnelo "
                    f"SNR {target_snr} — kategoria z §5 ograniczenie 6, do wypunktowania "
                    f"w pracy (liczba, udzial, rozklad snr_final i n_raw)")
        else:
            print(f"  przy limicie N_MAX={n_max}       brak")

        # Regula spojnosci §3.4.1: n_rendered_extra > 0 <=> snr_probe < TARGET_SNR.
        # Wyjatek strukturalny: probka, ktora juz na starcie ma n_planned == N_MAX,
        # nie MOZE dostac dodatkowych renderow — tam implikacja "w prawo" nie obowiazuje.
        at_cap_from_start = n_planned >= n_max
        chk = ~at_cap_from_start
        viol = ((n_extra > 0) != (snr_probe < target_snr)) & chk
        print(f"  regula (extra>0 <=> snr_probe<{target_snr})  "
              f"{'OK' if not viol.any() else f'{int(viol.sum())} NARUSZEN'} "
              f"({int(chk.sum())} sprawdzonych, {int(at_cap_from_start.sum())} pominietych na N_MAX)")
        if viol.any():
            k = np.flatnonzero(viol)[:5]
            failures.append(
                "naruszenie reguly spojnosci petli weryfikacyjnej dla "
                f"{int(viol.sum())} probek, np. " +
                "; ".join(f"lok {loc_id[i]} kat {angle[i]}: snr_probe={snr_probe[i]:.3f}, "
                          f"extra={n_extra[i]}" for i in k))

        bad_total = n_total != (n_planned + n_extra)
        print(f"  n_total == n_planned + extra  "
              f"{'OK' if not bad_total.any() else f'{int(bad_total.sum())} NARUSZEN'}")
        if bad_total.any():
            failures.append(f"{int(bad_total.sum())} probek ma n_total != n_planned + n_rendered_extra")

        nonuniform = []
        for lid in locs_present:
            vals = np.unique(n_planned[loc_id == lid])
            if vals.size != 1:
                nonuniform.append((int(lid), vals.tolist()))
        print(f"  n_planned jednolite w lokalizacji  "
              f"{'OK' if not nonuniform else f'{len(nonuniform)} NIEJEDNOLITYCH'}")
        if nonuniform:
            failures.append(f"n_planned rozni sie miedzy orientacjami w {len(nonuniform)} "
                            f"lokalizacjach: {nonuniform[:5]} — zlamana zasada z §3.3 pkt 2")

        for kind in ("min", "max"):
            sel = clamped == kind
            if not sel.any():
                print(f"  clamp '{kind}'              brak")
                continue
            locs = sorted(set(int(x) for x in loc_id[sel]))
            print(f"  clamp '{kind}'              {len(locs)} lokalizacji, "
                  f"{int(sel.sum())} probek")
            for lid in locs[:20]:
                m = sel & (loc_id == lid)
                print(f"      lok {lid:<5} n_raw={n_raw[m][0]:<4} n_planned={n_planned[m][0]:<3} "
                      f"snr_final {snr_final[m].min():.2f}-{snr_final[m].max():.2f}")
            if kind == "max":
                bad = sel & (snr_final < target_snr)
                if bad.any():
                    warnings_.append(
                        f"{int(bad.sum())} probek z clamped=='max' ma snr_final < {target_snr} "
                        f"— to kategoria z §5 ograniczenie 6, do wypunktowania w pracy")

        # ---------------- STATYSTYKI ------------------------------------
        print("\n--- STATYSTYKI DO PRACY ---")
        print(f"  histogram n_planned  (mediana {int(np.median(n_planned))}, "
              f"srednia {n_planned.mean():.2f})")
        for line in _hist(n_planned):
            print(line)
        print(f"\n  histogram snr_probe  (mediana {np.median(snr_probe):.3f}, "
              f"min {snr_probe.min():.3f}, max {snr_probe.max():.3f}, "
              f"ponizej progu {int((snr_probe < target_snr).sum())})")
        for line in _hist(snr_probe, bins=12):
            print(line)
        print(f"\n  histogram snr_final  (mediana {np.median(snr_final):.3f}, "
              f"min {snr_final.min():.3f}, max {snr_final.max():.3f})")
        for line in _hist(snr_final, bins=12):
            print(line)
        # sigma_1_probe jest stale w obrebie lokalizacji (decyzja zapada raz na
        # lokalizacje, §3.3), wiec histogram po probkach powielilby kazda wartosc
        # 36 razy — bierzemy po jednej na lokalizacje.
        sigma_per_loc = np.array([sigma_1[loc_id == lid][0] for lid in locs_present])
        print(f"\n  sigma_1_probe (na lokalizacje)  mediana {np.median(sigma_per_loc):.5f}, "
              f"zakres {sigma_per_loc.min():.5f}-{sigma_per_loc.max():.5f}")
        for line in _hist(sigma_per_loc, bins=10):
            print(line)
        print(f"\n  dorenderowane         {int((n_extra > 0).sum())} probek "
              f"({100.0*(n_extra > 0).mean():.2f} %), lacznie {int(n_extra.sum())} renderow")
        print(f"  renderow w pliku      {int(n_total.sum())} "
              f"(sonda liczona osobno: {locs_present.size} x {int(f.attrs.get('n_probe', N_PROBE))})")
        spr = float(f.attrs.get("seconds_per_render", 0.0))
        print(f"  tempo zmierzone       {spr:.4f} s/render "
              f"(spec {S_PER_RENDER_SPEC}, {100*(spr/S_PER_RENDER_SPEC - 1):+.1f} %)")
        print(f"  rozmiar pliku         {path.stat().st_size/2**20:.1f} MiB "
              f"({path.stat().st_size/max(n_written,1)/1024:.1f} KiB/probke, "
              f"bez kompresji {BYTES_PER_SAMPLE/1024:.1f})")

        # ---------------- INSPEKCJA WZROKOWA ----------------------------
        if n_plots > 0 and n_written > 0:
            out_dir = OUT_ROOT / "verify"
            out_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed)
            pick = rng.choice(idx, size=min(n_plots, idx.size), replace=False)
            made = _plot_samples(f, pick, out_dir, scene)
            print("\n--- INSPEKCJA WZROKOWA ---")
            for p in made:
                print(f"  {p}")
    finally:
        f.close()

    print(f"\n{'=' * 78}")
    if warnings_:
        print("  OSTRZEZENIA:")
        for w in warnings_:
            print(f"    - {w}")
    if failures:
        print(f"  WERDYKT: FAIL ({len(failures)})")
        for x in failures:
            print(f"    - {x}")
    else:
        print("  WERDYKT: PASS")
    print("=" * 78)
    return 1 if failures else 0


def _plot_samples(f, indices, out_dir, scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for i in indices:
        echo = f["echo"][i].astype(np.float32)
        rgb = f["rgb"][i]
        depth = f["depth"][i]
        lid = int(f["location_id"][i])
        ang = int(f["angle_deg"][i])
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        for ch in (0, 1):
            im = axes[ch].imshow(echo[ch], origin="lower", aspect="auto", cmap="magma")
            axes[ch].set_title(f"spektrogram, kanal {ch} ({'L' if ch == 0 else 'P'})")
            axes[ch].set_xlabel("ramka")
            axes[ch].set_ylabel("prazek czestotliwosci")
            fig.colorbar(im, ax=axes[ch], fraction=0.046)
        axes[2].imshow(rgb[..., :3])
        axes[2].set_title("RGB")
        axes[2].axis("off")
        im = axes[3].imshow(depth, cmap="viridis")
        axes[3].set_title("depth [m]")
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046)
        fig.suptitle(f"{scene}  lok={lid}  kat={ang} st.  "
                     f"N={int(f['n_total'][i])}  snr_final={float(f['snr_final'][i]):.2f}")
        fig.tight_layout()
        p = out_dir / f"{scene}_loc{lid}_ang{ang}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)
    return made


# ---------------------------------------------------------------------------
# --status (bez GPU)
# ---------------------------------------------------------------------------
def _writer_alive(scene):
    """Czy proces, ktory zapisal sidecar tej sceny, wciaz zyje.

    os.kill(pid, 0) nie wysyla sygnalu — tylko sprawdza istnienie procesu
    i uprawnienia. Ryzyko recyklingu PID-u jest tu bez znaczenia: najgorszy
    skutek to jeden zle opisany wiersz tabelki.
    """
    sp = scene_progress(scene)
    if not sp.exists():
        return False
    try:
        pid = int(json.loads(sp.read_text(encoding="utf-8")).get("pid", -1))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def status():
    print(f"\n{'=' * 96}")
    print(f"  STAN DATASETU — {OUT_ROOT}")
    print(f"  kolejnosc wg GENERATOR_PARAMS.md §4.2 (walidacyjna, held-out, po jednej z rodziny, reszta)")
    print("=" * 96)
    print(f"  {'#':<3}{'scena':<18}{'':<3}{'stan':<12}{'probki':>14}{'%':>7}{'rozmiar':>11}"
          f"{'s/render':>10}{'czas':>10}")
    print("  " + "-" * 92)

    tot_done = tot_expected = 0
    tot_bytes = 0
    tot_seconds = 0.0
    rates = []
    for i, scene in enumerate(SCENE_ORDER, start=1):
        path = scene_h5(scene)
        tag = "H" if scene in HELD_OUT else " "
        n_written = n_expected = 0
        size = 0
        spr = 0.0
        secs = 0.0
        state = "brak"

        if path.exists():
            size = path.stat().st_size
            info = None
            try:
                f = _open_readonly(path)
                try:
                    w = f["written"][:]
                    n_written = int(w.sum())
                    n_expected = int(f.attrs.get("n_samples_expected", w.size))
                    spr = float(f.attrs.get("seconds_per_render", 0.0))
                    secs = sum(r.get("wall_seconds", 0.0)
                               for r in json.loads(f.attrs.get("runs", "[]")))
                finally:
                    f.close()
                info = "h5"
            except Exception:
                # Plik trzymany do zapisu przez inny proces — sidecar JSON.
                sp = scene_progress(scene)
                if sp.exists():
                    d = json.loads(sp.read_text(encoding="utf-8"))
                    n_written = int(d.get("n_written", 0))
                    n_expected = int(d.get("n_samples_expected", 0))
                    spr = float(d.get("seconds_per_render", 0.0) or 0.0)
                    info = "sidecar"
            if info is None:
                state = "NIECZYTELNY"
            elif n_expected and n_written >= n_expected:
                state = "gotowa"
            else:
                # Zywotnosc procesu, a nie mtime sidecara, jest tu sygnalem
                # rozstrzygajacym: jedna lokalizacja przy N=29 trwa ~5 min, wiec
                # kazdy prog oparty na czasie mylilby "wolno" z "przerwane".
                state = "przerwana" if n_written else "pusta"
                if _writer_alive(scene):
                    state = "W TOKU"
        else:
            try:
                n_expected = len(load_scene_locations(scene)[0]) * N_ANGLES
            except Exception:
                n_expected = 0

        tot_done += n_written
        tot_expected += n_expected
        tot_bytes += size
        tot_seconds += secs
        if spr > 0:
            rates.append(spr)
        pct = 100.0 * n_written / n_expected if n_expected else 0.0
        print(f"  {i:<3}{scene:<18}{tag:<3}{state:<12}{f'{n_written}/{n_expected}':>14}"
              f"{pct:>6.1f}%{size/2**30:>10.2f}G{spr:>10.4f}{_fmt_hms(secs):>10}")

    print("  " + "-" * 92)
    pct = 100.0 * tot_done / tot_expected if tot_expected else 0.0
    print(f"  {'':<3}{'RAZEM':<18}{'':<3}{'':<12}{f'{tot_done}/{tot_expected}':>14}"
          f"{pct:>6.1f}%{tot_bytes/2**30:>10.2f}G"
          f"{(float(np.mean(rates)) if rates else 0.0):>10.4f}{_fmt_hms(tot_seconds):>10}")
    if rates and tot_expected > tot_done:
        rate = float(np.mean(rates))
        remaining_renders = (tot_expected - tot_done) * MEAN_N_SPEC
        print(f"\n  pozostalo ~{remaining_renders:,.0f} renderow".replace(",", " ") +
              f" x {rate:.4f} s = {_fmt_hms(remaining_renders * rate)} "
              f"({remaining_renders * rate / 3600:.1f} h)")
    print(f"  H = scena held-out ({', '.join(HELD_OUT)})")
    print("=" * 96)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", help="nazwa sceny Replica do wygenerowania (jedna, w jednym procesie)")
    p.add_argument("--limit", type=int, default=None,
                   help="tylko pierwsze N lokalizacji (smoke test); plik pozostanie niekompletny")
    p.add_argument("--resume", action="store_true",
                   help="dopisz brakujace probki do istniejacego pliku")
    p.add_argument("--force", action="store_true",
                   help="nadpisz istniejacy plik od zera (NISZCZY dane)")
    p.add_argument("--dry-run", action="store_true",
                   help="bez GPU: liczba lokalizacji, probek, szacowany czas i rozmiar")
    p.add_argument("--verify", metavar="SCENA", help="pelna walidacja gotowego pliku (bez GPU)")
    p.add_argument("--status", action="store_true",
                   help="tabelka wszystkich 18 scen (bez GPU)")
    p.add_argument("--flush-every", type=int, default=36,
                   help="flush pliku HDF5 co N probek (domyslnie 36 = jedna lokalizacja)")
    p.add_argument("--plots", type=int, default=3,
                   help="ile losowych probek zapisac jako PNG w --verify (0 = zadnych)")
    args = p.parse_args()

    modes = [bool(args.status), bool(args.verify), bool(args.dry_run), bool(args.scene)]
    if sum(modes) == 0:
        p.print_help()
        return 2
    if args.status:
        return status()
    if args.verify:
        if args.verify not in SCENE_ORDER:
            raise SystemExit(f"nieznana scena: {args.verify}\ndostepne: {', '.join(SCENE_ORDER)}")
        return verify(args.verify, n_plots=args.plots)
    if not args.scene:
        raise SystemExit("--dry-run wymaga --scene")
    if args.scene not in SCENE_ORDER:
        raise SystemExit(f"nieznana scena: {args.scene}\ndostepne: {', '.join(SCENE_ORDER)}")
    if args.dry_run:
        return dry_run(args.scene, limit=args.limit)
    return generate(args.scene, limit=args.limit, resume=args.resume, force=args.force,
                    flush_every=args.flush_every)


if __name__ == "__main__":
    sys.exit(main())
