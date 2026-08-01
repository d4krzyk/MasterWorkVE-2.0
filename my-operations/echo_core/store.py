"""Magazyn HDF5: uklad pol probki i lokalizacji, atrybuty reprodukowalnosci,
klasa DatasetStore obslugujaca zapis, wznawianie i sidecar postepu.
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .params import (ANGLES_DEG, AVERAGING_DOMAIN, CAMERA_HFOV, CAMERA_RESOLUTION,
                     INDIRECT_RAY_COUNT, N_ANGLES, N_MAX, N_MIN, N_PROBE, SCRIPT_VERSION,
                     SENSOR_HEIGHT, SIGNAL_10DEG, TARGET_SNR, THREAD_COUNT, WARMUP_DISCARD)
from .paths import (CHIRP_PATH, LOCATIONS_PKL, MATERIAL_CONFIG, REPO_ROOT, SCRIPT_PATH,
                    SPEC_DOC, scene_mesh, scene_progress)
from . import paths

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
        # Wariant datasetu zapisany JAWNIE, a nie tylko domyslnie przez sciezke
        # w `scene_id`: pliki obu wariantow maja te same nazwy i te same wymiary,
        # wiec bez tego atrybutu po skopiowaniu na maszyne treningowa nie da sie
        # ich rozroznic. `main` = geometria oryginalna, `patched` = domknieta.
        "variant": paths.VARIANT,
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
        "warmup_discard": WARMUP_DISCARD,
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

