#!/usr/bin/env python
"""BLOK 0: rozstrzygniecie, czym roznia sie warianty geometrii `main` i `patched`.

    python my-operations/ml/geometry_check.py            # calosc, pelny skan
    python my-operations/ml/geometry_check.py --only zeros
    python my-operations/ml/geometry_check.py --sample 512   # szybki przebieg

DLACZEGO TO MUSI PASC PRZED MACIERZA. Raport z 2026-08-05 (§3.3) wykazal, ze
`location_id` i `position` sa bit-identyczne miedzy wariantami -- ale nie
powiedzial nic o `rgb`, `depth` i `echo`. Od tego zalezy, czy metryki obu
wariantow sa porownywalne wprost:

  * jesli `depth` sie zmienia, to model wariantu `patched` jest oceniany na
    innych pikselach niz model wariantu `main`, bo maska `depth_gt != 0` jest
    inna. Roznica RMSE mieszalaby wtedy efekt akustyczny z efektem zmiany
    zbioru punktowanych pikseli;
  * dorobiony sufit to geometria SYNTETYCZNA, nie zmierzona. Uczenie galezi
    wizualnej przewidywania plaszczyzny, ktorej w skanie nie bylo, to uczenie
    fikcji -- a punktowanie na niej zanizalo by porownywalnosc.

Cztery pomiary, kazdy odpowiada na jedno pytanie:

  0.1 `channels`  -- czy latka w ogole zmienia rgb/depth/echo i o ile
  0.2 `sealed`    -- kontrola NEGATYWNA: 8 scen szczelnych musi dawac
                     bit-identyczne tensory w obu konfiguracjach wariantu
                     (sa serwowane z tego samego pliku -- jesli nie sa, to blad
                     w kompozycji sciezek, nie wlasciwosc danych)
  0.3 `zeros`     -- odsetek `depth == 0` per scena, main vs patched
  0.4 `energy`    -- mediana energii echa + KONTRAST KATOWY (RMSE 0 vs 90 st.),
                     osobno dla scen otwartych i szczelnych

Wiersze dopasowywane sa po kluczu `(location_id, angle_deg)`, a NIE po indeksie:
oba pliki powstaly z osobnych przebiegow generatora i zgodnosc kolejnosci
wierszy jest hipoteza do sprawdzenia, nie zalozeniem. Skrypt raportuje jawnie,
czy kolejnosc wyszla identyczna.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "ml.analysis"

from .. import paths  # noqa: E402
from ..dataset.splits import ALL_SCENES  # noqa: E402

# 10 scen, ktore mialy dziure i zostaly zalatane (GENERATOR_PARAMS.md §4.5).
PATCHED_SCENES = [
    "apartment_1", "apartment_2",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
    "office_2", "office_3",
]
# 8 scen akustycznie szczelnych -- w wariancie `patched` serwowanych z katalogu
# `main`, bo ich siatka jest w obu wariantach TYM SAMYM plikiem.
SEALED_SCENES = [s for s in ALL_SCENES if s not in PATCHED_SCENES]

# Sceny "otwarte" (z dziura w suficie) w wariancie `main` -- podzbior scen
# zalatanych, uzywany do stratyfikacji energii echa.
OPEN_SCENES = [s for s in PATCHED_SCENES if s.startswith("frl_apartment")]

# Sceny, dla ktorych pelny skan jest wymagany niezaleznie od --sample:
# obie sa w zbiorze odlozonym, wiec kazda roznica w nich dotyka wprost metryk.
FORCE_FULL_SCAN = ("apartment_2", "frl_apartment_5")

BLOCK = 512  # wierszy na odczyt; dobrane tak, zeby blok depth+echo+rgb mial ~200 MB

# Granica "pozno" w spektrogramie. Hop STFT to 16 probek przy 44,1 kHz, czyli
# 0,363 ms na ramke; 166 ramek to 60 ms okna echa. Zrodlo i sluchacz sa
# WSPOLLOKOWANE, wiec sciezka bezpospredna przychodzi w t = 0 i sama nie niesie
# zadnej informacji o pomieszczeniu -- a dominuje energie calkowita. Pierwsze
# odbicie od podlogi/sufitu przy wysokosci 1,25 m to 2*1,25/343 = 7,3 ms
# (ramka 20). Ramka 30 (10,9 ms) leży bezpiecznie za nim, wiec energia liczona
# od niej w gore jest energia POGLOSU, a nie sygnalu wejsciowego -- i to ona
# odpowiada na pytanie "ile energii ucieka przez brak sufitu".
LATE_FRAME_START = 30
FRAME_MS = 16 / 44100 * 1000


def _out_dir() -> Path:
    return paths.ML_OUTPUTS / "geometry_check"


# ----------------------------------------------------------------- dopasowanie


def _keys(f: h5py.File) -> np.ndarray:
    """Klucz wiersza: location_id * 1000 + angle_deg. Katy sa < 360, wiec
    kodowanie jest jednoznaczne."""
    loc = f["location_id"][:].astype(np.int64)
    ang = f["angle_deg"][:].astype(np.int64)
    return loc * 1000 + ang


def match_rows(fa: h5py.File, fb: h5py.File) -> dict:
    """Zwraca permutacje wierszy b odpowiadajaca wierszom a, po kluczu.

    Jawnie sprawdza, czy zbiory kluczy sa rowne -- brak dopasowania oznacza,
    ze warianty NIE opisuja tych samych probek i zadne dalsze porownanie
    nie mialoby sensu.
    """
    ka, kb = _keys(fa), _keys(fb)
    if ka.size != kb.size:
        raise RuntimeError(f"rozna liczba wierszy: {ka.size} vs {kb.size}")
    if np.unique(ka).size != ka.size:
        raise RuntimeError("klucze (loc, kat) nie sa unikalne w wariancie main")
    order = np.argsort(kb, kind="stable")
    pos = np.searchsorted(kb[order], ka)
    bad = (pos >= kb.size) | (kb[order][np.minimum(pos, kb.size - 1)] != ka)
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())} kluczy z main nie ma odpowiednika w patched")
    perm = order[pos]
    return {
        "n_rows": int(ka.size),
        "keys_identical_as_sets": True,
        # Jesli permutacja jest identycznoscia, to kolejnosc wierszy w obu
        # plikach wyszla ta sama -- ciekawe, ale NIE zalozone przez ten skrypt.
        "row_order_identical": bool(np.array_equal(perm, np.arange(ka.size))),
        "perm": perm,
    }


def _pick_rows(n: int, sample: int | None, scene: str, seed: int = 20260810) -> np.ndarray:
    """Deterministyczny podzbior wierszy.

    Ziarno z SHA-256, a nie z `hash()`: wbudowany hash stringow jest w Pythonie
    solony per proces (PYTHONHASHSEED), wiec ta sama komenda dawalaby innej
    probki przy kazdym uruchomieniu -- a wynik ma byc odtwarzalny.
    """
    if sample is None or scene in FORCE_FULL_SCAN or sample >= n:
        return np.arange(n)
    h = hashlib.sha256(f"{seed}|{scene}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return np.sort(rng.choice(n, size=sample, replace=False))


# ------------------------------------------------------- 0.1 porownanie kanalow


class _DiffAcc:
    """Akumulator roznicy jednego kanalu. Trzyma histogram wartosci
    bezwzglednych roznicy, zeby mediane dalo sie policzyc bez pamietania
    wszystkich pikseli (przy pelnym skanie to ~10^9 wartosci)."""

    def __init__(self, n_bins: int = 4096, vmax: float = 16.0):
        self.equal = True
        self.n = 0
        self.n_diff = 0
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.bins = np.zeros(n_bins + 1, dtype=np.int64)
        self.edges = np.linspace(0.0, vmax, n_bins + 1)
        self.vmax = vmax

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        # float32 wystarcza i jest DOKLADNE: rgb to uint8, depth float32,
        # echo float16 -- kazdy z tych typow miesci sie w float32 bez straty,
        # wiec roznica jest scisla, a nie zaokraglona.
        d = np.abs(a.astype(np.float32) - b.astype(np.float32))
        self.n += d.size
        nz = d > 0
        k = int(nz.sum())
        if k:
            self.equal = False
            self.n_diff += k
            dv = d[nz]
            self.max_abs = max(self.max_abs, float(dv.max()))
            self.sum_abs += float(dv.sum())
            idx = np.clip(np.searchsorted(self.edges, dv, side="right") - 1, 0, self.bins.size - 1)
            np.add.at(self.bins, idx, 1)

    def _quantile_of_diff(self, q: float) -> float:
        """Kwantyl liczony PO PIKSELACH, KTORE SIE ZMIENILY (nie po wszystkich)."""
        if self.n_diff == 0:
            return 0.0
        c = np.cumsum(self.bins)
        i = int(np.searchsorted(c, q * self.n_diff))
        i = min(i, self.edges.size - 2)
        return float(0.5 * (self.edges[i] + self.edges[i + 1]))

    def result(self) -> dict:
        return {
            "identical": self.equal,
            "n_values": int(self.n),
            "n_changed": int(self.n_diff),
            "frac_changed": (self.n_diff / self.n) if self.n else 0.0,
            "max_abs_diff": self.max_abs,
            "mean_abs_diff_over_changed": (self.sum_abs / self.n_diff) if self.n_diff else 0.0,
            "median_abs_diff_over_changed": self._quantile_of_diff(0.5),
            "p95_abs_diff_over_changed": self._quantile_of_diff(0.95),
        }


def compare_channels(scene: str, sample: int | None) -> dict:
    pa = paths.DATASET_DIRS["main"] / scene / f"{scene}.h5"
    pb = paths.DATASET_DIRS["patched"] / scene / f"{scene}.h5"
    t0 = time.perf_counter()

    with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
        m = match_rows(fa, fb)
        perm = m.pop("perm")
        rows_a = _pick_rows(m["n_rows"], sample, scene)
        rows_b = perm[rows_a]

        acc = {c: _DiffAcc(vmax=(255.0 if c == "rgb" else 16.0)) for c in ("rgb", "depth", "echo")}
        # Osobno: piksele glebi, ktore z 0 zrobily sie dodatnie. To sa wprost
        # DOROBIONE SUFITY -- geometria, ktorej w skanie nie bylo.
        filled = 0        # 0 -> >0
        emptied = 0       # >0 -> 0
        zeros_a = 0
        zeros_b = 0
        n_depth_px = 0

        for s in range(0, rows_a.size, BLOCK):
            ra = rows_a[s:s + BLOCK]
            rb = rows_b[s:s + BLOCK]
            for ch in ("rgb", "depth", "echo"):
                a = fa[ch][ra]
                b = fb[ch][rb]
                acc[ch].update(a, b)
                if ch == "depth":
                    za = a == 0
                    zb = b == 0
                    filled += int((za & ~zb).sum())
                    emptied += int((~za & zb).sum())
                    zeros_a += int(za.sum())
                    zeros_b += int(zb.sum())
                    n_depth_px += a.size

    out = {
        "scene": scene,
        "full_scan": bool(sample is None or scene in FORCE_FULL_SCAN or sample >= m["n_rows"]),
        "n_rows_total": m["n_rows"],
        "n_rows_compared": int(rows_a.size),
        "keys_identical_as_sets": m["keys_identical_as_sets"],
        "row_order_identical": m["row_order_identical"],
        "channels": {c: acc[c].result() for c in ("rgb", "depth", "echo")},
        "depth_holes": {
            "n_pixels": n_depth_px,
            "zeros_main": zeros_a,
            "zeros_patched": zeros_b,
            "frac_zeros_main": zeros_a / n_depth_px if n_depth_px else 0.0,
            "frac_zeros_patched": zeros_b / n_depth_px if n_depth_px else 0.0,
            "filled_0_to_positive": filled,
            "frac_filled_0_to_positive": filled / n_depth_px if n_depth_px else 0.0,
            "emptied_positive_to_0": emptied,
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return out


# --------------------------------------------------- 0.2 kontrola scen szczelnych


def sealed_control(n_per_scene: int = 8) -> dict:
    """Sceny szczelne MUSZA dac bit-identyczne tensory w obu wariantach.

    To kontrola NEGATYWNA logiki sklejki z `paths.scene_h5()`: wariant
    `patched` to 10 scen zalatanych + 8 szczelnych czytanych z katalogu `main`.
    Jesli tu wyjdzie roznica, to znaczy, ze kompozycja sciezek podaje inny plik,
    niz sadzi -- i kazdy wynik main-vs-patched bylby zanieczyszczony.

    Porownanie idzie przez PELNY `EchoH5Dataset`, a nie przez surowy HDF5:
    testowana jest cala droga od nazwy sceny do tensora, lacznie z indeksem
    i normalizacja. `augment=False`, bo sciezka treningowa losuje wzmocnienia
    PIL i bitowa zgodnosc bylaby tam wlasnoscia RNG, nie danych.
    """
    from ..dataset.echo_h5_dataset import DatasetConfig, EchoH5Dataset

    res = {"per_mode": {}, "path_mapping": {}}
    for s in SEALED_SCENES:
        pm = paths.scene_h5(s, "main")
        pp = paths.scene_h5(s, "patched")
        res["path_mapping"][s] = {
            "main": str(pm.relative_to(paths.REPO_ROOT)),
            "patched": str(pp.relative_to(paths.REPO_ROOT)),
            "same_file": pm.resolve() == pp.resolve(),
        }

    all_ok = True
    for mode in ("train", "val", "test"):
        ds_a = EchoH5Dataset(DatasetConfig(variant="main", mode=mode,
                                           angle_subset="all", augment=False))
        ds_b = EchoH5Dataset(DatasetConfig(variant="patched", mode=mode,
                                           angle_subset="all", augment=False))
        entry = {
            "n_samples": len(ds_a),
            "n_samples_patched": len(ds_b),
            "index_aligned": bool(
                ds_a.scenes == ds_b.scenes
                and np.array_equal(ds_a.index_scene, ds_b.index_scene)
                and np.array_equal(ds_a.index_loc, ds_b.index_loc)
                and np.array_equal(ds_a.index_angle, ds_b.index_angle)
            ),
            "scenes_checked": [],
            "n_compared": 0,
            "all_identical": True,
            "mismatches": [],
        }
        rng = np.random.default_rng(7)
        for si, scene in enumerate(ds_a.scenes):
            if scene not in SEALED_SCENES:
                continue
            idx_pool = np.flatnonzero(ds_a.index_scene == si)
            if idx_pool.size == 0:
                continue
            pick = rng.choice(idx_pool, size=min(n_per_scene, idx_pool.size), replace=False)
            entry["scenes_checked"].append(scene)
            for i in pick:
                a, b = ds_a[int(i)], ds_b[int(i)]
                entry["n_compared"] += 1
                for k in ("img", "depth", "audio"):
                    if not bool((a[k] == b[k]).all()):
                        entry["all_identical"] = False
                        entry["mismatches"].append(
                            {"scene": scene, "index": int(i), "key": k,
                             "max_abs_diff": float((a[k] - b[k]).abs().max())})
        ds_a.close()
        ds_b.close()
        all_ok = all_ok and entry["all_identical"] and entry["index_aligned"]
        res["per_mode"][mode] = entry

    # Kontrola pozytywna: scena ZALATANA musi sie roznic. Bez niej test wyzej
    # przechodzilby rowniez wtedy, gdyby porownywal plik sam ze soba.
    ds_a = EchoH5Dataset(DatasetConfig(variant="main", mode="test",
                                       angle_subset="all", augment=False))
    ds_b = EchoH5Dataset(DatasetConfig(variant="patched", mode="test",
                                       angle_subset="all", augment=False))
    pos = {"scene": None, "differs": False}
    for si, scene in enumerate(ds_a.scenes):
        if scene not in PATCHED_SCENES:
            continue
        idx_pool = np.flatnonzero(ds_a.index_scene == si)
        if idx_pool.size == 0:
            continue
        i = int(idx_pool[0])
        a, b = ds_a[i], ds_b[i]
        pos = {
            "scene": scene,
            "index": i,
            "differs": any(not bool((a[k] == b[k]).all()) for k in ("img", "depth", "audio")),
            "per_key_identical": {k: bool((a[k] == b[k]).all()) for k in ("img", "depth", "audio")},
        }
        break
    ds_a.close()
    ds_b.close()
    res["positive_control"] = pos

    res["ok"] = bool(all_ok and pos.get("differs", False)
                     and all(v["same_file"] for v in res["path_mapping"].values()))
    return res


# ------------------------------------------------------------ 0.3 zera w glebi


def depth_zeros(scene: str, variant: str, sample: int | None) -> dict:
    p = paths.scene_h5(scene, variant)
    with h5py.File(p, "r") as f:
        n = f["depth"].shape[0]
        rows = _pick_rows(n, sample, scene)
        zeros = 0
        total = 0
        dmax = 0.0
        for s in range(0, rows.size, BLOCK):
            d = f["depth"][rows[s:s + BLOCK]]
            zeros += int((d == 0).sum())
            total += d.size
            dmax = max(dmax, float(d.max()))
    return {"n_rows": int(rows.size), "n_rows_total": n, "n_pixels": total,
            "n_zeros": zeros, "frac_zeros": zeros / total if total else 0.0,
            "max_depth_m": dmax, "file": str(p.relative_to(paths.REPO_ROOT))}


# ----------------------------------------------------- 0.4 energia i kontrast


def echo_stats(scene: str, variant: str, sample: int | None) -> dict:
    """Mediana energii echa (suma kwadratow magnitud) + kontrast katowy.

    ENERGIA. Spektrogram jest magnituda STFT, wiec suma kwadratow komorek jest
    (z dokladnoscia do stalej Parsevala i nakladania okien) proporcjonalna do
    energii sygnalu. Liczona per probka, raportowana jako mediana po probkach
    sceny -- mediana, bo rozklad po lokalizacjach jest skosny (lokalizacje przy
    scianie maja silne wczesne odbicie).

    KONTRAST KATOWY. RMSE miedzy spektrogramem 0 i 90 stopni TEJ SAMEJ
    lokalizacji. Hipoteza do sprawdzenia, nie zalozenia: brak sufitu usuwa
    odbicie niemal NIEZALEZNE od orientacji (przychodzace z gory), wiec moze
    PODNOSIC wzgledny kontrast katowy przy jednoczesnym obnizeniu energii.
    Raportujemy takze kontrast znormalizowany energia, bo sam RMSE spada
    trywialnie razem z glosnoscia.
    """
    p = paths.scene_h5(scene, variant)
    with h5py.File(p, "r") as f:
        n = f["echo"].shape[0]
        rows = _pick_rows(n, sample, scene)
        energies = np.empty(rows.size, dtype=np.float64)
        energies_late = np.empty(rows.size, dtype=np.float64)
        for s in range(0, rows.size, BLOCK):
            e = f["echo"][rows[s:s + BLOCK]].astype(np.float64)
            sq = e ** 2
            energies[s:s + e.shape[0]] = sq.sum(axis=(1, 2, 3))
            energies_late[s:s + e.shape[0]] = sq[:, :, :, LATE_FRAME_START:].sum(axis=(1, 2, 3))

        # Kontrast katowy: osobny, maly odczyt tylko wierszy 0 i 90 stopni.
        loc = f["location_id"][:].astype(np.int64)
        ang = f["angle_deg"][:].astype(np.int64)
        i0 = {int(l): i for i, (l, a) in enumerate(zip(loc, ang)) if a == 0}
        i90 = {int(l): i for i, (l, a) in enumerate(zip(loc, ang)) if a == 90}
        common = sorted(set(i0) & set(i90))
        rmses = np.empty(len(common), dtype=np.float64)
        rel = np.empty(len(common), dtype=np.float64)
        rmses_late = np.empty(len(common), dtype=np.float64)
        rel_late = np.empty(len(common), dtype=np.float64)
        for j, l in enumerate(common):
            a = f["echo"][i0[l]].astype(np.float64)
            b = f["echo"][i90[l]].astype(np.float64)
            rmses[j] = np.sqrt(((a - b) ** 2).mean())
            denom = np.sqrt((a ** 2).mean()) + np.sqrt((b ** 2).mean())
            rel[j] = 2.0 * rmses[j] / denom if denom > 0 else np.nan
            al, bl = a[:, :, LATE_FRAME_START:], b[:, :, LATE_FRAME_START:]
            rmses_late[j] = np.sqrt(((al - bl) ** 2).mean())
            dl = np.sqrt((al ** 2).mean()) + np.sqrt((bl ** 2).mean())
            rel_late[j] = 2.0 * rmses_late[j] / dl if dl > 0 else np.nan

    return {
        "variant": variant,
        "n_rows": int(rows.size),
        "n_rows_total": n,
        "late_frame_start": LATE_FRAME_START,
        "late_starts_at_ms": round(LATE_FRAME_START * FRAME_MS, 2),
        "energy_median": float(np.median(energies)),
        "energy_mean": float(energies.mean()),
        "energy_p10": float(np.percentile(energies, 10)),
        "energy_p90": float(np.percentile(energies, 90)),
        "energy_late_median": float(np.median(energies_late)),
        "energy_late_mean": float(energies_late.mean()),
        "late_fraction_of_total_median": float(np.median(energies_late / energies)),
        "n_locations_contrast": len(common),
        "angular_contrast_rmse_median": float(np.median(rmses)) if len(common) else float("nan"),
        "angular_contrast_rmse_mean": float(rmses.mean()) if len(common) else float("nan"),
        "angular_contrast_relative_median": float(np.nanmedian(rel)) if len(common) else float("nan"),
        "angular_contrast_late_rmse_median": float(np.median(rmses_late)) if len(common) else float("nan"),
        "angular_contrast_late_relative_median": float(np.nanmedian(rel_late)) if len(common) else float("nan"),
    }


# ------------------------------------------------------------------- sterowanie


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("all", "channels", "sealed", "zeros", "energy"),
                    default="all")
    ap.add_argument("--sample", type=int, default=None,
                    help=f"wierszy na scene (domyslnie: pelny skan). {FORCE_FULL_SCAN} "
                         "zawsze skanowane w calosci -- sa w zbiorze odlozonym")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out_path = args.out or (_out_dir() / "geometry_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "script": "geometry_check.py",
        "version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rows_per_scene": args.sample,
        "patched_scenes": PATCHED_SCENES,
        "sealed_scenes": SEALED_SCENES,
        "open_scenes_main": OPEN_SCENES,
        "force_full_scan": list(FORCE_FULL_SCAN),
    }
    t_all = time.perf_counter()

    if args.only in ("all", "channels"):
        print("=" * 74)
        print("0.1  Czy latka zmienia kanaly danych (rgb / depth / echo)")
        print("=" * 74)
        ch: dict = {}
        for s in PATCHED_SCENES:
            r = compare_channels(s, args.sample)
            ch[s] = r
            c = r["channels"]
            print(f"  {s:18s} wierszy {r['n_rows_compared']:5d}/{r['n_rows_total']:5d} "
                  f"{'PELNY' if r['full_scan'] else 'probka'}  kolejnosc={'ta sama' if r['row_order_identical'] else 'INNA'}")
            for name in ("rgb", "depth", "echo"):
                x = c[name]
                if x["identical"]:
                    print(f"      {name:6s} BIT-IDENTYCZNY")
                else:
                    print(f"      {name:6s} rozny: {x['frac_changed']*100:7.3f} % wartosci, "
                          f"max {x['max_abs_diff']:.4f}, mediana(zm.) {x['median_abs_diff_over_changed']:.4f}")
            h = r["depth_holes"]
            print(f"      depth zera: main {h['frac_zeros_main']*100:6.3f} % -> "
                  f"patched {h['frac_zeros_patched']*100:6.3f} %   "
                  f"dorobione (0->+) {h['frac_filled_0_to_positive']*100:6.3f} %  "
                  f"usuniete (+->0) {h['emptied_positive_to_0']}")
        payload["channels"] = ch

    if args.only in ("all", "sealed"):
        print("\n" + "=" * 74)
        print("0.2  Kontrola negatywna: sceny szczelne w obu wariantach")
        print("=" * 74)
        sc = sealed_control()
        payload["sealed_control"] = sc
        for mode, e in sc["per_mode"].items():
            print(f"  {mode:6s} indeks zgodny={e['index_aligned']}  porownano {e['n_compared']:3d} probek "
                  f"z {len(e['scenes_checked'])} scen  -> "
                  f"{'BIT-IDENTYCZNE' if e['all_identical'] else 'ROZNE (BLAD!)'}")
        pc = sc["positive_control"]
        print(f"  kontrola pozytywna ({pc.get('scene')}): rozne = {pc.get('differs')}")
        print(f"  WYNIK: {'OK' if sc['ok'] else 'BLAD'}")

    if args.only in ("all", "zeros"):
        print("\n" + "=" * 74)
        print("0.3  Odsetek depth == 0 per scena, main vs patched")
        print("=" * 74)
        z: dict = {}
        print(f"  {'scena':18s} {'main %':>9s} {'patched %':>10s} {'roznica':>9s}")
        for s in ALL_SCENES:
            zm = depth_zeros(s, "main", args.sample)
            entry = {"main": zm}
            if s in PATCHED_SCENES:
                zp = depth_zeros(s, "patched", args.sample)
                entry["patched"] = zp
                entry["delta_frac"] = zp["frac_zeros"] - zm["frac_zeros"]
                print(f"  {s:18s} {zm['frac_zeros']*100:9.4f} {zp['frac_zeros']*100:10.4f} "
                      f"{entry['delta_frac']*100:+9.4f}")
            else:
                entry["patched"] = None
                entry["delta_frac"] = 0.0
                print(f"  {s:18s} {zm['frac_zeros']*100:9.4f} {'(ten sam plik)':>10s} {0.0:+9.4f}")
            z[s] = entry
        payload["depth_zeros"] = z

    if args.only in ("all", "energy"):
        print("\n" + "=" * 74)
        print("0.4  Energia echa i kontrast katowy")
        print("=" * 74)
        en: dict = {}
        print(f"  energia POZNA liczona od ramki {LATE_FRAME_START} "
              f"({LATE_FRAME_START * FRAME_MS:.1f} ms) -- za sciezka bezposrednia i "
              f"pierwszym odbiciem podloga/sufit")
        print(f"  {'scena':18s} {'wariant':8s} {'E_calk':>11s} {'E_pozna':>11s} "
              f"{'RMSE(0,90)':>11s} {'wzgl.':>8s} {'wzgl.pozn':>10s}")
        for s in ALL_SCENES:
            e_main = echo_stats(s, "main", args.sample)
            entry = {"main": e_main}

            def _row(label, e, extra=""):
                print(f"  {label:18s} {e['variant']:8s} {e['energy_median']:11.1f} "
                      f"{e['energy_late_median']:11.2f} "
                      f"{e['angular_contrast_rmse_median']:11.4f} "
                      f"{e['angular_contrast_relative_median']:8.4f} "
                      f"{e['angular_contrast_late_relative_median']:10.4f}{extra}")

            _row(s, e_main)
            if s in PATCHED_SCENES:
                e_pat = echo_stats(s, "patched", args.sample)
                entry["patched"] = e_pat
                entry["energy_ratio_patched_over_main"] = (
                    e_pat["energy_median"] / e_main["energy_median"]
                    if e_main["energy_median"] else float("nan"))
                entry["energy_late_ratio_patched_over_main"] = (
                    e_pat["energy_late_median"] / e_main["energy_late_median"]
                    if e_main["energy_late_median"] else float("nan"))
                _row("", e_pat,
                     f"   (E x{entry['energy_ratio_patched_over_main']:.3f}, "
                     f"E_pozna x{entry['energy_late_ratio_patched_over_main']:.3f})")
            else:
                entry["patched"] = None
            en[s] = entry
        payload["echo_stats"] = en
        payload["echo_groups"] = _summarize_energy(en)
        g = payload["echo_groups"]
        print("\n  grupy (mediana po scenach):")
        for k, v in g.items():
            if isinstance(v, dict) and "energy_median_of_scene_medians" in v:
                print(f"    {k:24s} E={v['energy_median_of_scene_medians']:10.1f}  "
                      f"E_pozna={v['energy_late_median_of_scene_medians']:8.2f}  "
                      f"RMSE(0,90)={v['angular_contrast_median']:.4f}  "
                      f"wzgl.={v['angular_contrast_relative_median']:.4f}  "
                      f"wzgl.pozn={v['angular_contrast_late_relative_median']:.4f}  "
                      f"(n={v['n_scenes']})")
        for k in ("open_vs_sealed", "main_vs_patched"):
            if k in g:
                print(f"\n  {k}:")
                for kk, vv in g[k].items():
                    print(f"    {kk:52s} {vv:+.4f}" if isinstance(vv, float) else f"    {kk}: {vv}")

    if args.only != "all" and out_path.exists():
        # Przebieg czesciowy DOPELNIA plik, a nie kasuje sekcje, ktorych nie
        # liczyl -- inaczej `--only energy` cicho wyrzucilby wynik `--only channels`.
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        merged = {**prev, **payload}
        merged["reran_sections"] = sorted(set(prev.get("reran_sections", [])) | {args.only})
        payload = merged

    payload["total_seconds"] = round(time.perf_counter() - t_all, 1)
    payload["verdict"] = _verdict(payload)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")

    print("\n" + "=" * 74)
    for line in payload["verdict"]["lines"]:
        print("  " + line)
    print("=" * 74)
    print(f"zapisano: {out_path}   ({payload['total_seconds']} s)")
    return 0


def _summarize_energy(en: dict) -> dict:
    def agg(scenes: list[str], variant: str) -> dict:
        vals = [en[s][variant] for s in scenes if en[s].get(variant)]
        if not vals:
            return {}
        return {
            "n_scenes": len(vals),
            "scenes": scenes,
            "energy_median_of_scene_medians": float(np.median([v["energy_median"] for v in vals])),
            "energy_late_median_of_scene_medians": float(
                np.median([v["energy_late_median"] for v in vals])),
            "late_fraction_of_total_median": float(
                np.median([v["late_fraction_of_total_median"] for v in vals])),
            "angular_contrast_median": float(np.median([v["angular_contrast_rmse_median"] for v in vals])),
            "angular_contrast_relative_median": float(
                np.median([v["angular_contrast_relative_median"] for v in vals])),
            "angular_contrast_late_relative_median": float(
                np.median([v["angular_contrast_late_relative_median"] for v in vals])),
        }

    def cmp(a: dict, b: dict, label_a: str, label_b: str) -> dict:
        """Iloraz b/a we wszystkich wielkosciach. Nazwy jawnie mowia, co przez co."""
        k_e = "energy_median_of_scene_medians"
        k_l = "energy_late_median_of_scene_medians"
        k_c = "angular_contrast_median"
        k_r = "angular_contrast_relative_median"
        k_rl = "angular_contrast_late_relative_median"
        return {
            f"energy_ratio_{label_b}_over_{label_a}": b[k_e] / a[k_e],
            f"energy_change_pct_{label_b}_vs_{label_a}": 100.0 * (b[k_e] / a[k_e] - 1),
            f"LATE_energy_ratio_{label_b}_over_{label_a}": b[k_l] / a[k_l],
            f"LATE_energy_change_pct_{label_b}_vs_{label_a}": 100.0 * (b[k_l] / a[k_l] - 1),
            f"angular_contrast_ratio_{label_b}_over_{label_a}": b[k_c] / a[k_c],
            f"relative_contrast_ratio_{label_b}_over_{label_a}": b[k_r] / a[k_r],
            f"LATE_relative_contrast_ratio_{label_b}_over_{label_a}": b[k_rl] / a[k_rl],
        }

    sealed = [s for s in SEALED_SCENES if s in en]
    out = {
        "main_open_frl": agg(OPEN_SCENES, "main"),
        "main_sealed": agg(sealed, "main"),
        "main_patched_scenes": agg(PATCHED_SCENES, "main"),
        "patched_patched_scenes": agg(PATCHED_SCENES, "patched"),
    }
    o, s = out["main_open_frl"], out["main_sealed"]
    if o and s:
        out["open_vs_sealed"] = cmp(s, o, "sealed", "open")
    m, p = out["main_patched_scenes"], out["patched_patched_scenes"]
    if m and p:
        out["main_vs_patched"] = cmp(m, p, "main", "patched")
    return out


def _verdict(payload: dict) -> dict:
    """Decyzja 0.5: czy metryki wymagaja maski przeciecia."""
    lines: list[str] = []
    depth_changes = None
    rgb_changes = None
    echo_changes = None
    if "channels" in payload:
        ch = payload["channels"]
        depth_changes = any(not v["channels"]["depth"]["identical"] for v in ch.values())
        rgb_changes = any(not v["channels"]["rgb"]["identical"] for v in ch.values())
        echo_changes = any(not v["channels"]["echo"]["identical"] for v in ch.values())
        lines.append(f"rgb zmieniony: {rgb_changes};  depth zmieniony: {depth_changes};  "
                     f"echo zmienione: {echo_changes}")
        if depth_changes:
            lines.append("DECYZJA: metryki main-vs-patched licz na MASCE PRZECIECIA "
                         "(piksele wazne w OBU wariantach = oryginalna maska `main`).")
            lines.append("POWOD: dorobiony sufit to geometria syntetyczna, nie zmierzona; "
                         "punktowanie na niej nie jest porownywalne miedzy wariantami.")
        else:
            lines.append("DECYZJA: depth identyczny -> maska przeciecia zbedna, "
                         "warianty roznia sie wylacznie akustyka (przypadek korzystny).")
        lines.append("Wielkosc porownywana miedzy wariantami to ZAWSZE Delta = RMSE(A) - RMSE(B), "
                     "nigdy surowe RMSE.")
    if "sealed_control" in payload:
        lines.append(f"kontrola scen szczelnych: {'OK' if payload['sealed_control']['ok'] else 'BLAD'}")
    return {
        "depth_changed": depth_changes,
        "rgb_changed": rgb_changes,
        "echo_changed": echo_changes,
        "needs_intersection_mask": depth_changes,
        "lines": lines,
    }


if __name__ == "__main__":
    raise SystemExit(main())
