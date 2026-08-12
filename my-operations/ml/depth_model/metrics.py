"""Metryki ewaluacji -- standardowe, STRATYFIKOWANE i per scena.

Standardowe (RMSE, REL, log10, delta<1.25^n) sa policzone dokladnie tak, jak w
`util/util.py` Paridy, zeby liczby dalo sie zestawic z opublikowanymi. Powtorzone
tu, a nie zaimportowane, tylko dlatego, ze tamta wersja liczy na CPU per probka
w numpy -- przy 6588 probkach walidacji x 40 000/validation_freq wywolan to
zauwazalny koszt. Zgodnosc obu implementacji sprawdza `test_matches_parida()`.

DLACZEGO STRATYFIKACJA. Teza pracy mowi, ze gestsze probkowanie katowe pomaga
przy KRAWEDZIACH i NAROZNIKACH -- tam, gdzie echo niesie informacje, ktorej
pojedynczy obraz nie ma. Piksele nieciaglosci glebi to jednak kilka procent
kadru; usredniony po calym obrazie RMSE rozcienczy poprawe na krawedziach
plaskimi scianami, ktore i tak sa latwe. Efekt rzedu kilku procent na 5 % pikseli
znika w trzecim miejscu po przecinku globalnego RMSE. Bez tej metryki mozna
przegapic wlasny wynik -- i to jest jedyny powod, dla ktorego tu jest.

Definicja maski krawedzi: piksel nalezy do "nieciaglosci", jesli maksimum
modulu gradientu glebi (roznice do czterech sasiadow) przekracza prog w METRACH
NA PIKSEL. Prog bezwzgledny, nie wzgledny, bo blad predykcji tez jest raportowany
bezwzglednie (RMSE w metrach) -- mieszanie skal utrudnialoby interpretacje.
Domyslne 0.10 m/px przy 128x128 i 90 stopni HFOV odpowiada scianie odchylonej o
kilkadziesiat stopni; wartosc jest parametrem, a nie stala, i raportujemy udzial
pikseli, ktory wpada do maski, zeby dalo sie ocenic czulosc na jej wybor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# Prog gradientu glebi [m / piksel] dzielacy piksele na "krawedziowe" i "gladkie".
EDGE_GRAD_THRESHOLD_M = 0.10

METRIC_NAMES = ("RMSE", "ABS_REL", "LOG10", "MAE", "DELTA1", "DELTA2", "DELTA3")

# Warstwy, na ktorych liczone sa te same metryki: caly kadr, piksele krawedziowe
# i piksele gladkie. Nazwy sa kluczami w tabeli per probka.
STRATA = ("all", "edge", "smooth")


def depth_edge_mask(depth: torch.Tensor, threshold: float = EDGE_GRAD_THRESHOLD_M) -> torch.Tensor:
    """Maska nieciaglosci glebi. depth: (B,1,H,W) w metrach -> bool (B,1,H,W).

    Gradient liczony tylko miedzy pikselami WAZNYMI (glebia > 0). Piksel
    sasiadujacy z dziura w prawdzie (glebia = 0) mialby gradient rowny wlasnej
    glebi, czyli kilka metrow, i trafilby do krawedzi bez zadnego powodu
    geometrycznego -- to zanieczyscilo by wlasnie te metryke, ktora ma byc
    najczulsza.
    """
    valid = depth > 0
    d = depth
    grad = torch.zeros_like(d)

    def acc(a_slice, b_slice, out_slice):
        va = valid[a_slice] & valid[b_slice]
        diff = torch.abs(d[a_slice] - d[b_slice]) * va
        grad[out_slice] = torch.maximum(grad[out_slice], diff)

    # Cztery kierunki; kazda roznica aktualizuje OBA piksele, ktorych dotyczy,
    # zeby krawedz miala grubosc 2 px i nie zalezala od kierunku skanowania.
    s = (slice(None), slice(None))
    acc((*s, slice(1, None), slice(None)), (*s, slice(0, -1), slice(None)),
        (*s, slice(1, None), slice(None)))
    acc((*s, slice(0, -1), slice(None)), (*s, slice(1, None), slice(None)),
        (*s, slice(0, -1), slice(None)))
    acc((*s, slice(None), slice(1, None)), (*s, slice(None), slice(0, -1)),
        (*s, slice(None), slice(1, None)))
    acc((*s, slice(None), slice(0, -1)), (*s, slice(None), slice(1, None)),
        (*s, slice(None), slice(0, -1)))

    return (grad > threshold) & valid


@dataclass
class MetricAccumulator:
    """Akumuluje sumy, nie srednie z wsadow.

    DLACZEGO SUMY. Srednia ze srednich wsadowych rowna sie sredniej globalnej
    tylko przy rownych wsadach; ostatni wsad epoki jest krotszy, a przy podziale
    per scena wsady sa jawnie nierowne. `train.py` Paridy usrednia po probkach
    (`np.array(errors).mean(0)`), co jest poprawne dla srednich, ale RMSE
    usredniony po probkach to NIE jest RMSE calego zbioru -- to pierwiastek
    sredniej z pierwiastkow. Tu liczymy oba i raportujemy oba, zeby zestawienie
    z Parida bylo mozliwe i zeby liczba do pracy byla ta poprawna.
    """

    n_pixels: int = 0
    n_samples: int = 0
    sq_err: float = 0.0
    abs_rel: float = 0.0
    log10: float = 0.0
    mae: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    per_sample_rmse: list[float] = field(default_factory=list)

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """pred/gt: (B,1,H,W) w metrach. mask: dodatkowe ograniczenie pikseli."""
        valid = gt > 0
        if mask is not None:
            valid = valid & mask
        if not bool(valid.any()):
            return

        p = pred.clamp_min(1e-6)
        g = gt

        for b in range(gt.shape[0]):
            vb = valid[b]
            if not bool(vb.any()):
                continue
            pb, gb = p[b][vb], g[b][vb]
            diff = gb - pb
            self.sq_err += float((diff ** 2).sum())
            self.mae += float(diff.abs().sum())
            self.abs_rel += float((diff.abs() / gb).sum())
            self.log10 += float((torch.log10(gb) - torch.log10(pb)).abs().sum())
            thresh = torch.maximum(gb / pb, pb / gb)
            self.d1 += float((thresh < 1.25).sum())
            self.d2 += float((thresh < 1.25 ** 2).sum())
            self.d3 += float((thresh < 1.25 ** 3).sum())
            self.n_pixels += int(vb.sum())
            self.n_samples += 1
            self.per_sample_rmse.append(float(torch.sqrt((diff ** 2).mean())))

    def result(self) -> dict[str, float]:
        if self.n_pixels == 0:
            return {k: float("nan") for k in METRIC_NAMES} | {"n_pixels": 0, "n_samples": 0}
        n = self.n_pixels
        return {
            "RMSE": float(np.sqrt(self.sq_err / n)),
            "RMSE_per_sample": float(np.mean(self.per_sample_rmse)) if self.per_sample_rmse else float("nan"),
            "ABS_REL": self.abs_rel / n,
            "LOG10": self.log10 / n,
            "MAE": self.mae / n,
            "DELTA1": self.d1 / n,
            "DELTA2": self.d2 / n,
            "DELTA3": self.d3 / n,
            "n_pixels": n,
            "n_samples": self.n_samples,
        }


class Evaluator:
    """Zbiera metryki globalne, stratyfikowane i per scena w jednym przebiegu.

    Jeden przebieg, a nie trzy: kazdy przebieg po zbiorze walidacyjnym to
    6588 probek do przeczytania z dysku, a mierzone wielkosci sa funkcjami tych
    samych tensorow.
    """

    def __init__(self, scene_names: list[str], edge_threshold: float = EDGE_GRAD_THRESHOLD_M):
        self.scene_names = list(scene_names)
        self.edge_threshold = edge_threshold
        self.overall = MetricAccumulator()
        self.edge = MetricAccumulator()
        self.smooth = MetricAccumulator()
        self.per_scene = {s: MetricAccumulator() for s in self.scene_names}
        self.edge_pixels = 0
        self.valid_pixels = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor, scene_idx: torch.Tensor | None = None) -> None:
        pred = pred.detach().float()
        gt = gt.detach().float()
        edge = depth_edge_mask(gt, self.edge_threshold)
        smooth = (gt > 0) & (~edge)

        self.overall.update(pred, gt)
        self.edge.update(pred, gt, edge)
        self.smooth.update(pred, gt, smooth)
        self.edge_pixels += int(edge.sum())
        self.valid_pixels += int((gt > 0).sum())

        if scene_idx is not None:
            si = scene_idx.detach().cpu().numpy().reshape(-1)
            for u in np.unique(si):
                sel = torch.from_numpy((si == u)).to(pred.device)
                name = self.scene_names[int(u)]
                self.per_scene[name].update(pred[sel], gt[sel])

    def result(self) -> dict:
        frac = (self.edge_pixels / self.valid_pixels) if self.valid_pixels else float("nan")
        return {
            "overall": self.overall.result(),
            "edge": self.edge.result(),
            "smooth": self.smooth.result(),
            "per_scene": {k: v.result() for k, v in self.per_scene.items() if v.n_pixels},
            "edge_threshold_m_per_px": self.edge_threshold,
            "edge_pixel_fraction": frac,
        }


# =====================================================================
# BLOK 2: protokol ewaluacji -- statystyki PER PROBKA
# =====================================================================
#
# DLACZEGO TABELA PER PROBKA, A NIE KOLEJNE AKUMULATORY. Blok 2 wymaga tej samej
# metryki policzonej w kilkunastu roznych grupowaniach: po scenie, po odleglosci
# katowej od siatki treningowej, po grupie otwarte/szczelne, po lokalizacji
# (bootstrap) i na dwoch zbiorach testowych. Osobny akumulator na kazde
# grupowanie oznaczalby albo kilkanascie przebiegow przez zbior testowy, albo
# kilkanascie rownoleglych akumulatorow zdefiniowanych z gory -- a bootstrap po
# lokalizacjach i tak by sie nie dal, bo wymaga LOSOWANIA grup po fakcie.
#
# Zamiast tego zapisujemy dla kazdej probki STATYSTYKI DOSTATECZNE: sumy, z
# ktorych kazda metryka odtwarza sie dokladnie przez zsumowanie wierszy grupy
# i podzielenie przez liczbe pikseli. Metryka dowolnej grupy jest wiec liczona
# BEZ ponownego czytania danych i jest IDENTYCZNA z ta, ktora dalby akumulator
# przepuszczony tylko przez te grupe -- to nie jest przyblizenie.
# Tabela ma ~6 588 wierszy po ~25 liczb, czyli ulamek megabajta.


def _per_sample_sums(pred: torch.Tensor, gt: torch.Tensor,
                     valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Sumy skladnikow metryk dla kazdej probki wsadu, w jednym przebiegu.

    `valid` jest juz gotowa maska (uwzglednia gt>0, warstwe i ewentualna maske
    przeciecia). Wartosci poza maska sa zerowane PO obliczeniu, a `gt`/`pred`
    sa wczesniej klipowane od dolu -- dzieki temu dzielenie i logarytm nigdy nie
    dostaja zera, a wynik i tak nie wchodzi do sumy.
    """
    v = valid.to(pred.dtype)
    g = gt.clamp_min(1e-6)
    p = pred.clamp_min(1e-6)
    # Roznica liczona na PRZYCIETEJ predykcji, dokladnie jak `MetricAccumulator`
    # (a wiec jak `compute_errors` Paridy, ktore dostaje juz przyciete `pred`).
    # Model wypuszcza sigmoid * max_depth, wiec moze zwrocic dokladnie 0 --
    # wtedy surowe `pred` i przyciete `p` rozjezdzaja sie i dwie sciezki metryk
    # przestaja dawac te same liczby.
    diff = (g - p) * v
    ratio = torch.maximum(g / p, p / g)
    dims = (1, 2, 3)
    return {
        "n_px": valid.sum(dim=dims),
        "sq": (diff ** 2).sum(dim=dims),
        "abs": diff.abs().sum(dim=dims),
        "rel": (diff.abs() / g).sum(dim=dims),
        "log10": ((torch.log10(g) - torch.log10(p)).abs() * v).sum(dim=dims),
        "d1": ((ratio < 1.25) & valid).sum(dim=dims),
        "d2": ((ratio < 1.25 ** 2) & valid).sum(dim=dims),
        "d3": ((ratio < 1.25 ** 3) & valid).sum(dim=dims),
    }


class SampleStatsCollector:
    """Zbiera statystyki dostateczne per probka dla wszystkich warstw naraz.

    `valid_ref` (opcjonalna) to MASKA PRZECIECIA: piksele wazne rowniez w
    drugim wariancie geometrii. Powod jest w `geometry_check.py` -- dorobiony
    sufit to geometria syntetyczna, nie zmierzona, wiec model wariantu
    `patched` byłby punktowany na pikselach, ktorych model wariantu `main` nie
    ma szans zobaczyc. Przy wlaczonej masce oba modele sa oceniane na DOKLADNIE
    tym samym zbiorze pikseli.
    """

    def __init__(self, scene_names: list[str], edge_threshold: float = EDGE_GRAD_THRESHOLD_M):
        self.scene_names = list(scene_names)
        self.edge_threshold = edge_threshold
        self._rows: list[dict] = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor, *,
               scene_idx: torch.Tensor, location_id: torch.Tensor,
               angle_deg: torch.Tensor, valid_ref: torch.Tensor | None = None) -> None:
        pred = pred.detach().float()
        gt = gt.detach().float()
        base = gt > 0
        if valid_ref is not None:
            base = base & valid_ref.to(pred.device).bool()
        # Krawedzie liczone z PELNEJ prawdy (bez maski przeciecia): przynaleznosc
        # piksela do nieciaglosci jest wlasnoscia geometrii, a nie tego, ktory
        # wariant go punktuje. Maska przeciecia ogranicza dopiero PUNKTOWANIE.
        edge = depth_edge_mask(gt, self.edge_threshold)
        masks = {"all": base, "edge": base & edge, "smooth": base & ~edge}

        sums = {k: _per_sample_sums(pred, gt, m) for k, m in masks.items()}
        b = gt.shape[0]
        si = scene_idx.detach().cpu().numpy().reshape(-1)
        li = location_id.detach().cpu().numpy().reshape(-1)
        ai = angle_deg.detach().cpu().numpy().reshape(-1)
        cpu = {k: {kk: vv.detach().cpu().numpy() for kk, vv in s.items()} for k, s in sums.items()}
        px_total = int(gt[0].numel())   # pikseli w kadrze, MIERZONE a nie zakladane
        for i in range(b):
            row = {"scene_idx": int(si[i]), "location_id": int(li[i]), "angle_deg": int(ai[i]),
                   "px_total": px_total}
            for st in STRATA:
                for kk, vv in cpu[st].items():
                    row[f"{st}.{kk}"] = float(vv[i])
            self._rows.append(row)

    def table(self) -> "SampleTable":
        cols = {k: np.asarray([r[k] for r in self._rows]) for k in (self._rows[0] if self._rows else {})}
        return SampleTable(cols=cols, scene_names=self.scene_names,
                           edge_threshold=self.edge_threshold)


@dataclass
class SampleTable:
    """Statystyki dostateczne per probka + narzedzia do grupowania."""

    cols: dict[str, np.ndarray]
    scene_names: list[str]
    edge_threshold: float = EDGE_GRAD_THRESHOLD_M

    def __len__(self) -> int:
        return int(self.cols["scene_idx"].size) if self.cols else 0

    @property
    def scenes(self) -> np.ndarray:
        return np.asarray([self.scene_names[i] for i in self.cols["scene_idx"]])

    def key(self) -> np.ndarray:
        """Klucz probki: (scene_idx, location_id, angle_deg) w jednym int64.
        Sluzy do sprawdzenia, ze dwa warunki byly oceniane na TYCH SAMYCH
        probkach -- bez tego porownanie sparowane nie ma podstawy."""
        return (self.cols["scene_idx"].astype(np.int64) * 10_000_000
                + self.cols["location_id"].astype(np.int64) * 1000
                + self.cols["angle_deg"].astype(np.int64))

    def location_key(self) -> np.ndarray:
        return (self.cols["scene_idx"].astype(np.int64) * 100_000
                + self.cols["location_id"].astype(np.int64))

    def aggregate(self, rows: np.ndarray | None = None,
                  strata: tuple[str, ...] = STRATA) -> dict:
        """Metryki dla podzbioru wierszy. Dokladnie te same liczby, ktore dalby
        akumulator przepuszczony wylacznie przez te wiersze."""
        idx = np.arange(len(self)) if rows is None else np.asarray(rows)
        out: dict = {"n_samples": int(idx.size)}
        for st in strata:
            n = float(self.cols[f"{st}.n_px"][idx].sum())
            if n == 0:
                out[st] = {k: float("nan") for k in METRIC_NAMES} | {"n_pixels": 0}
                continue
            sq = float(self.cols[f"{st}.sq"][idx].sum())
            npx = self.cols[f"{st}.n_px"][idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                per_sample = np.sqrt(np.divide(self.cols[f"{st}.sq"][idx], npx,
                                               out=np.full(idx.size, np.nan), where=npx > 0))
            out[st] = {
                "RMSE": float(np.sqrt(sq / n)),
                "RMSE_per_sample": float(np.nanmean(per_sample)) if idx.size else float("nan"),
                "MAE": float(self.cols[f"{st}.abs"][idx].sum() / n),
                "ABS_REL": float(self.cols[f"{st}.rel"][idx].sum() / n),
                "LOG10": float(self.cols[f"{st}.log10"][idx].sum() / n),
                "DELTA1": float(self.cols[f"{st}.d1"][idx].sum() / n),
                "DELTA2": float(self.cols[f"{st}.d2"][idx].sum() / n),
                "DELTA3": float(self.cols[f"{st}.d3"][idx].sum() / n),
                "n_pixels": int(n),
                # 2.4: liczba i ODSETEK waznych pikseli musza stac obok metryki.
                # RMSE liczone na 86 % kadru i na 100 % kadru to nie jest ta sama
                # wielkosc, a dziury nie sa losowe -- siedza tam, gdzie geometrii
                # brakuje. Mianownik bierze sie ze ZMIERZONEJ liczby pikseli kadru,
                # nie ze stalej 128x128 -- inaczej metryka cicho klamalaby przy
                # kazdej innej rozdzielczosci.
                "valid_pixel_fraction": float(n / max(self._px_denominator(idx), 1)),
            }
        return out

    def _px_denominator(self, idx: np.ndarray) -> float:
        if "px_total" in self.cols:
            return float(self.cols["px_total"][idx].sum())
        return float(idx.size * 128 * 128)

    def save(self, path) -> None:
        np.savez_compressed(path, scene_names=np.asarray(self.scene_names),
                            edge_threshold=self.edge_threshold, **self.cols)

    @classmethod
    def load(cls, path) -> "SampleTable":
        z = np.load(path, allow_pickle=False)
        names = [str(x) for x in z["scene_names"]]
        cols = {k: z[k] for k in z.files if k not in ("scene_names", "edge_threshold")}
        return cls(cols=cols, scene_names=names, edge_threshold=float(z["edge_threshold"]))


# ------------------------------------------------------------ odleglosc katowa


def circular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Odleglosc po okregu w stopniach, w zakresie [0, 180]."""
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % 360.0
    return np.minimum(d, 360.0 - d)


def min_distance_to_grid(angle_deg: np.ndarray, grid_deg: np.ndarray) -> np.ndarray:
    """Najmniejsza odleglosc katowa od DOWOLNEGO kata z siatki treningowej.

    To jest os rysunku z punktu 2.2: jesli istnieje luka generalizacji katowej,
    RMSE rosnie monotonicznie z ta odlegloscia. Dla warunku `cardinal` przyjmuje
    wartosci 0, 10, 20, 30, 40 stopni (45 NIE wystepuje, bo siatka renderow ma
    krok 10 stopni i zaden kat testowy nie lezy dokladnie w polowie miedzy
    dwoma kierunkami kardynalnymi).
    """
    a = np.asarray(angle_deg, dtype=np.float64).reshape(-1, 1)
    g = np.asarray(grid_deg, dtype=np.float64).reshape(1, -1)
    d = np.abs(a - g) % 360.0
    return np.minimum(d, 360.0 - d).min(axis=1)


# ---------------------------------------------------- bootstrap po lokalizacjach


def bootstrap_paired_by_location(
    table_a: "SampleTable", table_b: "SampleTable", *,
    stratum: str = "all", metric: str = "RMSE",
    n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
    rows_a: np.ndarray | None = None, rows_b: np.ndarray | None = None,
) -> dict:
    """Przedzial ufnosci RÓŻNICY metryki, losujac LOKALIZACJE ze zwracaniem.

    DLACZEGO PO LOKALIZACJACH. 36 probek jednej lokalizacji rozni sie wylacznie
    orientacja agenta -- pozycja, geometria i wiekszosc trudnosci sceny sa te
    same. Efektywne n zbioru testowego to 183 LOKALIZACJE, nie 6 588 probek.
    Test traktujacy probki jako niezalezne jest antykonserwatywny o czynnik
    rzedu sqrt(36) = 6, czyli zawyza istotnosc o rzad wielkosci.

    DLACZEGO SPAROWANY. Wszystkie warunki sa oceniane na IDENTYCZNYM zbiorze
    testowym, wiec ta sama losowa proba lokalizacji wchodzi do obu ramion i
    wspolna zmiennosc miedzy lokalizacjami (jedne sa po prostu trudniejsze)
    znosi sie w roznicy. Porownywanie dwoch srednich z dwoma osobnymi
    odchyleniami wyrzucilo by te informacje.
    """
    ia = np.arange(len(table_a)) if rows_a is None else np.asarray(rows_a)
    ib = np.arange(len(table_b)) if rows_b is None else np.asarray(rows_b)

    ka, kb = table_a.key()[ia], table_b.key()[ib]
    oa, ob = np.argsort(ka, kind="stable"), np.argsort(kb, kind="stable")
    if not np.array_equal(ka[oa], kb[ob]):
        raise ValueError("bootstrap sparowany wymaga identycznego zbioru probek "
                         "w obu ramionach (klucz scene/location/angle sie rozni)")
    ia, ib = ia[oa], ib[ob]

    loc = table_a.location_key()[ia]
    uniq, inv = np.unique(loc, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(uniq.size)]

    def stat(rows_local: np.ndarray) -> float:
        va = table_a.aggregate(ia[rows_local], strata=(stratum,))[stratum][metric]
        vb = table_b.aggregate(ib[rows_local], strata=(stratum,))[stratum][metric]
        return float(va - vb)

    point = stat(np.arange(ia.size))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        boot[i] = stat(np.concatenate([groups[j] for j in pick]))

    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {
        "metric": metric, "stratum": stratum,
        "delta_point": point,
        "ci_low": lo, "ci_high": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "boot_sd": float(boot.std(ddof=1)),
        "n_locations": int(uniq.size),
        "n_samples": int(ia.size),
        "n_boot": n_boot, "alpha": alpha, "seed": seed,
    }


def compute_errors_parida(gt: np.ndarray, pred: np.ndarray) -> tuple:
    """Kopia 1:1 `util.util.compute_errors` -- referencja dla testu zgodnosci."""
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    mae = (np.abs(gt - pred)).mean()
    return abs_rel, rmse, a1, a2, a3, log_10, mae


def test_matches_parida(seed: int = 0, n: int = 8, tol: float = 1e-5) -> dict:
    """Dowod, ze nasza implementacja daje te same liczby co kod Paridy.

    Porownujemy wersje "per probka" (`RMSE_per_sample`), bo to ta, ktora liczy
    `train.py`; globalne RMSE po pikselach musi sie roznic i o to chodzi.
    """
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.2, 10.0, size=(n, 1, 128, 128)).astype(np.float32)
    gt[rng.random(gt.shape) < 0.03] = 0.0  # dziury jak w prawdziwej glebi
    pred = np.clip(gt + rng.normal(0, 0.5, gt.shape), 0.05, None).astype(np.float32)

    ref = np.array([compute_errors_parida(gt[i], pred[i]) for i in range(n)]).mean(0)
    acc = MetricAccumulator()
    acc.update(torch.from_numpy(pred), torch.from_numpy(gt))
    ours = acc.result()

    diffs = {
        "RMSE_per_sample": abs(ours["RMSE_per_sample"] - ref[1]),
        "ABS_REL": abs(ours["ABS_REL"] - ref[0]),
        "DELTA1": abs(ours["DELTA1"] - ref[2]),
        "DELTA2": abs(ours["DELTA2"] - ref[3]),
        "DELTA3": abs(ours["DELTA3"] - ref[4]),
        "LOG10": abs(ours["LOG10"] - ref[5]),
        "MAE": abs(ours["MAE"] - ref[6]),
    }
    worst = float(max(diffs.values()))
    return {"max_diff": worst, "ok": bool(worst < tol),
            "diffs": {k: float(v) for k, v in diffs.items()}}


def test_table_matches_accumulator(seed: int = 0, n: int = 24, tol: float = 1e-5) -> dict:
    """Dowod, ze tabela per probka daje te same liczby co `MetricAccumulator`.

    To jest test ZASTAPIENIA: caly Blok 2 liczy metryki z tabeli, a `train.py`
    Paridy i dotychczasowa walidacja licza z akumulatora. Gdyby te dwie drogi
    sie rozjezdzaly, wyniki macierzy nie dalyby sie zestawic z niczym.

    Sprawdzane sa TRZY rzeczy naraz:
      1. zgodnosc na calym zbiorze,
      2. zgodnosc na PODZBIORZE (grupowanie po scenie/kacie -- to jest wlasnie
         to, na czym stoi Blok 2),
      3. maska przeciecia faktycznie ogranicza zbior punktowanych pikseli.
    """
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.2, 10.0, size=(n, 1, 128, 128)).astype(np.float32)
    gt[rng.random(gt.shape) < 0.05] = 0.0
    pred = np.clip(gt + rng.normal(0, 0.5, gt.shape), 0.05, None).astype(np.float32)
    gt_t, pred_t = torch.from_numpy(gt), torch.from_numpy(pred)

    scene_idx = torch.from_numpy(rng.integers(0, 3, size=n))
    loc = torch.from_numpy(rng.integers(0, 5, size=n))
    ang = torch.from_numpy(rng.choice(np.arange(0, 360, 10), size=n))

    col = SampleStatsCollector(["s0", "s1", "s2"])
    col.update(pred_t, gt_t, scene_idx=scene_idx, location_id=loc, angle_deg=ang)
    tab = col.table()

    acc = MetricAccumulator()
    acc.update(pred_t, gt_t)
    ref = acc.result()
    got = tab.aggregate()["all"]
    diffs = {k: abs(got[k] - ref[k]) for k in
             ("RMSE", "RMSE_per_sample", "MAE", "ABS_REL", "LOG10", "DELTA1", "DELTA2", "DELTA3")}

    # (2) ten sam test na podzbiorze -- tylko probki sceny 0.
    sub = np.flatnonzero(tab.cols["scene_idx"] == 0)
    if sub.size:
        acc2 = MetricAccumulator()
        acc2.update(pred_t[sub], gt_t[sub])
        ref2 = acc2.result()
        got2 = tab.aggregate(sub)["all"]
        for k in ("RMSE", "MAE", "ABS_REL", "DELTA1"):
            diffs[f"subset.{k}"] = abs(got2[k] - ref2[k])

    # (3) predykcja zawierajaca DOKLADNIE zera. Model wypuszcza sigmoid*max_depth,
    # wiec to jest osiagalne; przed poprawka obie sciezki rozjezdzaly sie tutaj,
    # bo jedna liczyla roznice na przycietej predykcji, a druga na surowej.
    pred0 = pred.copy()
    pred0[rng.random(pred0.shape) < 0.10] = 0.0
    p0_t = torch.from_numpy(pred0)
    acc0 = MetricAccumulator()
    acc0.update(p0_t, gt_t)
    ref0 = acc0.result()
    col0 = SampleStatsCollector(["s0", "s1", "s2"])
    col0.update(p0_t, gt_t, scene_idx=scene_idx, location_id=loc, angle_deg=ang)
    got0 = col0.table().aggregate()["all"]
    for k in ("RMSE", "MAE", "ABS_REL", "LOG10", "DELTA1"):
        diffs[f"pred_zero.{k}"] = abs(got0[k] - ref0[k])

    # (4) maska przeciecia: polowa pikseli wylaczona -> mniej punktowanych.
    ref_valid = torch.from_numpy(rng.random(gt.shape) > 0.5)
    col3 = SampleStatsCollector(["s0", "s1", "s2"])
    col3.update(pred_t, gt_t, scene_idx=scene_idx, location_id=loc, angle_deg=ang,
                valid_ref=ref_valid)
    masked = col3.table().aggregate()["all"]
    mask_works = masked["n_pixels"] < got["n_pixels"]

    worst = float(max(diffs.values()))
    return {"max_diff": worst, "ok": bool(worst < tol and mask_works),
            "intersection_mask_reduces_pixels": bool(mask_works),
            "n_pixels_full": int(got["n_pixels"]), "n_pixels_masked": int(masked["n_pixels"]),
            "diffs": {k: float(v) for k, v in diffs.items()}}
