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
