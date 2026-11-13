"""ED: oryginal (2026-08-11) vs powtorzenia nf (2026-11-12) -- rozklad roznicy RMSE test@36.

Tylko CPU, tylko odczyt. Uruchomienie (z katalogu repo):
    CUDA_VISIBLE_DEVICES="" python -I outputs/ml/overnight_2026-11/ed_vs_nf_analysis.py

RMSE grupy = sqrt(sum(all.sq) / sum(all.n_px)) -- jak SampleTable.aggregate.
Roznica RMSE nie jest addytywna; addytywny jest wklad do sumy kwadratow:
dSQ(grupa) = sum_grupa(sq_orig - sq_ref), udzial = dSQ(grupa) / dSQ(razem).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "my-operations"))
from ml.dataset.angles import select_angles  # noqa: E402

EVAL = REPO / "outputs/ml/eval"
TS = "test@36"
RUNS0_NF = ["ED_seed0_nf0", "ED_seed0_nf1", "ED_seed0_nf2", "ED_seed0_nf3", "ED_seed0_nf4"]
PAIRS = {1: ("ED_seed1", "ED_seed1_nf0"), 2: ("ED_seed2", "ED_seed2_nf0")}
ALL = ["ED_seed0", *RUNS0_NF, *[r for p in PAIRS.values() for r in p]]
CARD = (0, 90, 180, 270)


def load(run):
    z = np.load(EVAL / run / f"samples_{TS}.npz", allow_pickle=False)
    e = json.load(open(EVAL / run / "eval.json"))
    return {k: z[k] for k in z.files}, e


D, E = {}, {}
for r in ALL:
    D[r], E[r] = load(r)

ref = D["ED_seed0"]
key = np.stack([ref["scene_idx"], ref["location_id"], ref["angle_deg"]], 1)
scenes = list(ref["scene_names"])
npx = ref["all.n_px"]
print("## 0. Kontrola spojnosci")
worst_rmse, worst_scene = 0.0, 0.0
for r in ALL:
    d = D[r]
    k2 = np.stack([d["scene_idx"], d["location_id"], d["angle_deg"]], 1)
    assert np.array_equal(key, k2), r
    assert list(d["scene_names"]) == scenes, r
    assert np.array_equal(d["all.n_px"], npx), r
    rm = np.sqrt(d["all.sq"].sum() / npx.sum())
    worst_rmse = max(worst_rmse, abs(rm - E[r]["test_sets"][TS]["overall"]["all"]["RMSE"]))
    for si, s in enumerate(scenes):
        m = d["scene_idx"] == si
        rs = np.sqrt(d["all.sq"][m].sum() / npx[m].sum())
        worst_scene = max(worst_scene, abs(rs - E[r]["test_sets"][TS]["per_scene"][s]["all"]["RMSE"]))
print(f"wiersze {len(npx)}, lokalizacje {len(np.unique(key[:, :2], axis=0))}, katy {len(np.unique(key[:, 2]))}; "
      f"klucze/n_px identyczne we wszystkich {len(ALL)} runach")
print(f"max |RMSE_odtw - eval.json| overall = {worst_rmse:.2e}, per_scene = {worst_scene:.2e}")
print("ED_seed0 odtworzone:", f"{np.sqrt(D['ED_seed0']['all.sq'].sum() / npx.sum()):.8f}",
      "eval.json:", f"{E['ED_seed0']['test_sets'][TS]['overall']['all']['RMSE']:.8f}")

N = npx.sum()
sid, loc, ang = key[:, 0], key[:, 1], key[:, 2]
locid = sid * 100000 + loc
ulocs = np.unique(locid)
uang = np.arange(0, 360, 10)


def rmse(sq, m=None):
    m = np.ones(len(sq), bool) if m is None else m
    return float(np.sqrt(sq[m].sum() / npx[m].sum()))


def group_sums(v, labels, ulab):
    out = np.zeros(len(ulab))
    idx = np.searchsorted(ulab, labels)
    np.add.at(out, idx, v)
    return out


SQ0 = np.stack([D[r]["all.sq"] for r in RUNS0_NF])  # 5 x n
SQbar = SQ0.mean(0)
O0 = D["ED_seed0"]["all.sq"]


def describe(delta, label, noise_rows=None, noise_scale=1.0):
    """delta: per-sample dSQ (orig - ref). noise_rows: k x n per-sample dSQ realizacje
    pod H0 (te same zmienne co delta, ale miedzy powtorzeniami nf), ze skala noise_scale
    dopasowujaca wariancje do porownania delta."""
    tot = delta.sum()
    print(f"\n### {label}")
    print(f"dSQ razem = {tot:.1f} (sum n_px = {N:.0f}; dMSE = {tot / N:+.6f})")

    def nz(groupvals_fn):
        if noise_rows is None:
            return None
        g = np.stack([groupvals_fn(x) for x in noise_rows])
        return g.std(0, ddof=1) * noise_scale

    # (i) sceny
    sc = np.array([delta[sid == i].sum() for i in range(3)])
    sc_sd = nz(lambda x: np.array([x[sid == i].sum() for i in range(3)]))
    print("| scena | n_px | dSQ | udzial w dSQ | n_px udzial | z (vs szum nf) |")
    print("|---|---|---|---|---|---|")
    for i, s in enumerate(scenes):
        z = f"{sc[i] / sc_sd[i]:+.2f}" if sc_sd is not None else "-"
        print(f"| {s} | {npx[sid == i].sum():.0f} | {sc[i]:+.1f} | {sc[i] / tot:+.1%} | "
              f"{npx[sid == i].sum() / N:.1%} | {z} |")
    if sc_sd is not None:
        tsd = nz(lambda x: np.array([x.sum()]))[0]
        print(f"| RAZEM | {N:.0f} | {tot:+.1f} | 100% | 100% | {tot / tsd:+.2f} |")

    # (ii) katy
    a = group_sums(delta, ang, uang)
    a_sd = nz(lambda x: group_sums(x, ang, uang))
    card = np.isin(uang, CARD)
    print(f"\nkaty kardynalne (4/36 = 11.1% probek): udzial {a[card].sum() / tot:+.1%}; "
          f"pozostale 32: {a[~card].sum() / tot:+.1%}")
    q = [a[(uang >= 90 * k) & (uang < 90 * (k + 1))].sum() / tot for k in range(4)]
    print("cwiartki [0-80, 90-170, 180-260, 270-350]: " + ", ".join(f"{x:+.1%}" for x in q))
    same = int((np.sign(a) == np.sign(tot)).sum())
    print(f"katy ze znakiem zgodnym z roznica calkowita: {same}/36")
    order = np.argsort(-a * np.sign(tot))
    cum = np.cumsum(a[order]) / tot
    k50 = int(np.argmax(cum >= 0.5)) + 1
    print(f"najwieksze wklady katowe: " + ", ".join(f"{uang[j]}:{a[j] / tot:+.1%}" for j in order[:5])
          + f"; 50% roznicy osiaga {k50} najwiekszych katow (rownomiernie: 18)")
    best_w = {}
    for w in (3, 6, 9):
        sh = [sum(a[(j + t) % 36] for t in range(w)) / tot for j in range(36)]
        jb = int(np.argmax(sh))
        best_w[w] = (uang[jb], sh[jb])
    print("najlepsze ciagle okno katowe (start, udzial; oczekiwane w/36): "
          + ", ".join(f"w={w}: {s}..{(s + 10 * (w - 1)) % 360} -> {v:+.1%} (oczek. {w / 36:.1%})"
                      for w, (s, v) in best_w.items()))
    if a_sd is not None:
        za = a / a_sd
        print(f"z per kat: srednia {za.mean():+.2f}, min {za.min():+.2f}, max {za.max():+.2f}, "
              f"sum z^2 = {np.sum(za ** 2):.1f} (H0: z ~ t o 4 st. swobody, E[z^2]=2 -> ~72)")
    coh = a.sum() / np.abs(a).sum()
    print(f"spojnosc katowa sum(d)/sum|d| = {coh:+.3f}")

    # (iii) lokalizacje
    l = group_sums(delta, locid, ulocs)
    l_sd = nz(lambda x: group_sums(x, locid, ulocs))
    order = np.argsort(-l * np.sign(tot))
    k10 = int(round(0.1 * len(ulocs)))
    same = int((np.sign(l) == np.sign(tot)).sum())
    print(f"\nlokalizacje: {len(ulocs)}; ze znakiem zgodnym z roznica: {same} ({same / len(ulocs):.1%})")
    print(f"top 1 / top 5 / top 10% ({k10}) / top 25% lokalizacji (wg wkladu w kierunku roznicy): "
          f"{l[order[:1]].sum() / tot:+.1%} / {l[order[:5]].sum() / tot:+.1%} / "
          f"{l[order[:k10]].sum() / tot:+.1%} / {l[order[:46]].sum() / tot:+.1%} roznicy netto")
    absl = np.abs(l)
    oa = np.argsort(-absl)
    print(f"top 10% lokalizacji wg |d| niesie {absl[oa[:k10]].sum() / absl.sum():.1%} sumy |d| "
          f"(spojnosc sum(d)/sum|d| = {l.sum() / absl.sum():+.3f})")
    sc_of_top = [scenes[int(u // 100000)] for u in ulocs[order[:k10]]]
    print("sceny top-10% lokalizacji: " + ", ".join(f"{s}:{sc_of_top.count(s)}" for s in scenes))
    if l_sd is not None:
        zl = l / l_sd
        print(f"z per lokalizacja: srednia {zl.mean():+.3f} (sd {zl.std(ddof=1):.2f}; srednia*sqrt(183) = "
              f"{zl.mean() * np.sqrt(len(zl)):+.2f}), |z|>3: {int((np.abs(zl) > 3).sum())}, "
              f"z<-2: {int((zl < -2).sum())}, z>+2: {int((zl > 2).sum())}")
    return dict(sc=sc, a=a, l=l, tot=tot, absl=absl, k10=k10)


def per_scene_rmse_table(runs_o, runs_r, title):
    print(f"\n{title}")
    print("| scena | RMSE oryg. | RMSE ref | dRMSE |")
    print("|---|---|---|---|")
    for i, s in enumerate([*scenes, "RAZEM"]):
        m = sid == i if i < 3 else None
        ro = np.mean([rmse(D[r]["all.sq"], m) for r in runs_o])
        rr = [rmse(D[r]["all.sq"], m) for r in runs_r]
        extra = f" (nf sd {np.std(rr, ddof=1):.5f}, rozstep {np.ptp(rr):.5f})" if len(rr) > 1 else ""
        print(f"| {s} | {ro:.5f} | {np.mean(rr):.5f}{extra} | {ro - np.mean(rr):+.5f} |")


print("\n## 1. Ziarno 0: ED_seed0 vs srednia nf0..4")
per_scene_rmse_table(["ED_seed0"], RUNS0_NF, "RMSE per scena (ref = srednia RMSE pieciu nf)")
# szum: leave-one-out nf_i - mean(pozostalych 4) ma wariancje s^2(1+1/4); orig - mean(5) ma s^2(1+1/5)
loo = np.stack([SQ0[i] - np.delete(SQ0, i, 0).mean(0) for i in range(5)])
# z: sd sumy grupowej miedzy 5 nf (wariancja s^2 jednego runu) * sqrt(1 + 1/5),
# bo oryg - sr(5) ma pod H0 wariancje s^2 (1 + 1/5). sd z 5 realizacji: 4 stopnie swobody.
r0 = describe(O0 - SQbar, "ED_seed0 - srednia(nf0..4)  [z: dSQ / (sd_nf * sqrt(1.2))]",
              noise_rows=SQ0, noise_scale=np.sqrt(1.2))

print("\n## 1b. ED_seed0 vs kazdy nf osobno")
print("| para | dRMSE | udzial apartment_2 | frl_apartment_5 | office_4 | kardynalne | top10% lok. (netto) | spojnosc lok. |")
print("|---|---|---|---|---|---|---|---|")
for i, r in enumerate(RUNS0_NF):
    d = O0 - SQ0[i]
    t = d.sum()
    sc = [d[sid == j].sum() / t for j in range(3)]
    a = group_sums(d, ang, uang)
    l = group_sums(d, locid, ulocs)
    o = np.argsort(-l * np.sign(t))
    print(f"| oryg - {r} | {rmse(O0) - rmse(SQ0[i]):+.5f} | " + " | ".join(f"{x:+.1%}" for x in sc)
          + f" | {a[np.isin(uang, CARD)].sum() / t:+.1%} | {l[o[:18]].sum() / t:+.1%} | {l.sum() / np.abs(l).sum():+.3f} |")

print("\n## 1c. Szum miedzy powtorzeniami nf (nf_i - srednia pozostalych 4)")
print("| para | dRMSE | dSQ apartment_2 | frl_apartment_5 | office_4 | spojnosc lok. | top10% lok. wg |d| |")
print("|---|---|---|---|---|---|---|")
for i, r in enumerate(RUNS0_NF):
    d = loo[i]
    l = group_sums(d, locid, ulocs)
    al = np.abs(l)
    oth = np.delete(SQ0, i, 0)
    drm = rmse(SQ0[i]) - np.mean([rmse(x) for x in oth])
    print(f"| {r} - sr(reszta) | {drm:+.5f} | " + " | ".join(f"{d[sid == j].sum():+.1f}" for j in range(3))
          + f" | {l.sum() / al.sum():+.3f} | {np.sort(al)[::-1][:18].sum() / al.sum():.1%} |")
sd_sc = np.array([[x[sid == j].sum() for j in range(3)] for x in SQ0]).std(0, ddof=1)
print("sd dSQ per scena miedzy nf (pojedynczy run): " + ", ".join(f"{s}: {v:.1f}" for s, v in zip(scenes, sd_sc))
      + f"; sd dSQ razem: {SQ0.sum(1).std(ddof=1):.1f}")
print("dla porownania oryg - sr(nf) per scena: " + ", ".join(f"{s}: {v:+.1f}" for s, v in zip(scenes, r0['sc'])))
sd_rm_sc = np.array([[rmse(x, sid == j) for j in range(3)] for x in SQ0]).std(0, ddof=1)
print("sd RMSE per scena miedzy nf: " + ", ".join(f"{s}: {v:.5f}" for s, v in zip(scenes, sd_rm_sc)))

# rozklad pod H0 statystyk ksztaltu: kazdy nf_i jako "pseudo-oryginal" vs sr(pozostalych)
print("\nstatystyki ksztaltu pod H0 (nf_i vs sr. pozostalych 4) -- do porownania z oryg:")
for i, r in enumerate(RUNS0_NF):
    d = loo[i]
    a = group_sums(d, ang, uang)
    l = group_sums(d, locid, ulocs)
    print(f"  {r}: spojnosc katowa {a.sum() / np.abs(a).sum():+.3f}, lok. zgodne ze znakiem "
          f"{int((np.sign(l) == np.sign(d.sum())).sum())}/183, spojnosc lok. {l.sum() / np.abs(l).sum():+.3f}")

# korelacja wzorca: czy wzorzec per lokalizacja oryg-nf jest podobny do wzorca nf_i - sr?
print("\nkorelacja per lokalizacja: (oryg - sr nf) vs (nf_i - sr pozostalych):")
lo = group_sums(O0 - SQbar, locid, ulocs)
print("  " + ", ".join(f"{r[-3:]}: {np.corrcoef(lo, group_sums(loo[i], locid, ulocs))[0, 1]:+.2f}"
                       for i, r in enumerate(RUNS0_NF)))

print("\n## 2. Ziarna 1 i 2 (jedna para na ziarno)")
pair_noise = np.stack([SQ0[i] - SQ0[j] for i in range(5) for j in range(i + 1, 5)])  # var 2 s^2
for s, (o, n) in PAIRS.items():
    per_scene_rmse_table([o], [n], f"RMSE per scena, {o} vs {n}")
    describe(D[o]["all.sq"] - D[n]["all.sq"],
             f"{o} - {n}  [z: dSQ / sd(par nf ziarna 0) -- zalozenie: szum ziarna 0 ~ szum ziarna {s}]",
             noise_rows=SQ0, noise_scale=np.sqrt(2))

print("\n## 3. (iv) Katy treningowe ED (random_4, angle_seed=0)")
split = json.load(open(REPO / "outputs/ml/splits/replica_locations.json"))
cfg_seeds = {r: json.load(open(REPO / "outputs/ml/runs" / r / "config.json"))["condition"]["angle_seed"]
             for r in ["ED_seed0", "ED_seed1", "ED_seed2", "ED_seed0_nf0"]}
print("angle_seed w config.json:", cfg_seeds)
cnt = np.zeros(36, int)
for sc, locs in split["train"].items():
    for lid in locs:
        for a in select_angles("random_4", scene=sc, location_id=int(lid), seed=0):
            cnt[a // 10] += 1
print(f"probek treningowych: {cnt.sum()} (oczek. 5496); licznosc per kat: min {cnt.min()}, max {cnt.max()}, "
      f"srednia {cnt.mean():.1f}; kardynalne: {[int(cnt[c // 10]) for c in CARD]}")
test_overlap = set(split["test"]["apartment_2"]) & set(split["train"].get("apartment_2", []))
print("wspolne lokalizacje test/train w scenach testowych:", len(test_overlap),
      "| sceny testowe w train:", [s for s in scenes if s in split["train"]])


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


rho = spearman(cnt, r0["a"])
null = [spearman(cnt, group_sums(loo[i], ang, uang)) for i in range(5)]
print(f"Spearman(licznosc kata w treningu, dSQ per kat oryg - sr nf) = {rho:+.3f}; "
      f"to samo dla nf_i - sr(reszta): {', '.join(f'{x:+.2f}' for x in null)}")


print("\n## 4. Skupienie katowe vs szum (ziarno 0)")


def win3(d):
    a = group_sums(d, ang, uang)
    t = np.sign(a.sum())
    w = np.array([a[j] + a[(j + 1) % 36] + a[(j + 2) % 36] for j in range(36)]) * t
    j = int(np.argmax(w))
    return uang[j], w[j] / np.abs(a).sum()


s0, v0 = win3(O0 - SQbar)
print(f"oryg - sr(nf): najlepsze okno 3 katow {s0}..{s0 + 20}, suma okna / sum|d_kat| = {v0:.3f}")
for i, r in enumerate(RUNS0_NF):
    s1, v1 = win3(loo[i])
    print(f"  {r} - sr(reszta): okno {s1}..{s1 + 20}, {v1:.3f}")
m = np.isin(ang, (40, 50, 60, 70, 80, 90))
d = O0 - SQbar
print("katy 40..90, dSQ per scena: " + ", ".join(f"{s}: {d[m & (sid == j)].sum():+.0f}" for j, s in enumerate(scenes))
      + f"; razem {d[m].sum():+.0f} z {d.sum():+.0f}")
print("katy poza 40..90, dSQ per scena: "
      + ", ".join(f"{s}: {d[~m & (sid == j)].sum():+.0f}" for j, s in enumerate(scenes)))
lw = group_sums(np.where(m, d, 0), locid, ulocs)
o = np.argsort(lw)
print(f"katy 40..90: top 5 lokalizacji niesie {lw[o[:5]].sum() / d[m].sum():.1%} ich sumy; "
      f"lokalizacji z d<0: {(lw < 0).sum()}/183")
print(f"office_4 w okreslonych lokalizacjach: top-10% lokalizacji office_4 wg wkladu: ", end="")
lo4 = group_sums(np.where(sid == 2, d, 0), locid, ulocs)
o4 = np.argsort(lo4)
n4 = int((ulocs // 100000 == 2).sum())
print(f"{lo4[o4[:max(1, n4 // 10)]].sum() / d[sid == 2].sum():.1%} sumy office_4 ({n4} lokalizacji), "
      f"z d<0: {int((lo4[ulocs // 100000 == 2] < 0).sum())}/{n4}")
