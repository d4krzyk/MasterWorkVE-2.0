#!/usr/bin/env python3
"""EKSPERYMENT: zalatanie WSZYSTKICH duzych dziur w siatce Replica, z materialem wg typu dziury.

Nastepca patch_scene_ceiling.py, ktory umial tylko doklejac plaszczyzne sufitu. Tamten
zostaje, bo na nim oparty jest wynik z RAPORT_SESJI §2.12; ten obsluguje wszystkie sceny.

SKAD SIE BIORA DZIURY (zmierzone 2026-07-29 na 4 scenach nieszczelnych bokiem i 6 bez
sufitu): kazda scena ma DOKLADNIE 1-2 dziury o polu > 1 m2 i kilkaset o polu < 0.5 m2.
Te male to brzegi obiektow (spod krzesla, blat stolu) — NIE WOLNO ich latac, bo to nie sa
otwory pomieszczenia. Prog 1 m2 rozdziela je czysto (najwieksza "obiektowa" ma 0.47 m2,
najmniejsza "pomieszczeniowa" 1.12 m2).

CZYM SA DUZE DZIURY — to decyduje o materiale:
  * frl_apartment_*  — jedna pozioma petla ~91 m2 na szczycie scian (przy brzegu wall:3686):
                       BRAKUJACY SUFIT;
  * office_2/office_3 — po dwie petle pionowe, przy brzegu `window` i `door`: OKNO i DRZWI.
                       Szklo nie odbija swiatla strukturalnego IR, wiec skaner ich nie zlapal;
  * apartment_1/2    — jedna duza petla pionowa, przy brzegu floor+ceiling+wall:
                       URWANA KRAWEDZ SKANU (mieszkanie ciagnie sie dalej).

MATERIAL nie jest dobierany recznie: lata dziedziczy `object_id` z otoczenia dziury, wiec
material wynika z tej samej konfiguracji, co reszta sceny. Regula:
  * jesli dominujaca klasa przy brzegu to `window` albo `door` — bierzemy ja (Glass / wood);
  * w przeciwnym razie preferujemy `wall`, jesli wystepuje przy brzegu (urwana krawedz skanu
    to brakujaca SCIANA, a nie podloga — bez tej reguly dominowalby `floor`);
  * inaczej klasa dominujaca.
W konfiguracji Repliki `wall` i `ceiling` wskazuja ten sam material (Gypsum Board), wiec
lata sufitu dostaje material scian sceny.

GEOMETRIA: kazda dziura jest wypelniana SIATKA PELNYCH CZWOROKATOW w plaszczyznie dziury,
przycieta testem naleznosci do jej obrysu. Pierwsza wersja uzywala wachlarza trojkatow
z centroidu i ZAWIODLA POMIAROWO (ucieczka promieni 21.79 % -> 20.26 % zamiast do zera) —
powody i dowod w docstringu `fill_planar()`. Format Repliki ma stala dlugosc rekordu sciany
19 B tylko dlatego, ze wszystkie sciany sa czworokatami, wiec nowe tez musza nimi byc.

Bezpieczenstwo plaskiego wypelnienia sprawdzone osobno: petle sufitowe i okienne sa plaskie
(odchylka RMS 0.005-0.007 m), a jedyne mocno niepłaskie (urwane krawedzie `apartment_1/2`,
RMS 0.6-1.4 m) maja WSZYSTKIE pozycje agenta po jednej stronie swojej plaszczyzny, wiec lata
nie przecina obszaru nawigowalnego. Skrypt sprawdza to sam i ostrzega.

Uruchomienie:
    python my-operations/measurements/patch_scene_holes.py --scene frl_apartment_2
    python my-operations/measurements/patch_scene_holes.py --all
    python my-operations/measurements/patch_scene_holes.py --scene office_2 --report  # bez zapisu
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_core.paths import REPO_ROOT, SCENE_ROOT
from patch_scene_ceiling import FACE_DTYPE, VERTEX_DTYPE, read_ply

PATCHED_ROOT = REPO_ROOT / "outputs/patched_scenes"
MIN_AREA = 1.0          # m^2 — patrz naglowek: rozdziela dziury pomieszczen od brzegow mebli
PREFERRED = ("window", "door")

# Klasy tworzace POWLOKE pomieszczenia. Dziura, ktorej brzeg nie dotyka zadnej z nich,
# nie jest otworem pomieszczenia, tylko wnetrzem obiektu — np. frl_apartment_5 ma petle
# 1.13 m2 zlozona w calosci z klasy `handrail` (otwarty koniec rury porecze). Zalatanie
# jej niczego nie domyka, a dokłada powierzchnie odbijajaca w srodku sceny.
STRUCTURAL = {"wall", "floor", "ceiling", "window", "door", "pillar", "panel", "stair",
              "blinds", "curtain", "beam", "column"}


def detect_loops(V, F):
    """-> (loops, ba, bb, bf) — petle krawedzi brzegowych (indeksy do ba/bb/bf)."""
    I = F["i"].astype(np.int64)
    nf = len(I)
    a = np.concatenate([I[:, 0], I[:, 1], I[:, 2], I[:, 3]])
    b = np.concatenate([I[:, 1], I[:, 2], I[:, 3], I[:, 0]])
    fid = np.tile(np.arange(nf), 4)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    # Klucz kanoniczny krawedzi = lo<<shift | hi. `shift` liczony z liczby wierzcholkow,
    # a nie ustalony na sztywno: apartment_2 ma 2 136 963 wierzcholkow, czyli WIECEJ niz
    # 2^21 — staly shift=21 cicho sklejalby rozne krawedzie w jeden klucz.
    shift = int(len(V) - 1).bit_length()
    if 2 * shift >= 63:
        raise RuntimeError(f"scena ma {len(V)} wierzcholkow — klucz krawedzi nie miesci sie w int64")
    _, inv, cnt = np.unique((lo << shift) + hi, return_inverse=True, return_counts=True)
    bnd = cnt[inv] == 1                     # krawedz nalezaca do dokladnie jednej sciany
    ba, bb, bf = a[bnd], b[bnd], fid[bnd]

    succ = defaultdict(list)
    for k, u in enumerate(ba):
        succ[u].append(k)
    used = np.zeros(len(ba), bool)
    loops = []
    for s in range(len(ba)):
        if used[s]:
            continue
        chain, k = [], s
        while not used[k]:
            used[k] = True
            chain.append(k)
            nxt = [j for j in succ.get(bb[k], []) if not used[j]]
            if not nxt:
                break
            k = nxt[0]
        if len(chain) >= 3:
            loops.append(np.array(chain))
    return loops, ba, bb, bf


def loop_area(P):
    """Zgrubne pole: iloczyn dwoch najwiekszych rozciagliwosci bboxa petli."""
    e = np.sort(P.max(0) - P.min(0))[::-1]
    return float(e[0] * e[1])


def classify_loop(ch, bf, obj, label_of, force_class=None, class_pool=None):
    """-> (object_id laty, klasa laty, histogram klas przy brzegu)

    `force_class` sluzy WYLACZNIE do sweepu materialowego. Podmiana klasy laty nie ma
    zadnych efektow ubocznych: lata dostaje `object_id` jakiegos obiektu tej klasy, ale
    sciany TAMTEGO obiektu pozostaja niezmienione, wiec zmienia sie material dokladnie
    tych trojkatow, ktore doklejamy, i niczego wiecej.
    """
    hist = defaultdict(int)
    per_class_obj = defaultdict(lambda: defaultdict(int))
    for o in obj[bf[ch]]:
        o = int(o)
        lab = label_of(o)
        hist[lab] += 1
        per_class_obj[lab][o] += 1

    if force_class:
        if force_class in per_class_obj:
            oid = max(per_class_obj[force_class].items(), key=lambda t: t[1])[0]
        elif class_pool and force_class in class_pool:
            oid = class_pool[force_class]        # dowolny obiekt tej klasy w scenie
        else:
            raise SystemExit(f"scena nie ma obiektu klasy '{force_class}' — dostepne: "
                             f"{sorted(class_pool or [])[:40]}")
        return oid, force_class, dict(sorted(hist.items(), key=lambda t: -t[1])[:4])

    top = max(hist.items(), key=lambda t: t[1])[0]
    if top in PREFERRED:
        chosen = top
    elif "wall" in hist:
        chosen = "wall"
    else:
        chosen = top
    oid = max(per_class_obj[chosen].items(), key=lambda t: t[1])[0]
    return oid, chosen, dict(sorted(hist.items(), key=lambda t: -t[1])[:4])


def ceiling_height(V, F, label_of):
    """-> wysokosc `z` rzeczywistego sufitu sceny albo None, jesli sufitu nie ma wcale.

    Liczona jako srednia `z` POZIOMYCH scian klasy `ceiling`, wazona ich polem. Sluzy do
    korekty wysokosci laty sufitowej — patrz komentarz w build_patches(). Wartosci zmierzone
    2026-07-30: w scenach szczelnych sufit lezy 0.02-0.10 m pod szczytem siatki, a w
    frl_apartment_* ocalaly fragment sufitu jest 0.32-0.42 m ponizej szczytu scian. Wysokosc
    pomieszczenia liczona od tego fragmentu wychodzi 2.69-2.74 m, czyli dokladnie tyle, ile
    w scenach szczelnych (2.64-2.87 m) — fragment jest wiec na PRAWDZIWEJ wysokosci sufitu,
    a nadwyzka scian to material powyzej linii sufitu.
    """
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    obj = F["obj"].astype(np.int64)
    labs = np.array([label_of(o) for o in range(int(obj.max()) + 1)])
    mask = labs[obj] == "ceiling"
    if not mask.any():
        return None
    P = xyz[F["i"][mask]]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    area = (ln + np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]), axis=1)) / 2
    horiz = np.abs(nrm[:, 2]) / np.maximum(ln, 1e-12) > 0.8
    if not horiz.any() or area[horiz].sum() <= 0:
        return None
    return float(np.average(P[:, :, 2].mean(1)[horiz], weights=area[horiz]))


def fill_planar(P, C, n, oid, base, col, cell):
    """Wypelnia PLASKI obrys `P` siatka pelnych czworokatow. -> (wierzcholki, sciany)

    DLACZEGO NIE WACHLARZ Z CENTROIDU (pierwsza wersja tego skryptu): zawiodl pomiarowo —
    ucieczka promieni na frl_apartment_2 spadla tylko z 21.79 % do 20.26 % zamiast do zera.
    Dwie przyczyny, obie realne:
      1. trojkat zapisany jako zdegenerowany czworokat (c, u, v, v) nie daje powierzchni,
         ktora liczy sie w tym potoku;
      2. obrys dziury jest NIEWYPUKLY (mieszkanie ze schodami i filarami), a wachlarz
         z jednego punktu pokrywa wielokat tylko wtedy, gdy jest on gwiazdzisty wzgledem
         tego punktu — inaczej zostawia luki i wystaje poza obrys.
    Siatka pelnych czworokatow przycieta testem naleznosci do wielokata nie ma zadnej
    z tych wad; ta sama forma geometrii (grid proper quads) domykala scene do 0.00 %
    w patch_scene_ceiling.py.

    Komorke bierzemy, gdy JEJ SRODEK albo DOWOLNY z 4 rogow lezy wewnatrz obrysu. Sam
    test srodka zostawialby przy krawedzi zabki o glebokosci do `cell`, ktorymi promienie
    dalej by uciekaly; ten wariant nieco wystaje na sciane, co jest nieszkodliwe.
    """
    from matplotlib.path import Path as MplPath

    # baza w plaszczyznie dziury
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(n @ e1) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - n * (n @ e1)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    UV = np.stack([(P - C) @ e1, (P - C) @ e2], 1)
    poly = MplPath(UV)
    lo, hi = UV.min(0) - cell, UV.max(0) + cell
    nu = max(2, int(np.ceil((hi[0] - lo[0]) / cell)) + 1)
    nv = max(2, int(np.ceil((hi[1] - lo[1]) / cell)) + 1)
    us = np.linspace(lo[0], hi[0], nu)
    vs = np.linspace(lo[1], hi[1], nv)
    gu, gv = np.meshgrid(us, vs, indexing="ij")

    inside_node = poly.contains_points(np.stack([gu.ravel(), gv.ravel()], 1)).reshape(nu, nv)
    cu = 0.5 * (us[:-1] + us[1:])
    cv_ = 0.5 * (vs[:-1] + vs[1:])
    mu, mv = np.meshgrid(cu, cv_, indexing="ij")
    inside_centre = poly.contains_points(np.stack([mu.ravel(), mv.ravel()], 1)).reshape(nu - 1, nv - 1)
    corners = (inside_node[:-1, :-1] | inside_node[1:, :-1] |
               inside_node[:-1, 1:] | inside_node[1:, 1:])
    keep = inside_centre | corners
    if not keep.any():
        raise RuntimeError("test naleznosci nie zaakceptowal zadnej komorki — obrys sie nie zamyka?")

    XYZ = C + gu.ravel()[:, None] * e1 + gv.ravel()[:, None] * e2
    NVv = np.zeros(len(XYZ), dtype=VERTEX_DTYPE)
    NVv["x"], NVv["y"], NVv["z"] = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    NVv["nx"], NVv["ny"], NVv["nz"] = n
    NVv["r"], NVv["g"], NVv["b"] = col.astype(np.uint8)

    ii, jj = np.where(keep)
    def vid(i, j):
        return base + i * nv + j
    quad = np.stack([vid(ii, jj), vid(ii, jj + 1),
                     vid(ii + 1, jj + 1), vid(ii + 1, jj)], 1)

    # NAWINIECIE musi dawac normalna geometryczna zgodna z `n`, ktore wolajacy juz
    # obrocil w strone pomieszczenia (patrz build_patches). Sprawdzamy to POMIAREM na
    # pierwszym czworokacie, a nie algebra na bazie (e1, e2) — znak normalnej z SVD
    # jest dowolny i wlasnie na tym sie raz przejechalismy: lata wyszla zwrocona
    # w GORE, renderer odrzucil ja jako tylna i ucieczka promieni nie zmienila sie
    # ANI O JOTE (21.79 % przed i po), mimo poprawnej geometrii w pliku.
    #
    # Wczesniej obchodzilismy to powierzchnia DWUSTRONNA (kazdy czworokat dwa razy,
    # w obu nawinieciach). Akustycznie bylo poprawne — promien trafia w jedna strone,
    # dwie pokrywajace sie sciany zachowuja sie jak jedna — ale POLE liczylo sie
    # podwojnie, co zawyzalo S w Sabine i objetosc sceny z 191 do 396 m3. Stad wersja
    # jednostronna z jawnie sprawdzona orientacja.
    q0 = XYZ[quad[0] - base]
    gn = np.cross(q0[1] - q0[0], q0[2] - q0[0])
    if gn @ n < 0:
        quad = quad[:, ::-1]

    NFf = np.zeros(len(ii), dtype=FACE_DTYPE)
    NFf["cnt"] = 4
    NFf["i"] = quad
    NFf["obj"] = oid
    return NVv, NFf


def cell_for(area):
    """Bok komorki siatki: drobniejszy dla malych dziur (okno), grubszy dla sufitu."""
    return float(np.clip(np.sqrt(area) / 15.0, 0.04, 0.20))


def build_patches(V, F, loops, ba, bb, bf, label_of, min_area, agent_xyz=None,
                  force_class=None, class_pool=None, ceiling_class=None,
                  ceiling_z=None):
    """-> (nowe wierzcholki, nowe sciany, opisy zalatanych dziur)"""
    xyz = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)
    obj = F["obj"].astype(np.int64)
    new_v, new_f, report = [], [], []
    n_added = 0                      # ile wierzcholkow juz doklejono (indeksy sa globalne)

    for ch in loops:
        vids = ba[ch]
        P = xyz[vids]
        area = loop_area(P)
        if area < min_area:
            continue
        C = P.mean(0)
        M = P - C
        n = np.linalg.svd(M, full_matrices=False)[2][-1]
        flat = float(np.abs(M @ n).max())
        # `ceiling_class` dotyczy WYLACZNIE dziur poziomych (brakujacy sufit). Okna,
        # drzwi i urwane krawedzie scian zostaja przy typowaniu semantycznym — material
        # dobrany pomiarowo dotyczy sufitu i tylko do niego wolno go stosowac.
        fc = force_class
        if fc is None and ceiling_class and abs(n[2]) > 0.8:
            fc = ceiling_class
        elif fc is None and abs(n[2]) > 0.8 and class_pool and "ceiling" in class_pool:
            # Lata POZIOMA to sufit, wiec ma dostac klase `ceiling`, a nie `wall`
            # z otoczenia dziury. Akustycznie to NO-OP: w replica_material_config.json
            # obie klasy wskazuja ten sam material (Gypsum Board), wiec ani absorpcja,
            # ani rozpraszanie sie nie zmieniaja. Ma to natomiast znaczenie dla analiz
            # liczacych geometrie per kategoria — rt60_vs_sabine.py wyprowadza rzut
            # poziomy i pokrycie sufitem z kategorii `ceiling`, wiec bez tego lata
            # nie byłaby w tej statystyce widoczna.
            fc = "ceiling"
        oid, cls, hist = classify_loop(ch, bf, obj, label_of, fc, class_pool)
        if not (set(hist) & STRUCTURAL):
            print(f"    POMINIETO dziure {area:.2f} m2 — brzeg bez klasy konstrukcyjnej: {hist}")
            continue

        # Bezpieczenstwo: lata rozpina powierzchnie w plaszczyznie dziury. Jesli pozycje
        # agenta leza po OBU jej stronach, moze przeciac obszar nawigowalny i zamurowac
        # wnetrze sceny.
        sides = None
        if agent_xyz is not None and len(agent_xyz):
            s = np.sign((agent_xyz - C) @ n)
            sides = (int((s > 0).sum()), int((s < 0).sum()))
            if min(sides) > 0:
                print(f"    UWAGA: dziura {area:.2f} m2 ma pozycje agenta po OBU stronach "
                      f"({sides[0]}/{sides[1]}) — lata moze przecinac wnetrze sceny")
            # ORIENTACJA: `n` z SVD ma dowolny znak, wiec obracamy je w strone, w ktorej
            # sa pozycje agenta. Lata jest jednostronna, a niewidoczna od strony
            # pomieszczenia nie domykalaby go wcale (sprawdzone: wtedy ucieczka
            # promieni nie zmienia sie ani o jote).
            if sides[0] < sides[1]:
                n = -n

        # KOREKTA WYSOKOSCI SUFITU. Dziura sufitowa jest ograniczona SZCZYTEM SCIAN, ale
        # w rodzinie frl_apartment_* sciany wystaja 0.32-0.42 m NAD rzeczywisty sufit —
        # zmierzone przez porownanie z ocalalym fragmentem klasy `ceiling`. Kontrola na
        # scenach szczelnych: tam sufit lezy 0.02-0.10 m pod szczytem siatki, czyli
        # praktycznie na nim. Wypelnienie dziury w JEJ plaszczyznie zawyzaloby wiec
        # wysokosc pomieszczenia z 2.71 m do ~3.10 m (+14 % objetosci) i wydluzylo pogłos.
        # Opuszczamy wiec late do wysokosci ocalalego sufitu — NIGDY nie podnosimy.
        z_used = None
        if ceiling_z is not None and abs(n[2]) > 0.8 and ceiling_z < C[2] - 0.05:
            z_used = float(ceiling_z)
            C = np.array([C[0], C[1], z_used])

        col = np.median(np.stack([V["r"][vids], V["g"][vids], V["b"][vids]], 1), 0)
        gv, gf = fill_planar(P, C, n, oid, len(V) + n_added, col, cell_for(area))
        n_added += len(gv)
        new_v.append(gv)
        new_f.append(gf)

        report.append({
            "krawedzi": int(len(ch)), "pole_m2": round(area, 2),
            "centroid": [round(float(v), 2) for v in C],
            "normalna": [round(float(v), 2) for v in n],
            "odchylka_od_plaszczyzny_max_m": round(flat, 3),
            "klasa_laty": cls, "object_id": int(oid), "klasy_przy_brzegu": hist,
            "pozycje_agenta_po_stronach": sides,
            "scian_laty": int(len(gf)), "bok_komorki_m": round(cell_for(area), 3),
            "z_plaszczyzny_dziury": round(float(P[:, 2].mean()), 3),
            "z_uzyte": z_used,
        })

    if not new_v:
        return None, None, report
    return np.concatenate(new_v), np.concatenate(new_f), report


def patch(scene, min_area, force, report_only, force_class=None, suffix="",
          ceiling_class=None):
    src_dir = SCENE_ROOT / scene / "habitat"
    src_ply = src_dir / "mesh_semantic.ply"
    if not src_ply.exists():
        sys.exit(f"brak {src_ply}")

    info = json.loads((src_dir / "info_semantic.json").read_text())
    grav = np.array(info["gravity_dir"], dtype=float)
    if abs(grav[2]) < 0.99:
        sys.exit(f"gravity_dir={grav} — os pionowa NIE jest z, skrypt tego nie obsluguje")
    cls_name = {c["id"]: c["name"] for c in info["classes"]}
    i2l = info["id_to_label"]

    def label_of(oid):
        if oid < 0 or oid >= len(i2l):
            return "??"
        return cls_name.get(i2l[oid], f"cls{i2l[oid]}")

    header, V, F, foff, raw = read_ply(src_ply)
    loops, ba, bb, bf = detect_loops(V, F)
    ceiling_z = ceiling_height(V, F, label_of)
    # Pozycje agenta w ukladzie SUROWEGO PLY: habitat (x, y_gora, z) -> raw (x, -z, y).
    # Zweryfikowane na office_1 promieniem w dol: dystans do podlogi 1.25 m zgadza sie
    # z wysokoscia sensora tylko dla tego przeksztalcenia.
    from echo_core.scenes import load_scene_locations
    _ids, pos = load_scene_locations(scene)
    agent_xyz = np.array([[p[0], -p[2], p[1]] for p in pos.values()], dtype=np.float64)
    # klasa -> dowolny obiekt tej klasy w scenie (dla --patch-class, gdy zadana klasa
    # nie wystepuje przy samym brzegu dziury)
    class_pool = {}
    for o in info["objects"]:
        class_pool.setdefault(cls_name.get(o["class_id"], ""), o["id"])
    NV, NF, rep = build_patches(V, F, loops, ba, bb, bf, label_of, min_area, agent_xyz,
                                force_class, class_pool, ceiling_class, ceiling_z)

    print(f"  {scene}: {len(V)} wierzch., {len(F)} scian, {len(loops)} petli brzegowych")
    if not rep:
        print(f"  brak dziur o polu > {min_area} m2 — scena juz szczelna, nic nie zapisuje")
        return None
    for r in rep:
        print(f"    dziura {r['pole_m2']:>6.2f} m2, {r['krawedzi']:>5} kraw., "
              f"plaskosc max {r['odchylka_od_plaszczyzny_max_m']:.3f} m -> "
              f"lata jako '{r['klasa_laty']}' (object_id={r['object_id']}), "
              f"brzeg: {r['klasy_przy_brzegu']}")
    if report_only:
        return None

    # `suffix` rozdziela warianty sweepu materialowego — kazdy w osobnym katalogu, zeby
    # dalo sie je porownywac bez przelatywania sceny w kolko.
    dst_dir = PATCHED_ROOT / (scene + suffix) / "habitat"
    dst_ply = dst_dir / "mesh_semantic.ply"
    if dst_ply.exists() and not force:
        sys.exit(f"{dst_ply} juz istnieje — uzyj --force")

    new_header = (header
                  .replace(f"element vertex {len(V)}", f"element vertex {len(V) + len(NV)}")
                  .replace(f"element face {len(F)}", f"element face {len(F) + len(NF)}"))
    if new_header == header:
        raise RuntimeError("podmiana licznikow w naglowku nie zadzialala")

    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_ply, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(raw[len(header):foff])      # oryginalne wierzcholki, bajt w bajt
        f.write(NV.tobytes())
        f.write(raw[foff:])                 # oryginalne sciany, bajt w bajt
        f.write(NF.tobytes())
    for name in ("mesh_semantic.navmesh", "info_semantic.json"):
        shutil.copy2(src_dir / name, dst_dir / name)

    _, V2, F2, _, _ = read_ply(dst_ply)
    if len(V2) != len(V) + len(NV) or len(F2) != len(F) + len(NF):
        raise RuntimeError("liczniki po zapisie sie nie zgadzaja")
    if int(F2["i"].max()) >= len(V2):
        raise RuntimeError("indeks wierzcholka poza zakresem po zalataniu")
    if V2[:len(V)].tobytes() != V.tobytes() or F2[:len(F)].tobytes() != F.tobytes():
        raise RuntimeError("oryginalne rekordy zmienily sie przy zapisie")
    print(f"  +{len(NV)} wierzch., +{len(NF)} scian -> {dst_ply} "
          f"({dst_ply.stat().st_size/1e6:.1f} MB), oryginal nienaruszony")

    (dst_dir / "patch_report.json").write_text(json.dumps(
        {"scene": scene, "min_area_m2": min_area, "holes": rep}, indent=2, ensure_ascii=False))
    return dst_ply


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene")
    g.add_argument("--all", action="store_true", help="wszystkie sceny z ucieczka promieni > 0")
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="tylko wypisz dziury, nie zapisuj")
    ap.add_argument("--patch-class", help="wymus klase KAZDEJ laty (sweep materialowy)")
    ap.add_argument("--ceiling-class", help="klasa tylko dla lat POZIOMYCH (sufit), np. rug")
    ap.add_argument("--suffix", default="", help="sufiks katalogu wyjsciowego, np. __rug")
    args = ap.parse_args()

    if args.scene:
        patch(args.scene, args.min_area, args.force, args.report,
              args.patch_class, args.suffix, args.ceiling_class)
        return
    # Sceny nieszczelne wg ray_escape_survey.py — patrz OBSERWACJE_METODOLOGICZNE.md §1.
    leaky = ["frl_apartment_0", "frl_apartment_1", "frl_apartment_2", "frl_apartment_3",
             "frl_apartment_4", "frl_apartment_5",
             "apartment_1", "apartment_2", "office_2", "office_3"]
    for s in leaky:
        patch(s, args.min_area, args.force, args.report,
              args.patch_class, args.suffix, args.ceiling_class)


if __name__ == "__main__":
    main()
