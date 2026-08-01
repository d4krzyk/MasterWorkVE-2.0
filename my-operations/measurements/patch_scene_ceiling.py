#!/usr/bin/env python3
"""EKSPERYMENT: domkniecie sceny Replica sufitem — czy echa zbliza sie do SoundSpaces 1.0?

ZASTAPIONY przez patch_scene_holes.py, ktory laduje WSZYSTKIE dziury (nie tylko sufit),
w plaszczyznie faktycznego otworu i z materialem wg jego typu. Ten skrypt zostaje, bo to
na nim policzony jest wynik z RAPORT_SESJI §2.12 (plaszczyzna na wysokosci istniejacego
fragmentu sufitu, caly rzut bboxa, material `ceiling`) — bez niego tamta liczba nie jest
odtwarzalna. Do nowych pomiarow uzywaj patch_scene_holes.py.


Pytanie: sceny `frl_apartment_*` nie maja sufitu w siatce, przez co ~23 % promieni ucieka
gora, a RT60 wychodzi 2.5x krotszy niz w prekomputowanych RIR-ach SoundSpaces 1.0
(frl_apartment_2 @ 1 kHz: nasze 0.186 s wobec ich 0.463 s; na scenie ZAMKNIETEJ office_1
oba silniki zgadzaja sie w ~10 %). Wiodaca hipoteza brzmiala, ze potok SS 1.0 domykal
objetosc przed symulacja. Ten skrypt buduje zalatana wersje sceny, zeby hipoteze
przetestowac zamiast o niej spekulowac.

CO ROBI: dokleja do `mesh_semantic.ply` plaszczyzne sufitu na wysokosci rzeczywistego
sufitu sceny, pokrywajaca caly rzut poziomy, i zapisuje wynik jako OSOBNA scene w
`outputs/patched_scenes/`. Oryginalny dataset nie jest dotykany.

FORMAT (rozpoznany 2026-07-29, naglowek "replica-instance-mesh-format v0"):
  * binary_little_endian; wierzcholek = x,y,z,nx,ny,nz (float32) + r,g,b (uint8) = 27 B
  * sciana = uint8 licznik (ZAWSZE 4 — same czworokaty) + 4x uint32 indeks + uint16 object_id = 19 B
  * `gravity_dir` w info_semantic.json to [~0, ~0, -1], wiec os PIONOWA to **z**, a nie y.
    (habitat obraca scene do y-up dopiero przy ladowaniu — nie mylic ukladow.)
  * `id_to_label[object_id]` daje class_id; dla frl_apartment_2 `id_to_label[131] = 31`,
    czyli klasa "ceiling".

DLACZEGO object_id ISTNIEJACEGO SUFITU, a nie nowy: nowe sciany dostaja `object_id` obiektu
klasy "ceiling", ktory w scenie juz jest (fragment sufitu w rogu). Dzieki temu
info_semantic.json NIE wymaga zmiany, a `setAudioMaterialsJSON()` przypisze im material
"ceiling" — obecny w replica_material_config.json. Gdyby uzyc nowego id, trzeba by dopisac
obiekt do info_semantic.json i wpis do id_to_label, a kazda niespojnosc miedzy PLY a JSON-em
grozi tym samym SIGSEGV w loadSemanticMesh(), ktory naprawia habitat-sim/local_changes.patch.

WYSOKOSC SUFITU bierzemy z mediany `z` istniejacego fragmentu sufitu, a nie z gornej krawedzi
bboxa: bbox siega czubkow scian (u frl_apartment_2 z = 1.669 wobec sufitu 1.294), wiec
plaszczyzna na jego wysokosci zawyzylaby objetosc o ~13 % i zafalszowala RT60, ktory
wlasnie mierzymy.

ZAPIS jest bajtowo zachowawczy: oryginalne rekordy wierzcholkow i scian sa przepisywane
BEZ dekodowania i ponownego kodowania, a nowe DOKLEJANE na koncu kazdego bloku. Indeksy
w oryginalnych scianach pozostaja wiec poprawne (nowe wierzcholki maja indeksy >= nv).

Uruchomienie:
    python my-operations/measurements/patch_scene_ceiling.py --scene frl_apartment_2
    python my-operations/measurements/ray_escape_survey.py --scene frl_apartment_2 \
        --mesh outputs/patched_scenes/frl_apartment_2/habitat/mesh_semantic.ply   # kontrola
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo_core.paths import REPO_ROOT, SCENE_ROOT

PATCHED_ROOT = REPO_ROOT / "outputs/patched_scenes"

VERTEX_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                         ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                         ("r", "u1"), ("g", "u1"), ("b", "u1")])
FACE_DTYPE = np.dtype([("cnt", "u1"), ("i", "<u4", 4), ("obj", "<u2")])


def read_ply(path):
    """-> (naglowek_bytes, V, F, offset_scian, surowe_bajty)"""
    raw = path.read_bytes()
    marker = b"end_header\n"
    hdr_end = raw.index(marker) + len(marker)
    header = raw[:hdr_end].decode("ascii")
    lines = header.splitlines()

    def count(elem):
        hit = [l for l in lines if l.startswith(f"element {elem}")]
        if len(hit) != 1:
            raise RuntimeError(f"{path.name}: oczekiwano 1 wpisu 'element {elem}', jest {len(hit)}")
        return int(hit[0].split()[-1])

    nv, nf = count("vertex"), count("face")
    if VERTEX_DTYPE.itemsize * nv + FACE_DTYPE.itemsize * nf != len(raw) - hdr_end:
        raise RuntimeError(
            f"{path.name}: rozmiar pliku nie zgadza sie z zalozonym formatem "
            f"(27 B/wierzcholek, 19 B/sciana-czworokat). Plik ma inny uklad wlasciwosci "
            f"albo sciany nie sa czworokatami — patrz naglowek:\n{header}")

    V = np.frombuffer(raw, dtype=VERTEX_DTYPE, count=nv, offset=hdr_end)
    foff = hdr_end + nv * VERTEX_DTYPE.itemsize
    F = np.frombuffer(raw, dtype=FACE_DTYPE, count=nf, offset=foff)
    if not np.all(F["cnt"] == 4):
        raise RuntimeError(f"{path.name}: nie wszystkie sciany sa czworokatami")
    return header, V, F, foff, raw


def ceiling_object_id(info):
    """-> object_id istniejacego obiektu klasy 'ceiling' (dowolnego, jesli jest ich wiele)."""
    names = {c["id"]: c["name"] for c in info["classes"]}
    hits = [o["id"] for o in info["objects"] if names.get(o["class_id"]) == "ceiling"]
    if not hits:
        raise RuntimeError(
            "scena nie ma ZADNEGO obiektu klasy 'ceiling' — nie ma czego uzyc jako object_id "
            "bez modyfikacji info_semantic.json (patrz naglowek skryptu)")
    return hits


def build_ceiling(V, F, obj_ids, cell):
    """-> (nowe_wierzcholki, nowe_sciany, z_sufitu, opis)

    Plaszczyzna na wysokosci mediany `z` istniejacego sufitu, pokrywajaca caly rzut xy,
    podzielona na siatke o boku `cell` (podzial nie zmienia geometrii, ale utrzymuje
    rozsadne rozmiary trojkatow dla ray tracera i modelu rozpraszania).
    """
    xyz = np.stack([V["x"], V["y"], V["z"]], 1)
    mask = np.isin(F["obj"], obj_ids)
    if not mask.any():
        raise RuntimeError(f"object_id {obj_ids} nie wystepuje w zadnej scianie PLY")
    ceil_idx = np.unique(F["i"][mask].ravel())
    ceil_pts = xyz[ceil_idx]
    z_ceiling = float(np.median(ceil_pts[:, 2]))

    # Kolor z istniejacego sufitu — nowe sciany maja wygladac jak reszta sufitu,
    # gdyby scena byla kiedykolwiek renderowana wizualnie.
    col = np.median(np.stack([V["r"][ceil_idx], V["g"][ceil_idx], V["b"][ceil_idx]], 1), 0)

    lo, hi = xyz[:, :2].min(0), xyz[:, :2].max(0)
    nx = max(2, int(np.ceil((hi[0] - lo[0]) / cell)) + 1)
    ny = max(2, int(np.ceil((hi[1] - lo[1]) / cell)) + 1)
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    nvnew = nx * ny
    NV = np.zeros(nvnew, dtype=VERTEX_DTYPE)
    NV["x"] = gx.ravel()
    NV["y"] = gy.ravel()
    NV["z"] = z_ceiling
    # normalna skierowana W DOL, czyli do wnetrza pomieszczenia
    NV["nx"], NV["ny"], NV["nz"] = 0.0, 0.0, -1.0
    NV["r"], NV["g"], NV["b"] = col.astype(np.uint8)

    base = len(V)
    i0, j0 = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    i0, j0 = i0.ravel(), j0.ravel()

    def vid(i, j):
        return base + i * ny + j

    # kolejnosc wierzcholkow zgodna z normalna -z (CCW patrzac od dolu)
    NF = np.zeros(len(i0), dtype=FACE_DTYPE)
    NF["cnt"] = 4
    NF["i"] = np.stack([vid(i0, j0), vid(i0, j0 + 1),
                        vid(i0 + 1, j0 + 1), vid(i0 + 1, j0)], 1)
    NF["obj"] = obj_ids[0]

    desc = (f"z_sufitu={z_ceiling:.3f} (mediana z {mask.sum()} scian sufitu), "
            f"rzut {hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} m, siatka {nx}x{ny} "
            f"(bok {cell} m), +{nvnew} wierzcholkow, +{len(NF)} scian, "
            f"object_id={obj_ids[0]}, kolor={tuple(col.astype(int))}")
    return NV, NF, z_ceiling, desc


def patch(scene, cell, force):
    src_dir = SCENE_ROOT / scene / "habitat"
    src_ply = src_dir / "mesh_semantic.ply"
    if not src_ply.exists():
        sys.exit(f"brak {src_ply}")
    dst_dir = PATCHED_ROOT / scene / "habitat"
    dst_ply = dst_dir / "mesh_semantic.ply"
    if dst_ply.exists() and not force:
        sys.exit(f"{dst_ply} juz istnieje — uzyj --force")

    info = json.loads((src_dir / "info_semantic.json").read_text())
    grav = np.array(info["gravity_dir"], dtype=float)
    # Zabezpieczenie przed cicha pomylka ukladu: caly skrypt zaklada, ze pionem jest z.
    if abs(grav[2]) < 0.99:
        sys.exit(f"gravity_dir={grav} — os pionowa NIE jest z, skrypt tego nie obsluguje")

    header, V, F, foff, raw = read_ply(src_ply)
    obj_ids = ceiling_object_id(info)
    NV, NF, z_ceiling, desc = build_ceiling(V, F, np.array(obj_ids), cell)
    print(f"  {scene}: {len(V)} wierzch., {len(F)} scian")
    print(f"  sufit: {desc}")

    new_header = (header
                  .replace(f"element vertex {len(V)}", f"element vertex {len(V) + len(NV)}")
                  .replace(f"element face {len(F)}", f"element face {len(F) + len(NF)}"))
    if new_header == header:
        raise RuntimeError("podmiana licznikow w naglowku nie zadzialala")

    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_ply, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(raw[len(header):foff])       # oryginalne wierzcholki, bajt w bajt
        f.write(NV.tobytes())
        f.write(raw[foff:])                  # oryginalne sciany, bajt w bajt
        f.write(NF.tobytes())

    # navmesh MUSI byc kopia oryginalu — zbior lokalizacji ma zostac identyczny,
    # inaczej porownanie zalatana/oryginalna przestaje dotyczyc tych samych punktow.
    for name in ("mesh_semantic.navmesh", "info_semantic.json"):
        shutil.copy2(src_dir / name, dst_dir / name)

    size_mb = dst_ply.stat().st_size / 1e6
    print(f"  zapisano: {dst_ply}  ({size_mb:.1f} MB)")

    # kontrola: odczytaj z powrotem i sprawdz spojnosc
    h2, V2, F2, _, _ = read_ply(dst_ply)
    assert len(V2) == len(V) + len(NV) and len(F2) == len(F) + len(NF)
    if F2["i"].max() >= len(V2):
        raise RuntimeError("indeks wierzcholka poza zakresem po zalataniu")
    same_v = np.array_equal(V2[:len(V)].tobytes(), V.tobytes())
    same_f = np.array_equal(F2[:len(F)].tobytes(), F.tobytes())
    print(f"  kontrola odczytu: wierzcholki {len(V2)}, sciany {len(F2)}, "
          f"oryginal nienaruszony: wierzch. {same_v}, sciany {same_f}")
    if not (same_v and same_f):
        raise RuntimeError("oryginalne rekordy zmienily sie przy zapisie")
    return dst_ply


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--cell", type=float, default=0.5, help="bok komorki siatki sufitu [m]")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    patch(args.scene, args.cell, args.force)


if __name__ == "__main__":
    main()
