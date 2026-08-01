#!/usr/bin/env python3
"""DOWOD: ile promieni ucieka ze sceny — miara "domkniecia" akustycznego, per lokalizacja.

Pytanie: ktore sceny Replica sa akustycznie zamkniete, a ktore maja dziure w suficie
na tyle duza, ze promienie uciekaja i pogłos jest zanizony? Do tej pory klasyfikacja
opierala sie na heurystyce geometrycznej (udzial powierzchni poziomych w gornych 15 %
wysokosci). Tutaj mierzymy wielkosc, ktora autorzy silnika sami wskazuja jako wlasciwa.

DLACZEGO TA WIELKOSC: SoundSpaces 2.0 (arXiv 2206.08312) pisze wprost, ze siatki musza
byc pozbawione duzych otworow, "otherwise the rays will leak from the holes, resulting in
inaccurate simulation", i udostepnia API do sprawdzenia odsetka uciekajacych promieni.
W naszej kopii biblioteki funkcja istnieje —
  habitat-sim/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/headers/
  RLRAudioPropagation.h:503  RLRA_GetIndirectRayEfficiency()
z dokumentacja: "the fraction of indirect rays that hit geometry (...) can be used as a
measure of how enclosed an acoustic space is. A value close to 0 indicates a very open
space (e.g. outdoors), while a value closer to 1 indicates a closed geometry (e.g.
indoors)". PROBLEM: habitat-sim jej NIE eksponuje — nie ma jej w publicznym API
AudioSensor (src/esp/sensor/AudioSensor.h) ani w bindingach, wiec z Pythona jest
nieosiagalna bez zmiany w C++ i przebudowy rozszerzenia. Mierzymy wiec zastepczo.

DLACZEGO NIE `sim.cast_ray()`: sprawdzone 2026-07-29 — na scenach Replica nie dziala
w ogole. Przy `enable_physics=True` Bullet odrzuca siatke jako collision mesh
("BulletPhysicsManager.cpp:270 isMeshPrimitiveValid : Invalid primitive 0" ->
"Cannot load collision mesh, skipping"), po czym konstrukcja Simulatora konczy sie
AssertionError "Cannot load stage". Powod: mesh_semantic.ply Repliki to czworokaty,
a nie trojkaty. Ta droga jest zamknieta, nie tylko wolniejsza.

DLACZEGO NIE trimesh: w srodowisku `habitat` nie ma ani `rtree`, ani `embreex`, wiec
trimesh.ray nie ma struktury przyspieszajacej (RayMeshIntersector podnosi
ModuleNotFoundError na `triangles_tree`). Doinstalowanie zmienialoby zamrozone
srodowisko z configs/environment-final.yml.

METODA: rownoprostokatny sensor GLEBI w punkcie sluchacza. Jeden render daje pelna
sfere kierunkow (256 x 512 = 131 072 kierunkow, ~0.7 st. na piksel na rowniku).
Piksel o glebi dokladnie 0 to kierunek, w ktorym rasteryzator nie znalazl zadnej
geometrii — czyli promien opuszcza scene. Udzial takich kierunkow, wazony katem
brylowym (cos szerokosci), to dokladnie ulamek promieni uciekajacych przy rozkladzie
izotropowym, a wiec pierwsze odbicie miary RLR.

To pomiar geometrii, ktora WIDZI symulacja audio: dla Replica render-mesh sceny to ten
sam plik mesh_semantic.ply, ktory AudioSensor::loadSemanticMesh() przekazuje do
RLRAudioPropagation (potwierdzone w logu habitata: "Loading Semantic Mesh asset named:
.../mesh_semantic.ply" jako RENDER asset).

OGRANICZENIE, ktore trzeba czytac razem z wynikiem: mierzymy PIERWSZE odbicie z pozycji
sluchacza. Miara RLR liczy promienie wielokrotnie odbite, wiec faktyczny odsetek strat
w symulacji jest WYZSZY — kazde kolejne odbicie to nowa szansa na ucieczke. Nasze
liczby sa dolnym ograniczeniem otwartosci.

Wysokosc sensora i zbior lokalizacji sa te same co produkcyjne (SENSOR_HEIGHT,
load_scene_locations), wiec liczby odnosza sie do tych samych punktow, dla ktorych
generowany jest dataset.

KONTROLA NEGATYWNA jest wbudowana w pomiar: 8 scen daje ucieczke 0.00-0.09 % w KAZDEJ
lokalizacji. Gdyby "glebia == 0" lapala cokolwiek poza brakiem geometrii (artefakt
rasteryzacji, obciecie plaszczyzna blizsza), te sceny nie moglyby wyjsc zerowe. Wynik
dodatni na pozostalych scenach nie jest wiec artefaktem metody.

Wynik (2026-07-29): trzy grupy, nie dwie — 8 scen szczelnych (516 lok.), 4 nieszczelne
bokiem (417 lok.), 6 bez sufitu (807 lok.). Pelne liczby:
outputs/measurements/ray_escape/summary.csv, omowienie:
docs/OBSERWACJE_METODOLOGICZNE.md §1 ("Uscislenie 2026-07-29").

Uruchomienie — JEDNA SCENA NA PROCES (limit ~30 konstrukcji Simulatora na proces);
skrypt konstruuje dokladnie jeden:
    python my-operations/measurements/ray_escape_survey.py --scene office_1
    python my-operations/measurements/ray_escape_survey.py --all      # petla po procesach
    python my-operations/measurements/ray_escape_survey.py --summary  # bez GPU
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import quaternion  # noqa: F401  (wymagane PRZED habitat_sim,
                   #              patrz habitat-sim/local_changes.patch)
import habitat_sim

from echo_core.params import SENSOR_HEIGHT
from echo_core.paths import REPO_ROOT, scene_mesh
from echo_core.scenes import SCENE_ORDER, load_scene_locations

OUT_DIR = REPO_ROOT / "outputs/measurements/ray_escape"

# Rozdzielczosc sfery. 256x512 to 131 072 kierunkow; blad dyskretyzacji jest
# deterministyczny (rasteryzacja, nie Monte Carlo), wiec nie ma sensu zwiekszac —
# roznica miedzy scena zamknieta (0 %) a otwarta (>10 %) jest o rzedy wielkosci
# wieksza niz ziarnistosc siatki.
EQ_H, EQ_W = 256, 512

# Progi pasm wysokosci (elewacja w stopniach) do rozbicia ucieczek. Wiersz 0
# odpowiada gorze — zweryfikowane empirycznie: na frl_apartment_0 wszystkie
# brakujace piksele leza w gornych wierszach, a dolna polkula ma dokladnie 0.0 %.
ELEV_BANDS = ((60, 90), (30, 60), (10, 30), (-10, 10), (-90, -10))


def elevations():
    """-> (H,) elewacja srodka kazdego wiersza w stopniach, malejaca od +90."""
    return 90.0 - (np.arange(EQ_H) + 0.5) / EQ_H * 180.0


def solid_angle_weights():
    """-> (H, W) waga kata brylowego piksela.

    W odwzorowaniu rownoprostokatnym wiersz przy biegunie reprezentuje mniejszy
    kat brylowy niz przy rowniku — bez tej wagi dziura w suficie liczylaby sie
    tyle samo, co rowna jej powierzchniowo dziura przy horyzoncie, a to nie jest
    ulamek promieni izotropowych.
    """
    w = np.cos(np.deg2rad(elevations()))
    return np.repeat(w[:, None], EQ_W, axis=1)


def build_probe_simulator(scene, mesh=None):
    """Simulator z JEDNYM sensorem: rownoprostokatna glebia na wysokosci sluchacza.

    Osobna konfiguracja, a nie echo_core.audio.build_simulator(), bo tamta jest
    sciezka skalibrowana (kolejnosc wywolan audio = sekwencja RNG) i nie wolno jej
    ruszac dla pomiaru. Tutaj nie ma sensora audio w ogole — mierzymy geometrie.

    `mesh` pozwala wskazac inna siatke niz produkcyjna przy tych samych lokalizacjach —
    uzywane do kontroli scen zalatanych sufitem (patch_scene_ceiling.py).
    """
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(mesh or scene_mesh(scene))
    # load_semantic_mesh=False: dla Replica render-mesh TO JEST mesh_semantic.ply
    # (patrz naglowek), wiec geometria jest identyczna, a nie ladujemy jej drugi raz.
    cfg.load_semantic_mesh = False
    cfg.enable_physics = False
    cfg.create_renderer = True
    cfg.gpu_device_id = 0

    eq = habitat_sim.EquirectangularSensorSpec()
    eq.uuid = "eq_depth"
    eq.sensor_type = habitat_sim.SensorType.DEPTH
    eq.resolution = [EQ_H, EQ_W]
    eq.position = [0.0, SENSOR_HEIGHT, 0.0]
    # far jawnie: piksel bez trafienia ma byc rozpoznany po glebi 0, wiec plaszczyzna
    # dalsza nie moze obcinac prawdziwej geometrii i robic z niej falszywej "ucieczki".
    eq.far = 1000.0

    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [eq]
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


def survey_scene(scene, mesh=None, tag=None):
    loc_ids, positions = load_scene_locations(scene)
    sim = build_probe_simulator(scene, mesh)
    try:
        weights = solid_angle_weights()
        w_total = weights.sum()
        elev = elevations()
        band_masks = [(elev >= lo) & (elev < hi) for lo, hi in ELEV_BANDS]
        # waga kazdego pasma w pelnej sferze — mianownik dla udzialu WEWNATRZ pasma
        band_w = [weights[m].sum() for m in band_masks]

        rows = []
        agent = sim.get_agent(0)
        for lid in loc_ids:
            state = agent.get_state()
            state.position = positions[lid]
            agent.set_state(state)
            depth = np.asarray(sim.get_sensor_observations()["eq_depth"])
            if depth.shape != (EQ_H, EQ_W):
                raise RuntimeError(f"{scene}/{lid}: glebia ma ksztalt {depth.shape}")
            miss = depth == 0.0

            row = {
                "scene": scene,
                "location_id": lid,
                # udzial kierunkow bez trafienia, wazony katem brylowym = ulamek
                # uciekajacych promieni izotropowych (pierwsze odbicie)
                "escape": float((miss * weights).sum() / w_total),
                # to samo bez wagi — surowy udzial pikseli, zostawiony dla kontroli
                "escape_unweighted": float(miss.mean()),
                "max_depth": float(depth.max()),
            }
            for (lo, hi), m, bw in zip(ELEV_BANDS, band_masks, band_w):
                row[f"esc_{lo}_{hi}"] = float((miss[m] * weights[m]).sum() / bw)
            rows.append(row)
    finally:
        sim.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Wariant oznaczony tagiem trafia do osobnego pliku i jest pomijany przez
    # --summary: zestawienie ma opisywac dataset produkcyjny, a nie eksperymenty.
    out = OUT_DIR / (f"{scene}__{tag}.csv" if tag else f"{scene}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    esc = np.array([r["escape"] for r in rows])
    print(f"  {scene:<18} n={len(rows):>4}  ucieczka: mediana {np.median(esc)*100:6.2f} %  "
          f"zakres {esc.min()*100:5.2f}-{esc.max()*100:5.2f} %  -> {out.name}")
    return rows


SEALED = "szczelna"
LATERAL = "nieszczelna bokiem"
NO_CEILING = "BEZ SUFITU"


def classify(esc, bands):
    """Trojpodzial scen na podstawie ROZKLADU KATOWEGO ucieczek, nie samej wartosci.

    Pierwsza wersja tego skryptu dzielila sceny jednym progiem na medianie (1 %),
    z komentarzem, ze rozdzielenie jest tak ostre, ze prog nie ma znaczenia. Pomiar
    to OBALIL: sceny ukladaja sie w trzy grupy, a nie dwie, i granica miedzy druga
    a trzecia faktycznie zalezy od progu. Rozroznia je natomiast jednoznacznie to,
    GDZIE lezy dziura:

      * BEZ SUFITU — ucieczka niemal wylacznie NAD horyzontem: pasmo >60 st. ma
        99-100 %, a pasma przy horyzoncie i ponizej dokladnie 0.0 %. Tak wyglada
        brakujaca plaszczyzna sufitu. Dotyczy wszystkich frl_apartment_*.
      * nieszczelna bokiem — pasmo >60 st. ma 0.0 %, a ucieczka siedzi w pasmach
        10-30 st. i -10..10 st. To otwory w PIONIE: przejscia, okna, niezeskanowane
        fragmenty scian, krawedz sceny. Sufit jest na miejscu — dodanie go nic tu
        nie da.
      * szczelna — ucieczka ponizej 0.1 % w KAZDEJ lokalizacji.

    Prog 0.5 na pasmie >60 st. jest bezpieczny, bo ta wielkosc przyjmuje wartosci
    albo 0.000, albo 0.99-1.00 — nie ma scen posrednich.
    """
    if bands[0] > 0.5:                       # pasmo >60 st.
        return NO_CEILING
    if np.median(esc) < 0.001 and esc.max() < 0.005:
        return SEALED
    return LATERAL


def summary():
    """Zestawienie ze wszystkich CSV — bez GPU, mozna puscic w drugim terminalu."""
    files = sorted(OUT_DIR.glob("*.csv"))
    files = [f for f in files if f.name != "summary.csv" and "__" not in f.name]
    if not files:
        sys.exit(f"brak wynikow w {OUT_DIR} — najpierw --all")

    per_scene = {}
    for f in files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        per_scene[f.stem] = rows

    print(f"\n  Udzial promieni uciekajacych ze sceny (pierwsze odbicie, waga kata brylowego)")
    print(f"  {'scena':<18}{'n':>5}{'mediana':>9}{'p90':>8}{'max':>8}{'lok>10%':>7}   "
          f"{'>60st':>7}{'30-60':>7}{'10-30':>7}{'-10..10':>8}{'dol':>7}")
    print("  " + "-" * 100)

    order = [s for s in SCENE_ORDER if s in per_scene]
    order += [s for s in sorted(per_scene) if s not in order]
    summary_rows = []
    for scene in order:
        rows = per_scene[scene]
        esc = np.array([float(r["escape"]) for r in rows])
        bands = [np.median([float(r[f"esc_{lo}_{hi}"]) for r in rows])
                 for lo, hi in ELEV_BANDS]
        status = classify(esc, bands)
        print(f"  {scene:<18}{len(rows):>5}{np.median(esc)*100:8.2f}%"
              f"{np.percentile(esc, 90)*100:7.2f}%{esc.max()*100:7.2f}%"
              f"{np.mean(esc > 0.10)*100:6.0f}%   "
              + "".join(f"{b*100:6.1f}%" for b in bands[:3])
              + f"{bands[3]*100:7.1f}%{bands[4]*100:6.1f}%  {status}")
        summary_rows.append({
            "scene": scene, "n_locations": len(rows),
            "escape_median": round(float(np.median(esc)), 6),
            "escape_p90": round(float(np.percentile(esc, 90)), 6),
            "escape_max": round(float(esc.max()), 6),
            "frac_loc_over_10pct": round(float(np.mean(esc > 0.10)), 4),
            **{f"band_{lo}_{hi}_median": round(float(b), 6)
               for (lo, hi), b in zip(ELEV_BANDS, bands)},
            "class": status,
        })

    print("  " + "-" * 100)
    total = sum(r["n_locations"] for r in summary_rows)
    for cls in (SEALED, LATERAL, NO_CEILING):
        grp = [r for r in summary_rows if r["class"] == cls]
        n = sum(r["n_locations"] for r in grp)
        print(f"  {cls:<20} {len(grp):>2} scen, {n:>4} lok. ({n/total*100:4.1f} %)  "
              f"-> {', '.join(r['scene'] for r in grp)}")

    out = OUT_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  zapisano: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="jedna scena (ten proces konstruuje 1 Simulator)")
    g.add_argument("--all", action="store_true",
                   help="wszystkie sceny, kazda w OSOBNYM procesie potomnym")
    g.add_argument("--summary", action="store_true", help="zestawienie z CSV, bez GPU")
    ap.add_argument("--mesh", help="inna siatka niz produkcyjna (np. scena zalatana sufitem)")
    ap.add_argument("--tag", help="sufiks pliku wynikowego; wymagany razem z --mesh")
    args = ap.parse_args()

    if args.summary:
        return summary()
    if args.scene:
        if bool(args.mesh) != bool(args.tag):
            # Bez tagu wynik z innej siatki nadpisalby pomiar produkcyjny tej sceny.
            ap.error("--mesh i --tag trzeba podac razem")
        return survey_scene(args.scene, args.mesh, args.tag)

    # --all: osobny proces na scene. Nie petla w jednym procesie, bo konstrukcja
    # i niszczenie wielu Simulatorow w jednym procesie potrafi zawiesic GPU
    # (przeciek zasobow EGL/GL na sim.close()).
    print(f"  {len(SCENE_ORDER)} scen, kazda w osobnym procesie\n")
    failed = []
    for scene in SCENE_ORDER:
        r = subprocess.run([sys.executable, __file__, "--scene", scene],
                           cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            failed.append(scene)
            print(f"  {scene:<18} BLAD (kod {r.returncode})")
            print("    " + "\n    ".join(r.stderr.strip().splitlines()[-6:]))
        else:
            print(r.stdout.rstrip())
    if failed:
        # Nie polykamy bledu po cichu — brak sceny w zestawieniu musi byc widoczny.
        sys.exit(f"\n  NIEPOWODZENIE dla: {', '.join(failed)}")
    summary()


if __name__ == "__main__":
    main()
