#!/usr/bin/env python3
"""Generuje `my-operations/replica_material_config.json` - mapowanie kategorii
semantycznych Repliki na materialy akustyczne RLRAudioPropagation.

DLACZEGO generator, a nie recznie napisany JSON: bloki fizyczne materialow
(absorption/scattering/transmission/damping/density/speed) sa dokladnie te same,
co w `sound-spaces/data/mp3d_material_config.json` - to jest baza materialow
dostarczona przez autorow RLRAudioPropagation i nie mamy podstaw jej zmieniac.
Zmieniamy WYLACZNIE pole `labels`, czyli przypisanie kategoria -> material.
Przepisywanie 30 blokow po ~40 liczb recznie tylko zwiekszaloby ryzyko literowki.

DLACZEGO potrzebny jest walidator: RLRAudioPropagation NIE dopasowuje kategorii
przez rownosc napisow. Wg `RLRAudioPropagation.h` (opis RLRA_SetMaterialDatabaseJSON):

    "A material is determined from a material category string by inspecting all
     materials in the database, and finding the material which has the greatest
     number of label substring matches. A match is counted if the lowercase
     category name contains a label as a substring."

Konsekwencje, ktore realnie wystepuja na Replice:
  - etykieta "wall" pasuje takze do kategorii "wall-cabinet" i "wall-plug",
  - etykieta "bin" jest podlancuchem "ca-BIN-et",
  - etykieta "table" jest podlancuchem "tablet",
  - etykieta "pan" jest podlancuchem "panel",
  - etykieta "door" jest podlancuchem "in-DOOR-plant".
Przy remisie liczby dopasowan API nie precyzuje, ktory material wygrywa - czyli
wynik jest nieokreslony. Dlatego skrypt po zbudowaniu mapowania SYMULUJE regule
dopasowania na wszystkich kategoriach faktycznie wystepujacych w 18 scenach i
rozwiazuje kazdy remis, duplikujac etykiete bedaca doslowna nazwa kategorii
(duplikaty licza sie podwojnie - tej samej sztuczki uzyto w oryginalnym
mp3d_material_config.json, gdzie "floor" wystepuje dwa razy w materiale Carpet).
Zasada: doslowna nazwa kategorii zawsze bije przypadkowy podlancuch.

Weryfikacja samej reguly: symulacja odtwarza co do jednej liste 11 kategorii,
dla ktorych habitat-sim loguje "Material for category 'X' was not found" na
scenie room_0 z configiem mp3d (basket, book, candle, lamp, picture, pillar,
plate, pot, switch, vase, vent).

Uruchomienie:  python my-operations/make_replica_material_config.py
"""

import collections
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MP3D_CONFIG = REPO_ROOT / "sound-spaces/data/mp3d_material_config.json"
REPLICA_ROOT = REPO_ROOT / "sound-spaces/data/scene_datasets/replica"
OUT_JSON = REPO_ROOT / "my-operations/replica_material_config.json"

# ---------------------------------------------------------------------------
# MAPOWANIE: material RLR -> kategorie semantyczne Repliki.
#
# Kolejnosc grup mniej wiecej wg udzialu w POWIERZCHNI sceny (patrz
# my-operations/replica_category_area.json i replica_semantic_area.py) - to
# powierzchnia, a nie liczba obiektow, decyduje o wplywie na pogłos.
# Pelne uzasadnienie kazdej nietrywialnej decyzji: REPLICA_MATERIALS.md
# ---------------------------------------------------------------------------
MAPPING = {
    # Sciany i sufity: we wszystkich 18 scenach gladki, malowany tynk/plyta g-k.
    # Sufit CELOWO nie jest "Acoustic Tile" (jak w configu mp3d) - ogledziny
    # gornego pasa kadru we wszystkich 18 scenach nie pokazuja nigdzie rastru
    # plyt akustycznych, takze w biurach. Roznica jest ogromna: Acoustic Tile
    # pochlania 0.60 przy 500 Hz, Gypsum Board 0.05 - dwunastokrotnie mniej.
    "Gypsum Board": ["wall", "ceiling"],

    # Podloga: rodzina apartamentow (apartment_0..2 + frl_apartment_0..5) to
    # 72% calej powierzchni Repliki i ma jasne deski drewniane (widoczne sloje
    # w dolnym pasie kadru). Miekkie pokrycie jest w Replice modelowane OSOBNO
    # jako kategorie "rug" (144.7 m2, 13 scen) i "mat" - przypisanie samej
    # podlogi do dywanu liczyloby te absorpcje drugi raz.
    "Wood Floor": ["floor", "stair"],
    "Carpet, Heavy": ["rug"],
    "Carpet": ["mat"],

    # Porowate tkaniny - jedyny material o wysokiej absorpcji srednich i
    # wysokich czestotliwosci dostepny w bazie RLR.
    # "blinds" CELOWO tutaj, a nie w Glass (jak w mp3d): rolety i zaluzje to
    # tkanina albo lamele, nie szyba. To 5.01% powierzchni Repliki - trzecia
    # najwieksza kategoria po scianach/podlodze/suficie.
    "Curtain": ["curtain", "blinds", "sofa", "bed", "comforter", "blanket",
                "pillow", "cushion", "beanbag", "clothing", "cloth", "scarf",
                "towel", "handbag", "bag", "shoe", "umbrella", "lamp"],

    # Lite meble drewniane.
    "wood, Thick": ["door", "chair", "table", "desk", "shelf", "cabinet",
                    "wall-cabinet", "base-cabinet", "nightstand", "countertop",
                    "stool", "bench", "tv-stand", "plant-stand", "rack", "book",
                    "chopping-board", "desk-organizer", "utensil-holder",
                    "knife-block"],

    # Cienkie, puste w srodku skorupy: plastik, karton, wiklina. Akustycznie
    # zachowuja sie jak plyta rezonansowa (wysoka absorpcja nisko, niska wysoko),
    # a nie jak lita bryla.
    "wood, Thin": ["bin", "box", "basket", "panel", "switch", "wall-plug",
                   "tissue-paper", "coaster", "tablet", "remote-control",
                   "clock", "camera"],

    "Glass": ["window", "tv-screen", "monitor", "picture", "bottle"],

    "Steel": ["handrail", "refrigerator", "sink", "faucet", "pipe", "vent",
              "bike", "cooktop", "pan", "major-appliance", "small-appliance",
              "kitchen-utensil"],

    "Tile, Ceramic": ["shower-stall", "toilet", "bathtub", "bowl", "plate",
                      "cup", "vase", "pot", "sculpture", "candle"],

    "Foliage": ["indoor-plant"],

    # Slupy konstrukcyjne - beton/mur, akustycznie twarde i odbijajace.
    "Concrete": ["pillar"],
}


def load_replica_categories():
    """Kategorie faktycznie wystepujace w 18 scenach + ich powierzchnia."""
    area_file = REPO_ROOT / "my-operations/replica_category_area.json"
    if area_file.exists():
        inv = json.loads(area_file.read_text())
        return inv["area"], inv["total_area"]
    # awaryjnie: bez pola, sama lista kategorii
    cats = collections.Counter()
    for sd in sorted(REPLICA_ROOT.iterdir()):
        f = sd / "habitat/info_semantic.json"
        if not f.exists():
            continue
        for o in json.loads(f.read_text())["objects"]:
            cats["<class_id=-1>" if o["class_id"] == -1 else o["class_name"]] += 1
    return dict(cats), float(sum(cats.values()))


def winners(labels_by_material, category):
    """Symulacja reguly RLR: material z najwieksza liczba etykiet bedacych
    podlancuchami nazwy kategorii. Zwraca (liczba dopasowan, lista zwyciezcow)."""
    c = category.lower()
    best, hits = 0, []
    for mat, labels in labels_by_material.items():
        k = sum(1 for l in labels if l and l.lower() in c)
        if k > best:
            best, hits = k, [mat]
        elif k == best and k > 0:
            hits.append(mat)
    return best, hits


def resolve_ties(labels_by_material, categories):
    """Duplikuje doslowna nazwe kategorii tam, gdzie regula RLR daje remis albo
    wskazuje material inny niz zamierzony. Zwraca liste opisow interwencji."""
    intended = {cat: mat for mat, cats in MAPPING.items() for cat in cats}
    fixes = []
    for _ in range(4):  # kilka przebiegow: naprawa jednej kategorii moze odslonic kolejna
        changed = False
        for cat in categories:
            if cat.startswith("<") or cat not in intended:
                continue
            want = intended[cat]
            k, got = winners(labels_by_material, cat)
            if got == [want]:
                continue
            # Podbijamy wage doslownej nazwy kategorii az przebije konkurencje.
            others = max((sum(1 for l in labels_by_material[m] if l.lower() in cat.lower())
                          for m in labels_by_material if m != want), default=0)
            need = others + 1
            have = sum(1 for l in labels_by_material[want] if l.lower() in cat.lower())
            while have < need:
                labels_by_material[want].append(cat)
                have += 1
                changed = True
            fixes.append({"category": cat, "material": want, "conflicting": [m for m in got if m != want],
                          "label_weight": have})
        if not changed:
            break
    return fixes


def main():
    mp3d = json.loads(MP3D_CONFIG.read_text())
    known = {m["name"] for m in mp3d["materials"]}
    unknown = set(MAPPING) - known
    if unknown:
        sys.exit(f"Nieznane nazwy materialow (nie ma ich w bazie RLR): {sorted(unknown)}")

    # Kopiujemy bloki fizyczne 1:1, podmieniamy tylko `labels`.
    labels_by_material = {name: list(MAPPING.get(name, [])) for name in known}
    labels_by_material["Default"] = ["default"]  # material domyslny musi zostac

    area, total = load_replica_categories()
    categories = sorted(area)
    fixes = resolve_ties(labels_by_material, categories)

    out = {"materials": []}
    for m in mp3d["materials"]:
        m2 = dict(m)
        m2["labels"] = labels_by_material[m["name"]]
        out["materials"].append(m2)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    # --- raport ---
    print("KOLIZJE PODLANCUCHOW rozwiazane przez powielenie doslownej nazwy kategorii:")
    if not fixes:
        print("  (brak)")
    for f in fixes:
        print(f"  {f['category']:<16} -> {f['material']:<14} (kolidowalo z {f['conflicting']}, "
              f"waga etykiety {f['label_weight']})")

    covered = default_ = 0.0
    unmapped = []
    print(f"\n{'kategoria':<22}{'% pola':>8}  material")
    for cat, a in sorted(area.items(), key=lambda kv: -kv[1]):
        pct = a / total * 100
        if cat.startswith("<"):
            default_ += a
            print(f"{cat:<22}{pct:>7.2f}%  <- ZAWSZE domyslny (brak category() w habitat-sim)")
            continue
        k, got = winners(labels_by_material, cat)
        if k == 0:
            default_ += a
            unmapped.append((cat, pct))
            if pct >= 0.02:
                print(f"{cat:<22}{pct:>7.2f}%  Default")
        else:
            covered += a
            if pct >= 0.2:
                print(f"{cat:<22}{pct:>7.2f}%  {got[0]}{'  <-- REMIS ' + str(got) if len(got) > 1 else ''}")

    print(f"\nPOKRYCIE: {covered / total * 100:.2f}% powierzchni zmapowane, "
          f"{default_ / total * 100:.2f}% na materiale domyslnym")
    print(f"kategorii na domyslnym: {len(unmapped)} "
          f"(laczne pole {sum(p for _, p in unmapped):.2f}%)")
    print(f"\nZapisano: {OUT_JSON}")


if __name__ == "__main__":
    main()
