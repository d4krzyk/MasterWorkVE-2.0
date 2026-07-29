#!/bin/bash
# Pobiera prekomputowane RIR-y SoundSpaces 1.0 dla wybranych scen Replica
# i uruchamia analize kontrastu otwarte/zamkniete (soundspaces1_rt60.py).
#
# PO CO: rozstrzygniecie, czy SoundSpaces 1.0 uzywal tych samych scen Replica
# BEZ SUFITU. Szczegoly i predykcja falsyfikowalna w naglowku
# my-operations/measurements/soundspaces1_rt60.py oraz w
# my-operations/docs/OBSERWACJE_METODOLOGICZNE.md §1.
#
# Uruchomienie w tle, odporne na zamkniecie sesji:
#   setsid nohup my-operations/measurements/fetch_soundspaces1_rirs.sh \
#       > outputs/measurements/ss1_check.log 2>&1 &
#
# Wznawialne: wget -c dociaga przerwany transfer, a sceny juz rozpakowane
# sa pomijane. Mozna wiec puscic ponownie po zerwaniu lacza.

set -u
cd "$(dirname "$0")/../.." || exit 1          # katalog glowny repo
REPO="$PWD"
DATA="$REPO/sound-spaces/data"                # sound-spaces/.gitignore ignoruje 'data'
OUT="$REPO/outputs/measurements"
BASE="http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs/replica"

# Kolejnosc celowa: najpierw mala scena zamknieta (0.10 GB) — szybki wynik
# czastkowy i wczesne wykrycie problemu z formatem, dopiero potem duza otwarta.
SCENES=("office_1" "frl_apartment_2")

mkdir -p "$DATA" "$OUT"
echo "=== START $(date '+%F %T') ==="
echo "  repo:  $REPO"
echo "  dane:  $DATA/binaural_rirs/replica"
df -h "$REPO" | tail -1

for s in "${SCENES[@]}"; do
    dst="$DATA/binaural_rirs/replica/$s"
    if [ -d "$dst" ] && [ -n "$(ls -A "$dst" 2>/dev/null)" ]; then
        echo "[$(date '+%T')] $s: juz rozpakowane, pomijam"
        continue
    fi
    echo "[$(date '+%T')] $s: pobieranie..."
    cd "$DATA" || exit 1
    if ! wget -c -q --show-progress --progress=dot:giga "$BASE/$s.tar.gz" 2>&1 | tail -3; then
        echo "[$(date '+%T')] $s: BLAD pobierania — przerywam te scene"
        continue
    fi
    echo "[$(date '+%T')] $s: rozpakowywanie..."
    # Archiwa zawieraja wiodacy katalog `data/`, a my jestesmy juz w $DATA —
    # stad --strip-components=1, inaczej powstaje data/data/binaural_rirs/.
    # (README SoundSpaces podaje tez separator "[receiver]-[source].wav", ale
    # w rzeczywistosci jest to "_" — dla odczytu bez znaczenia, glob bierze *.wav.)
    if tar xzf "$s.tar.gz" --strip-components=1; then
        rm -f "$s.tar.gz"                      # ~7 GB, nie trzymamy obok rozpakowanych
        n=$(find "binaural_rirs/replica/$s" -name '*.wav' 2>/dev/null | wc -l)
        echo "[$(date '+%T')] $s: gotowe, $n plikow .wav"
    else
        echo "[$(date '+%T')] $s: BLAD rozpakowywania"
    fi
    df -h "$REPO" | tail -1
done

echo
echo "=== ANALIZA $(date '+%T') ==="
cd "$REPO" || exit 1
python my-operations/measurements/soundspaces1_rt60.py 2>&1 | tee "$OUT/ss1_verdict.txt"
echo
echo "=== KONIEC $(date '+%F %T') ==="
echo "  werdykt zapisany w: $OUT/ss1_verdict.txt"
