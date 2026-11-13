#!/usr/bin/env bash
# Nocny zestaw sprawdzen metodologicznych (kroki 1-4). Szczegoly: PLAN.md.
#
#   cd /home/d4krzyk/Dokumenty/MasterWorkVE
#   nohup setsid bash outputs/ml/overnight_2026-11/overnight.sh > outputs/ml/overnight_2026-11/overnight.log 2>&1 < /dev/null &
#
# Zasady: kazdy krok w osobnym bloku (blad/timeout -> PROGRESS.md, dalej); kroki GPU sekwencyjnie,
# przed kazdym nvidia-smi + brak innych procesow; po 10 h od startu nie zaczyna nowych krokow;
# ZADNYCH operacji gita; nic nie jest nadpisywane (skrypty same odmawiaja przy istniejacym pliku).
# Kroki 5-7 NIE byly okreslone w poleceniu -- dopisac nowy blok `run_step` ponizej, jesli beda.

REPO=/home/d4krzyk/Dokumenty/MasterWorkVE
OUT="$REPO/outputs/ml/overnight_2026-11"
PROGRESS="$OUT/PROGRESS.md"
LIMIT_S=$((10 * 3600))           # po tylu sekundach od startu nie zaczynamy nowych krokow
T0=$(date +%s)

cd "$REPO" || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate habitat
mkdir -p "$OUT"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "$*" >> "$PROGRESS"; }

if [ -e "$PROGRESS" ]; then
  echo "PROGRESS.md juz istnieje -- nie nadpisuje poprzedniego przebiegu; przerywam." >&2
  exit 2
fi
log "# PROGRESS — nocny zestaw 2026-11"
log ""
log "start: $(stamp)  (limit nowych krokow: 10 h)"
log ""
log "| krok | start | koniec | status | wyniki |"
log "|---|---|---|---|---|"

gpu_free() {
  # brak procesow treningu/ewaluacji i brak procesow na GPU
  if ps -eo args | grep -E "train_condition|evaluate\.py|noise_floor|train_pretext|transfer\.py|probe\.py|branch_diagnostics" \
       | grep -v grep | grep -v overnight.sh > /dev/null; then return 1; fi
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then return 1; fi
  return 0
}

# run_step <nr> <opis> <timeout_s> <gpu:0|1> <wyniki(opis)> -- <polecenie...>
run_step() {
  local nr="$1" opis="$2" tmo="$3" gpu="$4" wyn="$5"; shift 6
  local t_start t_end status rc
  t_start=$(stamp)
  if [ $(( $(date +%s) - T0 )) -ge "$LIMIT_S" ]; then
    log "| $nr $opis | $t_start | $t_start | POMINIETY (minelo 10 h) | — |"; return
  fi
  if [ "$gpu" = "1" ] && ! gpu_free; then
    log "| $nr $opis | $t_start | $(stamp) | POMINIETY (GPU zajete lub inny proces ML) | — |"; return
  fi
  echo "[$(stamp)] krok $nr: $opis"
  timeout "$tmo" "$@" > "$OUT/step${nr}.log" 2>&1
  rc=$?
  t_end=$(stamp)
  if   [ $rc -eq 0 ];   then status="OK"
  elif [ $rc -eq 124 ]; then status="TIMEOUT (${tmo}s)"
  else                       status="BLAD (kod $rc) — patrz step${nr}.log"; fi
  log "| $nr $opis | $t_start | $t_end | $status | $wyn; log: step${nr}.log |"
}

PY="python my-operations/ml/analysis"
run_step 1 "Holm w rodzinach testow"           600  0 "holm_families.json/.md"                     -- $PY/holm_families.py
run_step 2 "piksele > 10,29 m (test@36, train)" 7200 0 "depth_range_stats.json/.md"                -- $PY/depth_range_stats.py
run_step 3 "bootstrap EA vs EB, n_boot=2000"    7200 0 "bootstrap_EA_vs_EB_seed{0,1,2}_nboot2000.json, bootstrap_EA_vs_EB_summary.md" -- $PY/bootstrap_ea_eb.py
run_step 4 "branch_diagnostics A,D,B x 0,1,2"   7200 1 "branch_diag_*.json, branch_diagnostics_summary.md/.csv" -- $PY/branch_diagnostics.py --conditions A D B --seeds 0 1 2

log ""
log "koniec: $(stamp)  (czas calkowity: $(( ($(date +%s) - T0) / 60 )) min)"
python "$OUT/make_summary.py" > "$OUT/make_summary.log" 2>&1
log ""
log "SUMMARY.md: $OUT/SUMMARY.md (status generatora: $( [ -s "$OUT/SUMMARY.md" ] && echo ok || echo brak -- patrz make_summary.log ))"
echo "[$(stamp)] gotowe"
