#!/usr/bin/env bash
# Czesc B (GPU): krok 6 (ED z angle_seed 1,2,3) i krok 7 (kontrola RGB-only A, D, B x ziarna 0,1,2). Szczegoly: PLAN.md, RESUME.md.
#   cd /home/d4krzyk/Dokumenty/MasterWorkVE
#   nohup setsid bash outputs/ml/overnight_2026-11/overnight_b.sh > outputs/ml/overnight_2026-11/overnight_b.log 2>&1 < /dev/null &
# Blad kroku nie zatrzymuje nastepnego; po 10 h od startu nowe przebiegi sie nie zaczynaja; przed kazdym krokiem GPU nvidia-smi + brak innych procesow.
# Zadnych operacji gita; nic nie jest nadpisywane (skrypty odmawiaja przy istniejacym katalogu/pliku).
REPO=/home/d4krzyk/Dokumenty/MasterWorkVE
OUT="$REPO/outputs/ml/overnight_2026-11"
PROGRESS="$OUT/PROGRESS_B.md"
LIMIT_S=$((10 * 3600)); T0=$(date +%s)
cd "$REPO" || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate habitat
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "$*" >> "$PROGRESS"; }
[ -e "$PROGRESS" ] && { echo "PROGRESS_B.md juz istnieje -- przerywam" >&2; exit 2; }
log "# PROGRESS_B — czesc B (GPU), kroki 6 i 7"; log ""; log "start: $(stamp)  (limit nowych przebiegow: 10 h)"; log ""
log "| krok | start | koniec | status | wyniki |"; log "|---|---|---|---|---|"
gpu_free() {
  if ps -eo args | grep -E "train_condition|train_rgbonly|evaluate\.py|noise_floor|train_pretext|transfer\.py|probe\.py|branch_diagnostics" | grep -v grep | grep -v overnight_b.sh > /dev/null; then return 1; fi
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]
}
# run_step <id> <opis> <timeout_s> <gpu:0|1> <wyniki> -- <polecenie...>
run_step() {
  local id="$1" opis="$2" tmo="$3" gpu="$4" wyn="$5"; shift 6
  local t_start t_end status rc; t_start=$(stamp)
  if [ $(( $(date +%s) - T0 )) -ge "$LIMIT_S" ]; then log "| $id $opis | $t_start | $t_start | POMINIETY (minelo 10 h) | — |"; return; fi
  if [ "$gpu" = "1" ] && ! gpu_free; then log "| $id $opis | $t_start | $(stamp) | POMINIETY (GPU zajete lub inny proces ML) | — |"; return; fi
  echo "[$(stamp)] $id: $opis"
  timeout "$tmo" "$@" > "$OUT/stepB_${id}.log" 2>&1; rc=$?; t_end=$(stamp)
  if [ $rc -eq 0 ]; then status="OK"; elif [ $rc -eq 124 ]; then status="TIMEOUT (${tmo}s)"; else status="BLAD (kod $rc)"; fi
  log "| $id $opis | $t_start | $t_end | $status | $wyn; log: stepB_${id}.log |"
}
MLD="python my-operations/ml/depth_model"
# ---- KROK 6: ED, training seed 0, angle_seed 1..3 (konfiguracja jak ED_seed0: 40 000 kroków, AMP, bez --fast-bilinear)
for a in 1 2 3; do
  run_step "6_as${a}_train" "ED angle_seed=${a}: trening" 5400 1 "runs/ED_seed0_as${a}" -- $MLD/train_condition.py --condition ED --seed 0 --tag as${a} --angle-seed ${a}
  run_step "6_as${a}_eval"  "ED angle_seed=${a}: ewaluacja test@36" 900 1 "eval/ED_seed0_as${a}/eval.json" -- $MLD/evaluate.py --run-dir outputs/ml/runs/ED_seed0_as${a} --test-set test@36
done
run_step "6_summary" "krok 6: zestawienie" 300 0 "step6_summary.md" -- python outputs/ml/overnight_2026-11/step6_summary.py
# ---- KROK 7: RGB-only A, D, B x ziarna 0,1,2 (ziarno na zewnatrz, warunek wewnatrz)
for s in 0 1 2; do for c in A D B; do
  run_step "7_${c}_seed${s}" "RGB-only ${c} ziarno ${s}" 4500 1 "rgbonly/RGBONLY_${c}_seed${s}/" -- $MLD/train_rgbonly.py --condition ${c} --seed ${s}
done; done
run_step "7_summary" "krok 7: zestawienie" 300 0 "step7_summary.md" -- python outputs/ml/overnight_2026-11/step7_summary.py
log ""; log "koniec: $(stamp)  (czas calkowity: $(( ($(date +%s) - T0) / 60 )) min)"
python "$OUT/make_summary_b.py" > "$OUT/make_summary_b.log" 2>&1
log ""; log "SUMMARY_B.md: $OUT/SUMMARY_B.md"
echo "[$(stamp)] gotowe"
