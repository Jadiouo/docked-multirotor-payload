#!/usr/bin/env bash
# One-click reproduction.
#   bash scripts/reproduce.sh          fast: rebuild results/figures/main.png from saved json (seconds; panel (b) numbers are hard-coded from the docs/prereg.md section 5.b table, not recomputed)
#   bash scripts/reproduce.sh --full   re-run the classical experiments (hours, CPU only), then redraw
#   bash scripts/reproduce.sh --full --dry-run   only print the commands
#   bash scripts/reproduce.sh --help
# Env: PY (default .venv/bin/python). RL results (results/p3) need GPU checkpoints and are NOT re-run here;
# their saved eval json is used by the figure (see scripts/rl_eval.py, scripts/rl_train.py to regenerate).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1

FULL=0; DRY=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

run() {
  echo "+ $*"
  if [ "$DRY" -eq 0 ]; then "$@"; fi
}

if [ "$FULL" -eq 1 ]; then
  N=100
  # order matters: g1_coord tune writes the yawlim gains that g2_estim / rqd read
  run "$PY" -m scripts.g1_v2 --n-eps "$N"
  run "$PY" -m scripts.g1_coord tune --n-eps "$N"
  run "$PY" -m scripts.g1_coord eval --n-eps "$N" --stage t2
  run "$PY" -m scripts.g1_coord eval --n-eps "$N" --stage rest
  run "$PY" -m scripts.g1_coord report --n-eps "$N"
  run "$PY" -m scripts.g1_yawcost --n-eps "$N"
  run "$PY" -m scripts.g2_mismatch --n-eps "$N"
  run "$PY" -m scripts.g2_informed --n-eps "$N"
  run "$PY" -m scripts.g2_estim --stage tune
  run "$PY" -m scripts.g2_estim --stage eval --n-eps "$N"
  run "$PY" -m scripts.g2_estim --stage holdout --n-eps "$N"
  run "$PY" -m scripts.g2_estim --stage report --n-eps "$N"
  run "$PY" -m scripts.rqd_classical --stage tune
  run "$PY" -m scripts.rqd_classical --stage run --n-eps "$N"
  run "$PY" -m scripts.rqd_classical --stage report --n-eps "$N"
  run "$PY" -m scripts.p3d_factorial --stage run --n-eps "$N"
  run "$PY" -m scripts.p3d_factorial --stage report --n-eps "$N"
fi

run "$PY" -m scripts.make_figures
