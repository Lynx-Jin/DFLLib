#!/usr/bin/env bash
# Reproduce the single-teacher FedAvgCNN generator-only distillation experiment.
set -euo pipefail

if (( $# > 2 )); then
  echo "Usage: bash tools/reproduce_single_teacher_datafree.sh [badnets|clean] [output_dir]" >&2
  exit 2
fi
mode="${1:-badnets}"
preset="${DFKD_BACKDOOR_PRESET:-reference}"
case "$preset" in
  reference) poison_ratio=0.2; patch_size=4; patch_location=lower_right; patch_margin=1 ;;
  legacy) poison_ratio=0.1; patch_size=5; patch_location=upper_left; patch_margin=0 ;;
  *) echo "DFKD_BACKDOOR_PRESET must be reference or legacy" >&2; exit 2 ;;
esac
case "$mode" in
  badnets) ;;
  clean) poison_ratio=0 ;;
  *) echo "Mode must be badnets or clean" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
python_bin="${DFKD_PYTHON:-python}"
data_root="${DFKD_DATA_ROOT:-../dataset/Cifar10/rawdata}"
output="${2:-../results/reproduce_${mode}_${preset}_dfkd_$(date +%Y%m%d_%H%M%S)_$$}"

command=("$python_bin" -u -m tools.run_datafree_badnets
  --phase full --distillation-preset custom
  --data-root "$data_root" --output "$output" --seed 0
  --teacher-epochs 80 --teacher-lr 0.05 --batch-size 128
  --backdoor-preset "$preset" --attack-seed 0
  --poison-ratio "$poison_ratio" --target-label 0
  --patch-size "$patch_size" --patch-location "$patch_location" --patch-margin "$patch_margin"
  --distill-cycles 6000 --student-steps 10
  --student-lr 0.0002 --generator-lr 0.001
  --generator-objective adversarial --generator-output tanh
  --temperature 1 --attention-weight 250 --response-weight 0
  --eval-every 1000 --log-every 500 --collapse-patience 3 --amp)

if [[ "${DFKD_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -f "$data_root/cifar-10-batches-py/test_batch" || ! -f "$data_root/cifar-10-batches-py/data_batch_1" ]]; then
  echo "CIFAR-10 raw batches not found. Set DFKD_DATA_ROOT to the directory containing cifar-10-batches-py." >&2
  exit 2
fi
if [[ -e "$output/config.json" ]]; then
  echo "An experiment already exists at $output; choose a new output directory." >&2
  exit 2
fi
mkdir -p -- "$output"
{
  printf 'CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q ' \
    "${CUDA_VISIBLE_DEVICES:-0}" "${OMP_NUM_THREADS:-4}" "${MKL_NUM_THREADS:-4}"
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$output/reproduction_command.sh"
printf 'Mode: %s\nOutput: %s\n' "$mode" "$output"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  "${command[@]}" 2>&1 | tee "$output/stdout.log"
