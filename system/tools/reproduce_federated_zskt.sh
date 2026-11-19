#!/usr/bin/env bash
# Native framework experiment: local FedAvg training -> FedAvg student -> ZSKT.
set -euo pipefail
if (( $# > 1 )); then
  echo 'Usage: bash tools/reproduce_federated_zskt.sh [badnets|fedavg|clean]' >&2
  exit 2
fi
mode="${1:-badnets}"
clients="${FL_ZSKT_CLIENTS:-20}"
rounds="${FL_ZSKT_ROUNDS:-200}"
local_epochs="${FL_ZSKT_LOCAL_EPOCHS:-2}"
cycles="${FL_ZSKT_KD_CYCLES:-200}"
malicious="${FL_ZSKT_MALICIOUS_RATIO:-0.2}"
seed="${FL_ZSKT_SEED:-1000}"
attack_seed="${FL_ZSKT_ATTACK_SEED:-0}"
case "$mode" in
  badnets) ;;
  fedavg) cycles=0 ;;
  clean) malicious=0 ;;
  *) echo 'Mode must be badnets, fedavg, or clean' >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
python_bin="${DFKD_PYTHON:-python}"
raw="${DFKD_DATA_ROOT:-../dataset/Cifar10/rawdata}"
if [[ ! -f "$raw/cifar-10-batches-py/test_batch" && -f ../../PFLlib-master/dataset/Cifar10/rawdata/cifar-10-batches-py/test_batch ]]; then
  raw=../../PFLlib-master/dataset/Cifar10/rawdata
fi
data_protocol="${FL_ZSKT_DATA_PROTOCOL:-legacy}"
partition="${FL_ZSKT_PARTITION:-iid}"
alpha="${FL_ZSKT_ALPHA:-0.1}"
case "$partition" in
  iid) ;;
  dirichlet)
    alpha="$("$python_bin" -c 'import math,sys
try: value=float(sys.argv[1])
except ValueError: raise SystemExit("FL_ZSKT_ALPHA must be numeric")
if not math.isfinite(value) or value <= 0: raise SystemExit("FL_ZSKT_ALPHA must be finite and positive")
print(format(value,".12g"))' "$alpha")"
    ;;
  *) echo 'FL_ZSKT_PARTITION must be iid or dirichlet' >&2; exit 2 ;;
esac
legacy_root="${FL_ZSKT_LEGACY_DATA_ROOT:-}"
if [[ -z "$legacy_root" ]]; then
  if [[ "$partition" == iid ]]; then
    legacy_root=../dataset/Cifar10
    if [[ -f ../../PFLlib-master/dataset/Cifar10/config.json ]]; then
      legacy_root=../../PFLlib-master/dataset/Cifar10
    fi
  else
    compact_alpha="${alpha//./}"
    legacy_root="../dataset/Cifar10_dir${alpha}"
    for candidate in "../../PFLlib-master/dataset/Cifar10_dir${alpha}" \
                     "../../PFLlib-master/dataset/Cifar10_dir${compact_alpha}" \
                     "../dataset/Cifar10_dir${alpha}" "../dataset/Cifar10_dir${compact_alpha}"; do
      if [[ -f "$candidate/config.json" ]]; then
        legacy_root="$candidate"
        break
      fi
    done
  fi
fi
case "$data_protocol" in
  legacy)
    if [[ "$partition" == iid ]]; then
      dataset="Cifar10_ZSKT_legacy${clients}"
    else
      dataset="Cifar10_ZSKT_dir${alpha}_legacy${clients}"
    fi
    ;;
  official)
    if [[ "$partition" == dirichlet ]]; then
      echo 'Dirichlet mode reuses existing client splits; set FL_ZSKT_DATA_PROTOCOL=legacy' >&2
      exit 2
    fi
    dataset="Cifar10_ZSKT_IID${clients}_official_seed${seed}"
    ;;
  *) echo 'FL_ZSKT_DATA_PROTOCOL must be legacy or official' >&2; exit 2 ;;
esac
if [[ "$partition" == iid ]]; then
  default_logs=../results/federated_zskt_badnets_reference
  run_goal="${mode}_reference_${data_protocol}"
else
  default_logs="../results/federated_zskt_badnets_reference_dir${alpha}"
  run_goal="${mode}_reference_${data_protocol}_dir${alpha}"
fi
logger_root="${FL_ZSKT_LOGGER_ROOT:-$default_logs}"
prepare=("$python_bin" -u -m tools.prepare_federated_zskt
  --protocol "$data_protocol" --partition "$partition" --alpha "$alpha" --legacy-root "$legacy_root"
  --data-root "$raw" --output "../dataset/$dataset" --num-clients "$clients" --seed "$seed")
command=("$python_bin" -u main.py
  --algorithm FedAvgZSKT --model CNN --dataset "$dataset" --num_classes 10
  --device cuda --device_id "${CUDA_VISIBLE_DEVICES:-0}"
  --num_clients "$clients" --join_ratio 1 --global_rounds "$rounds"
  --local_epochs "$local_epochs" --batch_size 10 --local_learning_rate 0.005
  --attack_type none --defense_type none --attack_malicious_ratio "$malicious"
  --attack_seed "$attack_seed" --backdoor badnets --backdoor_preset reference --backdoor_target_label 0
  --backdoor_poison_ratio "${FL_ZSKT_POISON_RATIO:-0.2}"
  --backdoor_patch_size 4 --backdoor_patch_location lower_right --backdoor_patch_margin 1
  --backdoor_seed "$attack_seed"
  --zskt_cycles_per_round "$cycles" --zskt_student_steps 10 --zskt_batch_size 128
  --zskt_student_lr 0.0002 --zskt_generator_lr 0.001
  --zskt_temperature 1 --zskt_attention_weight 250
  --zskt_local_momentum 0 --zskt_local_weight_decay 0
  --zskt_local_augmentation 0 --zskt_local_lr_schedule constant --zskt_local_lr_min 0.005
  --zskt_evaluation_split auto
  --zskt_raw_data_root "$raw" --zskt_seed "$seed" --zskt_threads 4
  --zskt_teacher_eval_every 10 --zskt_health_every 50 --zskt_collapse_patience 3
  --attack_monitor_plot_every 100 --eval_gap 1 --times 1 --goal "$run_goal"
  --logger_root "$logger_root")
if [[ -n "${FL_ZSKT_RESUME:-}" ]]; then
  command+=(--zskt_resume "$FL_ZSKT_RESUME")
fi
if [[ "${DFKD_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${prepare[@]}"; printf '\n'
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
"$python_bin" -c 'import torch, h5py, calmsize; assert torch.cuda.is_available(), "A CUDA Python environment is required"'
"${prepare[@]}"
printf 'Mode=%s protocol=%s clients=%s malicious_ratio=%s rounds=%s local_epochs=%s KD_cycles/round=%s\n' \
  "$mode" "$data_protocol" "$clients" "$malicious" "$rounds" "$local_epochs" "$cycles"
printf 'Partition=%s alpha=%s dataset=%s source=%s\n' "$partition" "$alpha" "$dataset" "$legacy_root"
printf 'Local recipe: SGD lr=0.005 constant, batch=10, momentum=0, weight_decay=0, augmentation=off\n'
printf 'BadNets reference: poison fraction of ALL local samples, fixed client indices, 4x4 lower-right margin1; experiment_seed=%s attack_seed=%s\n' "$seed" "$attack_seed"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  "${command[@]}"
