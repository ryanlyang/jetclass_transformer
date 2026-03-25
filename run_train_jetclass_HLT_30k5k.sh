#!/bin/bash
#SBATCH -J jcHLTtb
#SBATCH -p tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o offline_reconstructor_logs/jetclass_hlt_teacher_baseline_%j.out
#SBATCH -e offline_reconstructor_logs/jetclass_hlt_teacher_baseline_%j.err

set -euo pipefail

WORKDIR="/home/ryreu/atlas/jetclass_transformer"
SCRIPT="/home/ryreu/atlas/PracticeTagging/evaluate_jetclass_hlt_teacher_baseline.py"
DATA_DIR="/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
SAVE_DIR="${WORKDIR}/checkpoints/jetclass_hlt_teacher_baseline"

cd "${WORKDIR}"

source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate atlas_kd

mkdir -p offline_reconstructor_logs "${SAVE_DIR}"

export PYTHONWARNINGS="ignore"
export TQDM_DISABLE=1

RAW_LOG="offline_reconstructor_logs/jetclass_hlt_teacher_baseline_raw_${SLURM_JOB_ID}.log"

CMD=(
  python -u "${SCRIPT}"
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
  --run_name "jetclass_part0_hlt_teacher_baseline_30k5k10k"
  --n_train_jets 30000
  --n_val_jets 5000
  --n_test_jets 10000
  --max_constits 128
  --feature_mode full
  --batch_size 512
  --epochs 30
  --patience 8
  --num_workers 2
  --device cuda
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${RAW_LOG}" | awk '
/Teacher ep [0-9]+:/ {
  n=$0; sub(/^.*Teacher ep /,"",n); sub(/:.*/,"",n);
  if ((n % 5) == 0) print;
  next
}
/Baseline ep [0-9]+:/ {
  n=$0; sub(/^.*Baseline ep /,"",n); sub(/:.*/,"",n);
  if ((n % 5) == 0) print;
  next
}
/FINAL SUMMARY|FINAL TEST METRICS|AUC|FPR|Saved outputs|Run completed|Traceback|RuntimeError|Error:|^=+/ { print }
'

echo "Full raw log: ${RAW_LOG}"
