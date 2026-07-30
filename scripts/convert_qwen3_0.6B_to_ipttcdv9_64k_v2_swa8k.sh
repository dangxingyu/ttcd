#!/bin/bash
#SBATCH --job-name=convert_qwen3_v9_64k_v2_swa8k
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/convert_qwen3_v9_64k_v2_swa8k_%j.out
#SBATCH --error=logs/convert_qwen3_v9_64k_v2_swa8k_%j.err

set -e
mkdir -p logs
cd /scratch/gpfs/ARORA/xd7812/ipttcd

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export PYTHONPATH="/scratch/gpfs/ARORA/xd7812/ipttcd:$PYTHONPATH"

CONFIG_PATH="configs/ipttcd/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.json"
OUTPUT_DCP="exp/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k/checkpoint/step-0"

echo "Converting Qwen3-0.6B to IPTTCDv9 (64K, v2 init massage + SWA-8192) DCP..."
echo "  Config: ${CONFIG_PATH}"
echo "  Output: ${OUTPUT_DCP}"

python3 -m flame.utils.convert_qwen_to_ipttcd_dcp \
    --model /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
    --config "${CONFIG_PATH}" \
    --tokenizer tokenizers/Qwen3-0.6B-Base \
    --checkpoint "${OUTPUT_DCP}"

echo "Conversion complete: ${OUTPUT_DCP}"
