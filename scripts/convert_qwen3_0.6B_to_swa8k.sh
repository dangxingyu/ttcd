#!/bin/bash
#SBATCH --job-name=conv_qwen3_swa8k
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/conv_qwen3_swa8k_%j.out
#SBATCH --error=logs/conv_qwen3_swa8k_%j.err

set -e
mkdir -p logs
cd /scratch/gpfs/ARORA/xd7812/ipttcd

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export PYTHONPATH="/scratch/gpfs/ARORA/xd7812/ipttcd:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG_PATH="configs/transformer/transformer_qwen3_0.6B_swa8k.json"
OUTPUT_DCP="exp/transformer_qwen3_0.6B_swa8k_ct/checkpoint/step-0"

echo "Converting Qwen3-0.6B HF -> flame SWA-8192 transformer DCP..."

python3 -m flame.utils.convert_qwen_to_ipttcd_dcp \
    --model /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
    --config "${CONFIG_PATH}" \
    --tokenizer tokenizers/Qwen3-0.6B-Base \
    --checkpoint "${OUTPUT_DCP}"

echo "Conversion complete: ${OUTPUT_DCP}"
