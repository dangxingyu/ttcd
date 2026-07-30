#!/bin/bash
#SBATCH --job-name=conv_smol_v9_swa8k
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/conv_smol_v9_swa8k_%j.out
#SBATCH --error=logs/conv_smol_v9_swa8k_%j.err

set -e
mkdir -p logs
cd /scratch/gpfs/ARORA/xd7812/ipttcd
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH="/scratch/gpfs/ARORA/xd7812/ipttcd:$PYTHONPATH"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

CONFIG_PATH="configs/ipttcd/ipttcdv9_smollm2_360M_ct_64k_v2_swa8k.json"
OUTPUT_DCP="exp/ipttcdv9_smollm2_360M_ct_64k_v2_swa8k/checkpoint/step-0"

echo "Converting SmolLM2-360M -> IPTTCDv9 v2 SWA-8192 DCP..."
python3 -m flame.utils.convert_qwen_to_ipttcd_dcp \
    --model /scratch/gpfs/ARORA/xd7812/models/SmolLM2-360M \
    --config "${CONFIG_PATH}" \
    --tokenizer tokenizers/SmolLM2-360M \
    --checkpoint "${OUTPUT_DCP}"
echo "Done: ${OUTPUT_DCP}"
