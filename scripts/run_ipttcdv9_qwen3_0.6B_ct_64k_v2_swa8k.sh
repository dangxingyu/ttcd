#!/bin/bash
#SBATCH --job-name=ipttcdv9_qwen3_ct64k_v2_swa8k
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ipttcdv9_qwen3_ct64k_v2_swa8k_%j.out
#SBATCH --error=logs/ipttcdv9_qwen3_ct64k_v2_swa8k_%j.err

set -e
mkdir -p logs
cd /scratch/gpfs/ARORA/xd7812/ipttcd

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
export PYTHONPATH="/scratch/gpfs/ARORA/xd7812/ipttcd:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export CONFIG_PATH="configs/ipttcd/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.json"
export TOKENIZER_PATH="tokenizers/Qwen3-0.6B-Base"
export EXP_DIR="exp/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k/run"
export DCP_INIT="exp/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k/checkpoint/step-0"

export NGPU=8
export BATCH_SIZE=1
export SEQ_LEN=65536
export CONTEXT_LEN=65536
export GRAD_ACCUM=1
export LR=1e-4
export LR_FINAL=1e-6
export WARMUP_STEPS=200
export TOTAL_STEPS=10000

mkdir -p "${EXP_DIR}"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m flame.train \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder ${EXP_DIR} \
  --model.config "${CONFIG_PATH}" \
  --model.tokenizer_path "${TOKENIZER_PATH}" \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr ${LR} \
  --lr_scheduler.warmup_steps ${WARMUP_STEPS} \
  --lr_scheduler.lr_min_absolute ${LR_FINAL} \
  --lr_scheduler.decay_type cosine \
  --training.batch_size ${BATCH_SIZE} \
  --training.seq_len ${SEQ_LEN} \
  --training.context_len ${CONTEXT_LEN} \
  --training.gradient_accumulation_steps ${GRAD_ACCUM} \
  --training.steps ${TOTAL_STEPS} \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset /scratch/gpfs/ARORA/xd7812/ipttcd/data/books_65k \
  --training.dataset_split train \
  --training.num_workers 4 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.data_parallel_replicate_degree 1 \
  --training.data_parallel_shard_degree -1 \
  --checkpoint.initial_load_path ${DCP_INIT} \
  --checkpoint.initial_load_model_weights_only \
  --checkpoint.interval 2000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 10 \
  --activation_checkpoint.mode full \
  --comm.train_timeout_seconds 600

echo "Training completed at $(date)"
