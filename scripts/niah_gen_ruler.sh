#!/bin/bash
#SBATCH --job-name=niah_gen
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/niah_gen_%j.out
#SBATCH --error=logs/niah_gen_%j.err

# Generate the 6 NIAH task JSONLs per tokenizer × ctx_len using the RULER
# official prepare.py. Output:
#   data/niah/{tokenizer}/{task}_ctx{ctx_len}/validation.jsonl

set -e
cd /scratch/gpfs/ARORA/xd7812/ipttcd
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH="/scratch/gpfs/ARORA/xd7812/ipttcd:$PYTHONPATH"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

NUM=${NUM:-100}
CTXS="${CTXS:-4096 8192 16384 32768 65536}"
TASKS="${TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multivalue niah_multiquery}"
PREPARE="3rdparty/RULER/scripts/data/prepare.py"

for tok in Qwen3-0.6B-Base SmolLM2-360M; do
    case "$tok" in
        Qwen3-0.6B-Base) OUT="data/niah/qwen3" ;;
        SmolLM2-360M)    OUT="data/niah/smollm2" ;;
    esac
    TOK_PATH="/scratch/gpfs/ARORA/xd7812/ipttcd/tokenizers/$tok"
    for task in $TASKS; do
        for ctx in $CTXS; do
            SAVE_DIR="$OUT/ctx${ctx}"
            OUT_FILE="$SAVE_DIR/$task/validation.jsonl"
            if [ -f "$OUT_FILE" ]; then
                echo "  [skip] $OUT_FILE already exists"
                continue
            fi
            echo "=== $tok / $task / ctx=$ctx ==="
            mkdir -p "$SAVE_DIR"
            (cd 3rdparty/RULER/scripts/data && python3 prepare.py \
                --save_dir "/scratch/gpfs/ARORA/xd7812/ipttcd/$SAVE_DIR" \
                --benchmark synthetic \
                --task "$task" \
                --tokenizer_path "$TOK_PATH" \
                --tokenizer_type hf \
                --max_seq_length "$ctx" \
                --num_samples "$NUM" \
                --model_template_type base) 2>&1 | tail -4
        done
    done
done

echo "Done at $(date)"
