#!/bin/bash
#SBATCH --job-name=ruler_gen_5
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ruler_gen_5_%j.out
#SBATCH --error=logs/ruler_gen_5_%j.err

# Generate JSONL data for the 5 non-NIAH RULER tasks at 5 ctx_lens for the
# Qwen3 tokenizer only. 100 samples/cell. Output:
#   data/niah/qwen3/ctx{N}/{task}/validation.jsonl
# (re-using the existing data/niah/qwen3 layout so eval_ruler.py just works.)

set -e
mkdir -p logs
cd /scratch/gpfs/ARORA/xd7812/ipttcd
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH="/scratch/gpfs/ARORA/xd7812/ipttcd:$PYTHONPATH"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

NUM=${NUM:-100}
CTXS="${CTXS:-4096 8192 16384 32768 65536}"
TASKS="${TASKS:-vt cwe fwe qa_1 qa_2}"
TOK_PATH="/scratch/gpfs/ARORA/xd7812/ipttcd/tokenizers/Qwen3-0.6B-Base"
OUT="data/niah/qwen3"

for task in $TASKS; do
    for ctx in $CTXS; do
        OUT_FILE="$OUT/ctx${ctx}/$task/validation.jsonl"
        if [ -f "$OUT_FILE" ] && [ $(wc -l < "$OUT_FILE") -ge "$NUM" ]; then
            echo "  [skip] $OUT_FILE already has $(wc -l < $OUT_FILE) lines"
            continue
        fi
        echo "=== $task / ctx=$ctx (NUM=$NUM) ==="
        mkdir -p "$OUT/ctx${ctx}"
        (cd 3rdparty/RULER/scripts/data && python3 prepare.py \
            --save_dir "/scratch/gpfs/ARORA/xd7812/ipttcd/$OUT/ctx${ctx}" \
            --benchmark synthetic \
            --task "$task" \
            --tokenizer_path "$TOK_PATH" \
            --tokenizer_type hf \
            --max_seq_length "$ctx" \
            --num_samples "$NUM" \
            --model_template_type base) 2>&1 | tail -3
    done
done

echo "Done at $(date)"
