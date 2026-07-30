# TTCD: Test-Time Training via Context Distillation

Official implementation of **"Learning What to Remember: Test-Time Training via Context Distillation"**.

TTCD is a test-time training (TTT) framework that unifies past-context compression and online adaptation through a single context-distillation objective: a long-window **teacher** attention and a short-window **student** attention run over shared Q/K/V, and the hidden-state discrepancy between them is distilled into fast weights — a low-rank correction to the existing MLP down-projection (**IP-TTCD**, the in-place instantiation). The model learns *what to remember* from distant context instead of trying to store everything.

This repository contains:

- **`custom_models/`** — the IP-TTCD model (`ipttcdv9`, HuggingFace-registered) and the IP-TTT baseline (`ipttt`), plus a DeltaNet baseline.
- **`flame/`** — a minimal training framework built on `torchtitan` (FSDP/TP/CP, DCP checkpoints, online tokenization and pre-tokenized MDS data loading), used for both from-scratch pre-training and long-context continued training (CT).
- **`3rdparty/flash-linear-attention/`** (imported as `fla/`) — Triton linear-attention operators and the transformer baseline, with local modifications (sliding-window + YaRN support) that the CT baselines depend on.
- **`profiling/`** — the fused inference kernels described in the paper's efficiency appendix: a fused dual-window attention Triton kernel (both windowed outputs in one pass over K/V), an LSE-combination variant for FlashAttention-3, and a chunked depthwise-convolution kernel, plus benchmark and numerical-equivalence tooling.
- **`configs/`**, **`scripts/`**, **`test/`** — model configs, convert/train/eval pipelines, and end-to-end tests.

## Installation

Python 3.10+, `torch>=2.5`, `triton>=3.0`, `transformers>=4.45`. We use `uv`/`venv` (not conda):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes editable installs of flame and fla
```

Scripts assume the repo root is on `PYTHONPATH` so `custom_models`, `flame`, and `fla` import:

```bash
export PYTHONPATH=$PWD
```

`import custom_models` registers `ipttcdv9` with HuggingFace `AutoConfig`/`AutoModelForCausalLM`; `import fla` registers the `transformer` baseline. Import both before `AutoModelForCausalLM.from_config`.

## Continued training (CT) pipeline

The main experimental loop converts a pretrained HF checkpoint into an IP-TTCD model and continues training at 64K context. Three stages, using Qwen3-0.6B as the example (SmolLM2-360M scripts are analogous):

**1. Convert** an HF checkpoint into a flame DCP init:

```bash
python -m flame.utils.convert_qwen_to_ipttcd_dcp \
  --model <path-to>/Qwen3-0.6B \
  --config configs/ipttcd/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.json \
  --tokenizer <path-to>/Qwen3-0.6B-Base \
  --checkpoint exp/qwen3_v9_64k/checkpoint/step-0
```

**2. Train** with `torchrun -m flame.train` (see `scripts/run_ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.sh` for the full SLURM invocation):

```bash
torchrun --nproc_per_node=8 -m flame.train \
  --job.config_file flame/models/fla.toml \
  --model.config configs/ipttcd/ipttcdv9_qwen3_0.6B_ct_64k_v2_swa8k.json \
  --model.tokenizer_path <path-to>/Qwen3-0.6B-Base \
  --checkpoint.initial_load_path exp/qwen3_v9_64k/checkpoint/step-0 \
  --checkpoint.initial_load_model_weights_only \
  ...
```

`--training.dataset` accepts an HF dataset path (online tokenization) or an `mds:` URI for pre-tokenized shards (e.g. [`princeton-nlp/prolong-data-64K`](https://huggingface.co/datasets/princeton-nlp/prolong-data-64K)): `mds:/path/prolong-data-64k?domain=book-65536`.

**3. Evaluate** directly from the DCP checkpoint (no HF export needed):

```bash
# per-token-position NLL curves (TTT-E2E methodology)
python scripts/eval_token_ppl.py --config <json> --checkpoint exp/.../checkpoint/step-N ...
# RULER NIAH accuracy over context lengths
python scripts/eval_ruler.py  --config <json> --checkpoint exp/.../checkpoint/step-N ...
```

NIAH/RULER data is pre-generated per tokenizer and context length by `scripts/niah_gen_ruler.sh` / `scripts/ruler_gen_5tasks.sh`, which require a clone of [NVIDIA/RULER](https://github.com/NVIDIA/RULER) under `3rdparty/RULER`.

For from-scratch pre-training, use the configs under `configs/ipttcd/`, `configs/ipttt/`, `configs/transformer/`, `configs/delta_net/`, `configs/gated_deltanet/` with the same `flame.train` entry point.

## Model configuration

Key config fields for `ipttcdv9` (see `custom_models/ipttcd/configuration_ipttcd.py`):

| Field | Meaning |
|---|---|
| `window_size` | teacher sliding window $w_T$ |
| `ttt_chunk`, `ttt_visible_chunks` | student window $w_S = $ chunk $\times$ visible chunks (recipe: $w_S = w_T/2$) |
| `ttt_layers` | which layers carry fast weights |
| `ttt_lr` | fast-weight learning rate |
| `ttt_proj_init`, `ttt_teacher_conv_init` | CT initialization (`diagonal_gaussian` + identity teacher conv = the "v2 init massage") |
| `ttt_force_grouped_scan` | `true` = fused scan path, `false` = unfused fallback |

## Fused inference kernels

`profiling/` contains the inference kernels from the paper's efficiency appendix:

- `fused_dual_attn.py` — fused dual-window attention (Triton): both teacher and student outputs in a single pass over K/V, recovering the student's PV product via the online-softmax rescaling identity. Numerics track CUTLASS FlashAttention-2 step by step (bit-identical on 99.7–99.9% of outputs).
- `lse_dual_attn.py` — LSE-combination variant: student-window call + shifted far-band call merged exactly through their log-sum-exp; the fastest option on the FlashAttention-3 stack.
- `fused_conv.py` — chunked depthwise causal convolution (bitwise-equal to the cuDNN path).
- `patches.py` + `bench_prefill.py` — apply the kernels to the model and reproduce the end-to-end prefill benchmarks; `FUSED_ATTENTION_DESIGN.md` documents the kernel design.

## Tests

```bash
pytest test/     # conversion + kernel tests (GPU required)
pytest 3rdparty/flash-linear-attention/tests/ops/test_delta.py
```

## Acknowledgements

This codebase builds on [flame](https://github.com/fla-org/flame), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and [torchtitan](https://github.com/pytorch/torchtitan). Long-context CT follows the [ProLong](https://github.com/princeton-nlp/ProLong) recipe; evaluation methodology follows TTT-E2E and [RULER](https://github.com/NVIDIA/RULER).

## Citation

```bibtex
@article{ttcd2026,
  title={Learning What to Remember: Test-Time Training via Context Distillation},
  year={2026},
  note={Under review}
}
```
