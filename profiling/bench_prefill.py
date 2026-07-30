#!/usr/bin/env python3
"""Benchmark prefill latency: ipttcdv9 vs ipttt vs transformer.

Random-weight models built from config JSONs (weights don't affect speed),
bf16, single GPU. Measures wall time of the prefill forward under several
modes that mirror how the eval scripts actually invoke the model, plus an
optional torch.profiler breakdown.

Modes:
  eval_mask_cache   model(input_ids, attention_mask=ones, use_cache=True,
                    logits_to_keep=1) -- what HF generate() does at prefill
  eval_nomask_cache same but without attention_mask (no unpad/varlen path)
  eval_nocache      use_cache=False (like PPL eval, but logits_to_keep=1)
  train_fwd         model.train() forward, use_cache=False (the training
                    branch: shared-QKV, two flash_attn calls) under no_grad
  gen1              model.generate(max_new_tokens=1) end-to-end

Ablations (ipttcdv9 only):
  full        unmodified
  no_ttt      block.is_ttt_layer=False on all blocks -> pure teacher path
              (single attention + plain fused MLP; ~= transformer)
  attn_only   mlp.is_ttt_layer=False -> teacher + student attention both run,
              but the TTT MLP scan is skipped (plain down_proj)

Usage:
  python profiling/bench_prefill.py --config configs/ipttcd/..._v2_swa8k.json \
      --name qwen3_v9_swa8k --ablation full --seqlens 4096 16384 65536 \
      --out profiling/results/qwen3_v9_swa8k.json [--profile-seqlen 65536]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import custom_models  # noqa: F401  (registers ipttcdv9, deltanet_baseline, ipttt)
import fla  # noqa: F401  (registers transformer)
from transformers import AutoConfig, AutoModelForCausalLM


def _parse_override(val: str):
    low = val.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def build_model(config_path: str, device: str, device_map: str | None = None,
                overrides: list[str] | None = None) -> torch.nn.Module:
    cfg = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    cfg.fuse_cross_entropy = False
    cfg.fuse_linear_cross_entropy = False
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        setattr(cfg, k, _parse_override(v))
        print(f"[bench] cfg override: {k} = {_parse_override(v)!r}", flush=True)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    m = m.to(dtype=torch.bfloat16)
    if device_map == "auto":
        from accelerate import dispatch_model, infer_auto_device_map
        no_split = ["IPTTCDBlock", "IPTTTBlock", "TransformerBlock"]
        n_gpu = torch.cuda.device_count()
        dmap = infer_auto_device_map(
            m, max_memory={i: "60GiB" for i in range(n_gpu)},
            no_split_module_classes=no_split, dtype=torch.bfloat16)
        print(f"[bench] device_map over {n_gpu} GPUs: "
              f"{sorted(set(dmap.values()), key=str)}", flush=True)
        m = dispatch_model(m, dmap)
    else:
        m = m.to(device=device)
    m.eval()
    return m


def barrier_wait(path: str, n: int, timeout_s: float = 600.0):
    """Cross-process rendezvous: touch a unique file, wait until n exist."""
    import glob as _glob
    import time as _time
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{path}.{os.getpid()}").touch()
    t0 = _time.time()
    while len(_glob.glob(f"{path}.*")) < n:
        if _time.time() - t0 > timeout_s:
            raise TimeoutError(f"barrier {path}: only {len(_glob.glob(path + '.*'))}/{n}")
        _time.sleep(0.5)
    print(f"[bench] barrier passed ({n} workers)", flush=True)


def apply_ablation(model: torch.nn.Module, ablation: str) -> None:
    if ablation == "full":
        return
    blocks = model.model.layers
    if ablation == "no_ttt":
        for blk in blocks:
            blk.is_ttt_layer = False
            if hasattr(blk, "mlp"):
                blk.mlp.is_ttt_layer = False
    elif ablation == "attn_only":
        for blk in blocks:
            if hasattr(blk, "mlp"):
                blk.mlp.is_ttt_layer = False
    else:
        raise ValueError(f"unknown ablation: {ablation}")


@torch.no_grad()
def run_mode(model, input_ids, mode: str, iters: int, warmup: int):
    device = input_ids.device
    b, L = input_ids.shape

    def call():
        if mode == "eval_mask_cache":
            model.eval()
            mask = torch.ones(b, L, dtype=torch.long, device=device)
            return model(input_ids=input_ids, attention_mask=mask,
                         use_cache=True, logits_to_keep=1)
        if mode == "eval_nomask_cache":
            model.eval()
            return model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        if mode == "eval_nocache":
            model.eval()
            return model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        if mode == "train_fwd":
            model.train()
            out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
            model.eval()
            return out
        if mode == "gen1":
            model.eval()
            return model.generate(input_ids=input_ids, max_new_tokens=1,
                                  do_sample=False, use_cache=True,
                                  pad_token_id=0)
        if mode == "eval_ppl_style":
            # Faithful replica of scripts/eval_token_ppl.py: full logits over
            # all positions (config-default use_cache) + chunked NLL on GPU.
            model.eval()
            out = model(input_ids=input_ids)
            shift_logits = out.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            bsz, tm1, _ = shift_logits.shape
            nll = torch.empty((bsz, tm1), dtype=torch.float32, device=device)
            for c0 in range(0, tm1, 512):
                c1 = min(c0 + 512, tm1)
                lp = torch.nn.functional.log_softmax(shift_logits[:, c0:c1, :].float(), dim=-1)
                nll[:, c0:c1] = -lp.gather(-1, targets[:, c0:c1, None]).squeeze(-1)
            return nll.mean()
        raise ValueError(mode)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    times.sort()
    return {
        "ms_median": times[len(times) // 2],
        "ms_min": times[0],
        "ms_max": times[-1],
        "ms_all": times,
        "peak_mem_gb": round(peak_gb, 2),
        "tok_per_s": round(b * L / (times[len(times) // 2] / 1e3)),
    }


@torch.no_grad()
def run_profile(model, input_ids, out_prefix: str, mode: str = "eval_mask_cache"):
    from torch.profiler import ProfilerActivity, profile

    device = input_ids.device
    b, L = input_ids.shape
    mask = torch.ones(b, L, dtype=torch.long, device=device)
    model.eval()

    def call():
        if mode == "eval_mask_cache":
            model(input_ids=input_ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
        elif mode == "eval_nomask_cache":
            model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        elif mode == "train_fwd":
            model.train()
            model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
            model.eval()
        else:
            raise ValueError(mode)

    call()  # warmup
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize(device)

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=80)
    with open(f"{out_prefix}.keyavg.txt", "w") as f:
        f.write(table)
    prof.export_chrome_trace(f"{out_prefix}.trace.json.gz")
    print(f"[profile] wrote {out_prefix}.keyavg.txt and trace", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ablation", default="full", choices=["full", "no_ttt", "attn_only"])
    ap.add_argument("--patch", default="none",
                    choices=["none", "shared_qkv", "batched_scan", "both", "fast_conv", "all",
                             "shared_qkv_stream", "stream_scan", "mlp_stream",
                             "fused_attn", "fused_attn_mlp_stream", "fused_conv", "fused_conv_scan", "triton_conv", "fa3_lse_conv"],
                    help="apply prototype inference optimizations from profiling/patches.py")
    ap.add_argument("--seqlens", nargs="+", type=int, default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--modes", nargs="+",
                    default=["eval_mask_cache", "eval_nomask_cache", "eval_nocache", "train_fwd", "gen1"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--check-patch", action="store_true",
                    help="compare logits before/after patching at L=16384")
    ap.add_argument("--profile-seqlen", type=int, default=None)
    ap.add_argument("--profile-mode", default="eval_mask_cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None, choices=[None, "auto"])
    ap.add_argument("--override", nargs="*", default=[],
                    help="config overrides key=value (e.g. ttt_chunk=1024 window_size=none)")
    ap.add_argument("--mlp-impl", default="scanfuse", choices=["scanfuse", "lowcopy", "base"],
                    help="force TTT MLP implementation generation (v1-era = base, fp32 full-scan)")
    ap.add_argument("--compile-mlp", action="store_true",
                    help="torch.compile every TTT MLP (after patches)")
    ap.add_argument("--fa3", action="store_true",
                    help="install FlashAttention-3 (tuple-unwrapped) into fla.layers.attn")
    ap.add_argument("--barrier", default=None,
                    help="rendezvous file prefix for concurrent multi-worker runs")
    ap.add_argument("--barrier-n", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[bench] {args.name} config={args.config} ablation={args.ablation}", flush=True)
    print(f"[bench] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    if args.fa3:
        import fla.layers.attn as _am
        import flash_attn_interface as _fi

        def _fa3_func(*a, **kw):
            out = _fi.flash_attn_func(*a, **kw)
            return out[0] if isinstance(out, tuple) else out

        _am.flash_attn_func = _fa3_func
        print("[bench] FA3 installed into fla.layers.attn (non-varlen path)", flush=True)

    model = build_model(args.config, args.device, device_map=args.device_map,
                        overrides=args.override)
    apply_ablation(model, args.ablation)
    if args.mlp_impl != "scanfuse" and model.config.model_type == "ipttcdv9":
        import types
        from custom_models.ipttcd.modeling_ipttcdv9 import IPTTCDMLP as _BaseMLP
        from custom_models.ipttcd.modeling_ipttcdv9_lowcopy import IPTTCDMLP as _LowMLP
        impl = _BaseMLP if args.mlp_impl == "base" else _LowMLP
        n = 0
        for blk in model.model.layers:
            mlp = getattr(blk, "mlp", None)
            if mlp is not None and getattr(mlp, "is_ttt_layer", False):
                mlp.forward = types.MethodType(impl.forward, mlp)
                n += 1
        print(f"[bench] forced mlp impl={args.mlp_impl} on {n} TTT layers", flush=True)
    if args.patch != "none":
        import patches
        ref = None
        if args.check_patch:
            torch.manual_seed(7)
            check_ids = torch.randint(0, model.config.vocab_size, (1, 16384), device=args.device)
            with torch.no_grad():
                model.eval()
                ref = model(input_ids=check_ids, use_cache=True, logits_to_keep=16).logits.float()
        if args.patch in ("shared_qkv", "both", "all"):
            print(f"[bench] patched shared_qkv blocks: {patches.patch_shared_qkv(model)}", flush=True)
        if args.patch in ("shared_qkv_stream", "stream_scan"):
            print(f"[bench] patched shared_qkv+stream blocks: "
                  f"{patches.patch_shared_qkv(model, use_stream=True)}", flush=True)
        if args.patch == "stream_scan":
            print(f"[bench] patched batched_scan mlps: {patches.patch_batched_scan(model)}", flush=True)
        if args.patch == "mlp_stream":
            print(f"[bench] patched mlp_stream mlps: {patches.patch_mlp_stream(model)}", flush=True)
        if args.patch in ("fused_attn", "fused_attn_mlp_stream", "fused_conv", "fused_conv_scan"):
            print(f"[bench] patched fused_attn blocks: {patches.patch_fused_attn(model)}", flush=True)
        if args.patch == "fa3_lse_conv":
            print(f"[bench] patched fa3_lse attn blocks: "
                  f"{patches.patch_fused_attn(model, kernel="fa3_lse")}", flush=True)
        if args.patch in ("fused_conv", "fused_conv_scan", "triton_conv", "fa3_lse_conv"):
            print(f"[bench] patched triton_conv mlps: {patches.patch_triton_conv(model)}", flush=True)
        if args.patch == "fused_conv_scan":
            print(f"[bench] patched batched_scan mlps: {patches.patch_batched_scan(model)}", flush=True)
        if args.patch == "fused_attn_mlp_stream":
            print(f"[bench] patched mlp_stream mlps: {patches.patch_mlp_stream(model)}", flush=True)
        if args.patch in ("batched_scan", "both", "all"):
            print(f"[bench] patched batched_scan mlps: {patches.patch_batched_scan(model)}", flush=True)
        if args.patch in ("fast_conv", "all"):
            print(f"[bench] patched fast_conv mlps: {patches.patch_fast_conv(model)}", flush=True)
        if args.patch == "all":
            print(f"[bench] patched fused_gateup mlps: {patches.patch_fused_gateup(model)}", flush=True)
        if ref is not None:
            with torch.no_grad():
                out = model(input_ids=check_ids, use_cache=True, logits_to_keep=16).logits.float()
            diff = (out - ref).abs().max().item()
            rel = diff / ref.abs().max().clamp_min(1e-6).item()
            print(f"[bench] patch numerics: max|Δlogit|={diff:.4e} rel={rel:.2e}", flush=True)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[bench] params={n_params:.1f}M", flush=True)

    if args.compile_mlp:
        n = 0
        for blk in model.model.layers:
            mlp = getattr(blk, "mlp", None)
            if mlp is not None and getattr(mlp, "is_ttt_layer", False):
                mlp.compile(dynamic=False)
                n += 1
        print(f"[bench] torch.compile on {n} TTT MLPs", flush=True)

    vocab = model.config.vocab_size
    results = {"name": args.name, "config": args.config, "ablation": args.ablation,
               "patch": args.patch, "gpu": torch.cuda.get_device_name(0), "runs": {}}

    if args.barrier:
        torch.manual_seed(99)
        warm_ids = torch.randint(0, vocab, (1, min(args.seqlens)), device=args.device)
        run_mode(model, warm_ids, args.modes[0], iters=1, warmup=1)
        barrier_wait(args.barrier, args.barrier_n)

    for L in args.seqlens:
        torch.manual_seed(1234)
        input_ids = torch.randint(0, vocab, (args.batch, L), device=args.device)
        torch.cuda.empty_cache()
        for mode in args.modes:
            t0 = time.time()
            try:
                r = run_mode(model, input_ids, mode, args.iters, args.warmup)
            except Exception as e:  # noqa: BLE001 -- keep going, report the failure
                r = {"error": f"{type(e).__name__}: {e}"}
            results["runs"][f"L{L}/{mode}"] = r
            msg = r.get("ms_median", r.get("error"))
            print(f"[bench] L={L:6d} {mode:18s} -> {msg} (wall {time.time()-t0:.1f}s)", flush=True)

    if args.profile_seqlen is not None:
        torch.manual_seed(1234)
        input_ids = torch.randint(0, vocab, (1, args.profile_seqlen), device=args.device)
        prof_dir = Path(_script_dir) / "results"
        prof_dir.mkdir(parents=True, exist_ok=True)
        run_profile(model, input_ids,
                    str(prof_dir / f"{args.name}_{args.ablation}_L{args.profile_seqlen}_{args.profile_mode}"),
                    mode=args.profile_mode)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
