"""Prototype inference-path optimizations for IPTTCDv9, applied by monkeypatch.

Measures the recoverable headroom of the current `elif self.is_ttt_layer:`
inference branch in modeling_ipttcdv9.py without touching the model code.

patch_shared_qkv(model)
    Prefill fast path for TTT layers: compute Q/K/V + RoPE once, run
    flash_attn twice (teacher window, student window), share g_proj/gate.
    KV cache is still updated so decode continues to work. Assumes no padding
    (batch of one or right-aligned full sequences), which is exactly the
    eval/generate() situation; falls back to the stock path when a mask with
    real padding shows up or q_len < ttt_chunk (decode).

patch_batched_scan(model)
    Replaces the sequential per-chunk output loop in the scanfuse MLP with the
    batched cumsum-scan (one cumsum + one batched matmul, bf16) at inference
    time only (self.training False). Training path untouched.
"""
from __future__ import annotations

import types

import torch
from einops import rearrange

import custom_models.ipttcd.modeling_ipttcdv9 as _v9mod
from custom_models.ipttcd.modeling_ipttcdv9 import _fla_attn


def _mask_has_padding(attention_mask, q_len: int) -> bool:
    if attention_mask is None:
        return False
    # all-ones mask (the generate() default) carries no information
    return bool((attention_mask == 0).any().item())


def patch_shared_qkv(model, use_stream: bool = False) -> int:
    """Bind a shared-QKV prefill forward onto every TTT block. Returns #patched.

    use_stream=True additionally runs the student flash-attention + its output
    projection on a side CUDA stream, overlapping it with the teacher
    attention (they are independent once Q/K/V are shared).
    """
    side_stream = torch.cuda.Stream() if use_stream else None
    n = 0
    for blk in model.model.layers:
        if not getattr(blk, "is_ttt_layer", False):
            continue
        blk._side_stream = side_stream
        blk._stock_forward = blk.forward

        def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                    output_attentions=False, use_cache=False, **kwargs):
            q_len = hidden_states.shape[1]
            if self.training or q_len < self.ttt_chunk or _mask_has_padding(attention_mask, q_len):
                return self._stock_forward(
                    hidden_states, attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions, use_cache=use_cache, **kwargs)

            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            attn = self.attn
            batch_size = hidden_states.shape[0]

            q = rearrange(attn.q_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            k = rearrange(attn.k_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            v = rearrange(attn.v_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            if attn.qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)

            seqlen_offset = 0
            max_seqlen = q_len
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(attn.layer_idx)
                max_seqlen = q_len + seqlen_offset
            if attn.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, attn.max_position_embeddings)
            if not attn.use_nope:
                q, k = attn.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen,
                                   cu_seqlens=kwargs.get('cu_seqlens'))

            if past_key_values is not None:
                cache_has_content = past_key_values.get_seq_length(attn.layer_idx) > 0
                k_c, v_c = past_key_values.update(
                    attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                    layer_idx=attn.layer_idx, offset=q_len,
                    cache_kwargs=dict(window_size=attn.window_size),
                )['attn_state']
                if cache_has_content:
                    k = rearrange(k_c, '... (h d) -> ... h d', d=attn.head_dim)
                    v = rearrange(v_c, '... (h d) -> ... h d', d=attn.head_dim)

            teacher_window = (-1, -1) if attn.window_size is None else (attn.window_size - 1, 0)
            student_window = self.ttt_visible_chunks * self.ttt_chunk

            def _postprocess(o):
                if attn.use_output_gate:
                    g = attn.g_proj(hidden_states)
                    if attn.fuse_norm_and_gate:
                        gg = rearrange(g, '... (h d) -> ... h d', d=attn.head_dim)
                        return rearrange(attn.g_norm_swish_gate(o, gg), '... h d -> ... (h d)')
                    o = rearrange(attn.g_norm(o), '... h d -> ... (h d)')
                    return o * attn.gate_fn(g)
                return o.reshape(batch_size, q_len, -1)

            def _student_tail():
                s_o = _fla_attn.flash_attn_func(q, k, v, causal=True,
                                                window_size=(student_window - 1, 0))
                s_attn = attn.o_proj(_postprocess(s_o))
                if self.config.fuse_norm:
                    out, _ = self.mlp_norm(s_attn, residual, True)
                else:
                    out = self.mlp_norm(residual + s_attn)
                return out

            side = getattr(self, "_side_stream", None)
            if side is not None:
                ev_qkv = torch.cuda.Event()
                ev_qkv.record()
                with torch.cuda.stream(side):
                    side.wait_event(ev_qkv)
                    for tsr in (q, k, v, residual):
                        tsr.record_stream(side)
                    student_output = _student_tail()
                    ev_student = torch.cuda.Event()
                    ev_student.record(side)
                teacher_o = _fla_attn.flash_attn_func(q, k, v, causal=True, window_size=teacher_window)
                teacher_attn_output = attn.o_proj(_postprocess(teacher_o))
                if self.config.fuse_norm:
                    hidden_states, residual = self.mlp_norm(teacher_attn_output, residual, True)
                else:
                    hidden_states = residual + teacher_attn_output
                    residual = hidden_states
                    hidden_states = self.mlp_norm(hidden_states)
                torch.cuda.current_stream().wait_event(ev_student)
            else:
                teacher_o = _fla_attn.flash_attn_func(q, k, v, causal=True, window_size=teacher_window)
                student_output = _student_tail()
                teacher_attn_output = attn.o_proj(_postprocess(teacher_o))
                if self.config.fuse_norm:
                    hidden_states, residual = self.mlp_norm(teacher_attn_output, residual, True)
                else:
                    hidden_states = residual + teacher_attn_output
                    residual = hidden_states
                    hidden_states = self.mlp_norm(hidden_states)

            hidden_states = self.mlp(hidden_states, t=student_output, **kwargs)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (past_key_values,)
            return outputs

        blk.forward = types.MethodType(forward, blk)
        n += 1
    return n


def patch_fast_conv(model) -> int:
    """Replace the depthwise Conv1d in _apply_ttt_conv with a shifted-add
    computed directly on the (b, T, C, d) layout — no rearrange copies, no
    cuDNN depthwise kernel. Mathematically identical (causal, per-chunk):
        y[c] = sum_j w[j] * x[c - (k-1) + j]
    """
    n = 0
    for blk in model.model.layers:
        mlp = getattr(blk, "mlp", None)
        if mlp is None or not getattr(mlp, "is_ttt_layer", False):
            continue

        def _apply_ttt_conv(self, x, conv):
            if conv is None:
                return x
            if not self.ttt_conv_causal:
                from custom_models.ipttcd.modeling_ipttcdv9 import IPTTCDMLP as _B
                return _B._apply_ttt_conv(self, x, conv)  # non-causal fallback
            k = conv.kernel_size[0]
            w = conv.weight.view(-1, k).to(x.dtype)          # (d, k)
            y = x * w[:, k - 1]
            for i in range(1, k):
                y[:, :, i:, :] += x[:, :, :-i, :] * w[:, k - 1 - i]
            return y

        mlp._apply_ttt_conv = types.MethodType(_apply_ttt_conv, mlp)
        n += 1
    return n


def _gateup(mlp, x):
    """gate/up projections; one fused GEMM when patch_fused_gateup installed."""
    w = getattr(mlp, "_fused_gateup_w", None)
    if w is None:
        return mlp.gate_proj(x), mlp.up_proj(x)
    gu = torch.nn.functional.linear(x, w)
    di = mlp.gate_proj.weight.shape[0]
    return gu[..., :di], gu[..., di:]


def patch_fused_gateup(model) -> int:
    """Install a concatenated gate|up weight on every TTT MLP. Only takes
    effect in forwards that call _gateup() (i.e. after patch_batched_scan).
    Mathematically identical; halves the GEMM launches for gate/up.
    """
    n = 0
    for blk in model.model.layers:
        mlp = getattr(blk, "mlp", None)
        if mlp is None or not getattr(mlp, "is_ttt_layer", False):
            continue
        w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
        mlp.register_buffer("_fused_gateup_w", w, persistent=False)
        n += 1
    return n


def patch_fused_attn(model, kernel: str = "triton") -> int:
    """Replace the two flash_attn calls in the inference fast-prefill branch
    with a dual-window implementation: kernel="triton" (one-pass fused Triton
    kernel, FA2-world optimum) or kernel="fa3_lse" (two FA3 calls + one-pass
    LSE merge, FA3-world optimum; student output bitwise = FA3).
    NOT bitwise vs the two-call baseline — validate with eval metrics.
    """
    if kernel == "triton":
        from fused_dual_attn import fused_dual_window_attn
    elif kernel == "fa3_lse":
        from lse_dual_attn import lse_dual_window_attn_fa3_fast as fused_dual_window_attn
    else:
        raise ValueError(kernel)

    n = 0
    for blk in model.model.layers:
        if not getattr(blk, "is_ttt_layer", False):
            continue
        blk._pre_fused_forward = blk.forward

        def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                    output_attentions=False, use_cache=False, **kwargs):
            q_len = hidden_states.shape[1]
            import custom_models.ipttcd.modeling_ipttcdv9 as _m
            if (self.training or not _m._use_fast_prefill_path(self, hidden_states, attention_mask, kwargs)):
                return self._pre_fused_forward(
                    hidden_states, attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions, use_cache=use_cache, **kwargs)

            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
            attn = self.attn
            batch_size = hidden_states.shape[0]

            q = rearrange(attn.q_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            k = rearrange(attn.k_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            v = rearrange(attn.v_proj(hidden_states), '... (h d) -> ... h d', d=attn.head_dim)
            if attn.qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)

            seqlen_offset, max_seqlen = 0, q_len
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(attn.layer_idx)
                max_seqlen = q_len + seqlen_offset
            if attn.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, attn.max_position_embeddings)
            if not attn.use_nope:
                q, k = attn.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen,
                                   cu_seqlens=kwargs.get('cu_seqlens'))

            if past_key_values is not None:
                cache_has_content = past_key_values.get_seq_length(attn.layer_idx) > 0
                if cache_has_content:
                    # fused kernel assumes Tq == Tk (fresh prefill); chunked
                    # prefill with a populated cache falls back to the stock path
                    # (residual still holds the pre-attn_norm input)
                    return self._pre_fused_forward(
                        residual, attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions, use_cache=use_cache, **kwargs)
                past_key_values.update(
                    attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                    layer_idx=attn.layer_idx, offset=q_len,
                    cache_kwargs=dict(window_size=attn.window_size),
                )

            wt = -1 if attn.window_size is None else attn.window_size
            ws = self.ttt_visible_chunks * self.ttt_chunk
            teacher_o, student_o = fused_dual_window_attn(q, k, v, wt, ws)

            teacher_o = teacher_o.reshape(batch_size, q_len, -1)
            student_o = student_o.reshape(batch_size, q_len, -1)
            teacher_attn_output = attn.o_proj(teacher_o)
            student_attn_output = attn.o_proj(student_o)

            if self.config.fuse_norm:
                student_output, _ = self.mlp_norm(student_attn_output, residual, True)
                hidden_states, residual = self.mlp_norm(teacher_attn_output, residual, True)
            else:
                student_output = self.mlp_norm(residual + student_attn_output)
                hidden_states = residual + teacher_attn_output
                residual = hidden_states
                hidden_states = self.mlp_norm(hidden_states)

            hidden_states = self.mlp(hidden_states, t=student_output, **kwargs)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (past_key_values,)
            return outputs

        blk.forward = types.MethodType(forward, blk)
        n += 1
    return n


def patch_mlp_stream(model) -> int:
    """Overlap the MLP's student pipeline (gate/up GEMMs + swiglu + student
    conv + beta) on the inference side stream with the teacher z path, joining
    before the diff/ΔW computation. Scan tail = batched cumsum (bit-exact).
    """
    from custom_models.ipttcd.modeling_ipttcdv9 import _get_infer_side_stream, swiglu
    import torch.nn.functional as F

    n = 0
    for blk in model.model.layers:
        mlp = getattr(blk, "mlp", None)
        if mlp is None or not getattr(mlp, "is_ttt_layer", False):
            continue
        mlp._stock_forward_ms = mlp.forward

        def forward(self, x, t=None, **kwargs):
            side = None if self.training else _get_infer_side_stream()
            if (t is None or not self.is_ttt_layer or self.ttt_weight_renorm
                    or side is None):
                return self._stock_forward_ms(x, t, **kwargs)

            ev_in = torch.cuda.Event()
            ev_in.record()
            with torch.cuda.stream(side):
                side.wait_event(ev_in)
                t.record_stream(side)
                t_padded = self._padding(t)
                b_, T_, C_, _ = t_padded.shape
                t_flat = t_padded.reshape(b_, T_ * C_, -1)
                z_hat = swiglu(self.gate_proj(t_flat), self.up_proj(t_flat))
                z_hat_padded = z_hat.reshape(b_, T_, C_, -1)
                z_hat_padded = self._apply_ttt_conv(z_hat_padded, self.ttt_student_conv)
                beta_padded = None
                if self.dynamic_lr:
                    beta_padded = (t_padded @ self.b_proj).unsqueeze(-1).sigmoid()
                ev_stud = torch.cuda.Event()
                ev_stud.record(side)

            gate, up = self.gate_proj(x), self.up_proj(x)
            z = swiglu(gate, up)
            seq_len = x.shape[1]
            z_padded = self._padding(z)
            b, T, C, _ = z_padded.shape
            compute_dtype = z_padded.dtype
            z_target_padded = self._apply_ttt_conv(z_padded, self.ttt_teacher_conv)
            teacher_output_padded = (
                z_target_padded if self.ttt_output_use_teacher_conv else z_padded
            )
            torch.cuda.current_stream().wait_event(ev_stud)

            z_output_padded = self._select_ttt_output_padded(
                teacher_output_padded=teacher_output_padded,
                student_output_padded=z_hat_padded,
            )

            W_base_t = self.down_proj.weight.transpose(0, 1).to(compute_dtype).unsqueeze(0).unsqueeze(0)
            ttt_proj_w = None
            if self.ttt_proj is not None:
                ttt_proj_w = self.ttt_proj.weight.to(compute_dtype).unsqueeze(0).unsqueeze(0)
            fuse = (self.ttt_fuse_proj_into_update and ttt_proj_w is not None
                    and not self.ttt_norm_proj_diff)
            W_update_t = torch.matmul(W_base_t, ttt_proj_w) if fuse else W_base_t

            if T > 1:
                diff = z_target_padded[:, :-1] - z_hat_padded[:, :-1]
                proj_diff = torch.matmul(diff, W_update_t)
                if self.ttt_norm_proj_diff:
                    proj_diff = F.normalize(proj_diff, p=2, dim=-1, eps=1e-6)
                z_hat_norm = (
                    F.normalize(z_hat_padded[:, :-1], p=2, dim=-1, eps=1e-6)
                    if self.ttt_normalize_student_features else z_hat_padded[:, :-1]
                )
                if self.dynamic_lr:
                    proj_diff = proj_diff * beta_padded[:, :-1]
                if ttt_proj_w is not None and not fuse:
                    proj_diff = torch.matmul(proj_diff, ttt_proj_w)
                d_w_chunks = torch.matmul(proj_diff.transpose(-1, -2), z_hat_norm)
                W_base = self.down_proj.weight.to(compute_dtype).unsqueeze(0).unsqueeze(0).expand(b, 1, -1, -1)
                lr = torch.tensor(self.ttt_lr, dtype=compute_dtype, device=x.device)
                w_states = torch.cat([W_base, lr * d_w_chunks], dim=1)
                w_run = torch.cumsum(w_states, dim=1)
            else:
                w_run = self.down_proj.weight.to(compute_dtype).unsqueeze(0).unsqueeze(0)

            output = torch.matmul(z_output_padded, w_run.transpose(-1, -2))
            return rearrange(output, "b t c d -> b (t c) d")[:, :seq_len, :]

        mlp.forward = types.MethodType(forward, mlp)
        n += 1
    return n


def patch_triton_conv(model) -> int:
    """Replace _apply_ttt_conv with the Triton causal depthwise conv
    (bitwise-equal to the cuDNN path, ~11x faster incl. removed rearranges)."""
    from fused_conv import causal_dwconv_chunked

    n = 0
    for blk in model.model.layers:
        mlp = getattr(blk, "mlp", None)
        if mlp is None or not getattr(mlp, "is_ttt_layer", False):
            continue

        def _apply_ttt_conv(self, x, conv):
            if conv is None:
                return x
            if not self.ttt_conv_causal:
                from custom_models.ipttcd.modeling_ipttcdv9 import IPTTCDMLP as _B
                return _B._apply_ttt_conv(self, x, conv)
            return causal_dwconv_chunked(x, conv)

        mlp._apply_ttt_conv = types.MethodType(_apply_ttt_conv, mlp)
        n += 1
    return n


def patch_batched_scan(model) -> int:
    """Bind a batched (cumsum + bmm) inference scan onto every TTT MLP."""
    n = 0
    for blk in model.model.layers:
        mlp = getattr(blk, "mlp", None)
        if mlp is None or not getattr(mlp, "is_ttt_layer", False):
            continue
        mlp._stock_forward = mlp.forward

        def forward(self, x, t=None, **kwargs):
            if self.training or t is None or not self.is_ttt_layer or self.ttt_weight_renorm:
                return self._stock_forward(x, t, **kwargs)

            from custom_models.ipttcd.modeling_ipttcdv9 import swiglu
            import torch.nn.functional as F

            gate, up = _gateup(self, x)
            z = swiglu(gate, up)
            seq_len = x.shape[1]

            z_padded = self._padding(z)
            b, T, C, _ = z_padded.shape
            compute_dtype = z_padded.dtype
            z_target_padded = self._apply_ttt_conv(z_padded, self.ttt_teacher_conv)
            teacher_output_padded = (
                z_target_padded if self.ttt_output_use_teacher_conv else z_padded
            )

            t_padded = self._padding(t)
            t_flat = t_padded.reshape(b, T * C, -1)
            z_hat = swiglu(*_gateup(self, t_flat))
            z_hat_padded = z_hat.reshape(b, T, C, -1)
            z_hat_padded = self._apply_ttt_conv(z_hat_padded, self.ttt_student_conv)

            z_output_padded = self._select_ttt_output_padded(
                teacher_output_padded=teacher_output_padded,
                student_output_padded=z_hat_padded,
            )

            beta_padded = None
            if self.dynamic_lr:
                beta_padded = (t_padded @ self.b_proj).unsqueeze(-1).sigmoid()

            W_base_t = self.down_proj.weight.transpose(0, 1).to(compute_dtype).unsqueeze(0).unsqueeze(0)
            ttt_proj_w = None
            if self.ttt_proj is not None:
                ttt_proj_w = self.ttt_proj.weight.to(compute_dtype).unsqueeze(0).unsqueeze(0)

            fuse = (self.ttt_fuse_proj_into_update and ttt_proj_w is not None
                    and not self.ttt_norm_proj_diff)
            W_update_t = torch.matmul(W_base_t, ttt_proj_w) if fuse else W_base_t

            if T > 1:
                diff = z_target_padded[:, :-1] - z_hat_padded[:, :-1]
                proj_diff = torch.matmul(diff, W_update_t)
                if self.ttt_norm_proj_diff:
                    proj_diff = F.normalize(proj_diff, p=2, dim=-1, eps=1e-6)
                z_hat_norm = (
                    F.normalize(z_hat_padded[:, :-1], p=2, dim=-1, eps=1e-6)
                    if self.ttt_normalize_student_features else z_hat_padded[:, :-1]
                )
                if self.dynamic_lr:
                    proj_diff = proj_diff * beta_padded[:, :-1]
                if ttt_proj_w is not None and not fuse:
                    proj_diff = torch.matmul(proj_diff, ttt_proj_w)
                d_w_chunks = torch.matmul(proj_diff.transpose(-1, -2), z_hat_norm)
                W_base = self.down_proj.weight.to(compute_dtype).unsqueeze(0).unsqueeze(0).expand(b, 1, -1, -1)
                lr = torch.tensor(self.ttt_lr, dtype=compute_dtype, device=x.device)
                w_states = torch.cat([W_base, lr * d_w_chunks], dim=1)
                w_run = torch.cumsum(w_states, dim=1)
            else:
                w_run = self.down_proj.weight.to(compute_dtype).unsqueeze(0).unsqueeze(0)

            output = torch.matmul(z_output_padded, w_run.transpose(-1, -2))
            return rearrange(output, "b t c d -> b (t c) d")[:, :seq_len, :]

        mlp.forward = types.MethodType(forward, mlp)
        n += 1
    return n
