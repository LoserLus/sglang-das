# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 DiT: fused in-place Q/K RMSNorm + partial NeoX RoPE (HCU-only).

The DiT attention path applies an affine RMSNorm on Q/K (``head_dim=128``)
followed by a 96-of-128 dim partial NeoX rotation through the native
``_apply_rope_qk`` chain. This mirrors ``fused_inplace_qknorm_rope``
(``rope/qknorm_rope_jit.py``), which the model already uses when available,
but that kernel emits CUDA source and is explicitly disabled on HCU
(``_can_use_fused_qknorm_rope`` returns ``False`` for ``is_hcu()``). This
Triton kernel fills that gap: same in-place contract, HCU only.

Not bit-exact: rounds the normalized value to the activation dtype (bf16)
before rotating, one rounding step earlier than the split RMSNorm + RoPE
reference (which keeps the affine-scaled norm output in fp32 through the
rotation). Verified ``max_abs_err`` **< 0.04** at bf16 against the split
HCU reference chain for ``head_dim=128``, ``rotary_dim=96`` (see
``test_minimax_h3_dit_qknorm_rope.py``).

Only engages on HCU, bf16, inference, ``head_dim=128``, ``rope_dim=96``,
matching affine RMSNorm eps. Falls back to the existing
``_apply_qk_norm`` + ``_apply_rope_qk`` chain otherwise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op

_HEAD_DIM = 128
_ROTARY_DIM = 96


@triton.jit
def _minimax_h3_dit_qknorm_rope_kernel(
    Q,
    K,
    WQ,
    WK,
    CACHE,
    POS,
    Q_ROW: tl.constexpr,
    K_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    D: tl.constexpr = _HEAD_DIM,
    R: tl.constexpr = _ROTARY_DIM,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.arange(0, D)
    qptr = Q + token * Q_ROW + head * Q_HEAD + i
    kptr = K + token * K_ROW + head * K_HEAD + i
    q = tl.load(qptr).to(tl.float32)
    k = tl.load(kptr).to(tl.float32)
    wq = tl.load(WQ + i).to(tl.float32)
    wk = tl.load(WK + i).to(tl.float32)
    qscale = tl.rsqrt(tl.sum(q * q, 0) / D + EPS)
    kscale = tl.rsqrt(tl.sum(k * k, 0) / D + EPS)
    # Round to the activation dtype before RoPE, matching the affine
    # RMSNorm's own output dtype at that point in the split chain.
    qn = (q * qscale * wq).to(tl.bfloat16).to(tl.float32)
    kn = (k * kscale * wk).to(tl.bfloat16).to(tl.float32)
    partner = tl.where(i < R // 2, i + R // 2, tl.where(i < R, i - R // 2, i))
    qp = tl.gather(qn, partner, 0)
    kp = tl.gather(kn, partner, 0)
    pos = tl.load(POS + token)
    freq = i % (R // 2)
    cos = tl.load(CACHE + pos * R + freq).to(tl.float32)
    sin = tl.load(CACHE + pos * R + R // 2 + freq).to(tl.float32)
    sign = tl.where(i < R // 2, -1.0, 1.0)
    qo = tl.where(i < R, qn * cos + sign * qp * sin, qn)
    ko = tl.where(i < R, kn * cos + sign * kp * sin, kn)
    tl.store(qptr, qo)
    tl.store(kptr, ko)


def can_use_minimax_h3_dit_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps_q: float,
    eps_k: float,
) -> bool:
    """True when ``q``/``k`` match the H3 DiT attention block's exact signature."""
    if q.shape != k.shape or q.shape[-1] != _HEAD_DIM:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return False
    if q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16:
        return False
    if q_weight.numel() != _HEAD_DIM or k_weight.numel() != _HEAD_DIM:
        return False
    if eps_q != eps_k:
        return False
    return (
        cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.shape[-1] == _ROTARY_DIM
    )


@register_custom_op(mutates_args=["q", "k"])
def minimax_h3_dit_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> None:
    """In-place fused affine RMSNorm + partial NeoX RoPE for the H3 DiT.

    ``q``/``k`` are ``[tokens, heads, 128]`` contiguous-in-last-dim views;
    rotated and normalized in place. ``cos_sin_cache`` is
    ``[tokens, 96] = [cos_half(48) | sin_half(48)]``. Call
    ``can_use_minimax_h3_dit_qknorm_rope`` first; this function does not
    re-validate every condition.
    """
    grid = (q.shape[0], q.shape[1])
    _minimax_h3_dit_qknorm_rope_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        q.stride(1),
        k.stride(1),
        eps,
        num_warps=4,
    )


__all__ = ["can_use_minimax_h3_dit_qknorm_rope", "minimax_h3_dit_qknorm_rope"]
