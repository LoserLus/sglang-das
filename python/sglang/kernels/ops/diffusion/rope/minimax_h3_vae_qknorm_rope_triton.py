# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 video VAE ViT: fused Q/K RMSNorm + partial NeoX RoPE (HCU-only).

The VAE ViT attention path runs an unaffine RMSNorm on Q/K followed by a
48-of-64 dim partial NeoX rotation (``rope_dim_ratio=0.75``) applied through
the HCU ``minimax_h3_rotary_embedding`` op. Both steps read/write the
strided ``[B, S, H, 3*head_dim]`` view produced by chunking the fused
``to_qkv`` projection (head stride 192, since the layout is
``Q0..63 K0..63 V0..63`` per head, not head-major QKV).

This kernel fuses the two steps into one Triton launch per Q/K pair,
reading the interleaved view directly and writing contiguous outputs.  It
is **not bit-exact**: the reference path accumulates the RMSNorm reduction
in fp32 via the CUDA/HIP ``RMSNorm`` kernel and then runs a separate
fp32-accumulating rotary kernel, while this kernel accumulates the square
sum in fp32 but rounds the normalized value to the activation dtype before
rotating (so the output differs from the reference by about one bf16/fp16
rounding step; verified ``max_abs_err`` **< 0.004** at fp16 against the
existing HCU chain, see ``test_minimax_h3_vae_qknorm_rope.py``).

Only engages for the exact signature the H3 VAE ViT decoder uses:
unaffine RMSNorm, ``head_dim=64``, ``rotary_dim=48``, fp16, batch size 1,
HCU, inference. Falls back to the existing RMSNorm + rotary chain for every
other input (different dtype, head_dim, batch size, or an affine norm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from torch import nn

_HEAD_DIM = 64
_ROTARY_DIM = 48


@triton.jit
def _minimax_h3_vae_qknorm_rope_kernel(
    Q,
    K,
    QOUT,
    KOUT,
    CACHE,
    POS,
    IN_ROW: tl.constexpr,
    IN_HEAD: tl.constexpr,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    D: tl.constexpr = _HEAD_DIM,
    R: tl.constexpr = _ROTARY_DIM,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.arange(0, D)
    q = tl.load(Q + token * IN_ROW + head * IN_HEAD + i).to(tl.float32)
    k = tl.load(K + token * IN_ROW + head * IN_HEAD + i).to(tl.float32)
    # Unaffine RMSNorm (elementwise_affine=False): no weight multiply.
    qn = (q * tl.rsqrt(tl.sum(q * q, 0) / D + EPS)).to(QOUT.dtype.element_ty).to(
        tl.float32
    )
    kn = (k * tl.rsqrt(tl.sum(k * k, 0) / D + EPS)).to(KOUT.dtype.element_ty).to(
        tl.float32
    )
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
    tl.store(QOUT + token * OUT_ROW + head * OUT_HEAD + i, qo)
    tl.store(KOUT + token * OUT_ROW + head * OUT_HEAD + i, ko)


def can_use_minimax_h3_vae_qknorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_pos_emb: tuple | None,
    norm_q: "nn.Module",
    norm_k: "nn.Module",
) -> bool:
    """True when ``query``/``key`` match the H3 VAE ViT decoder's exact signature.

    ``rotary_pos_emb`` is the 4-tuple ``(_, _, cache, positions)`` the VAE
    attention module threads through; only ``cache``'s dtype/shape matter here.
    """
    if (
        query.dim() != 4
        or query.shape[0] != 1
        or query.shape[-1] != _HEAD_DIM
        or key.shape != query.shape
    ):
        return False
    if query.dtype != torch.float16 or key.dtype != torch.float16:
        return False
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return False
    if not isinstance(norm_q, torch.nn.RMSNorm) or not isinstance(
        norm_k, torch.nn.RMSNorm
    ):
        return False
    if norm_q.weight is not None or norm_k.weight is not None:
        return False
    if norm_q.eps != norm_k.eps:
        return False
    if rotary_pos_emb is None or len(rotary_pos_emb) != 4:
        return False
    cache = rotary_pos_emb[2]
    seq_len = query.shape[1]
    return (
        cache.dtype == query.dtype
        and cache.shape == (seq_len, _ROTARY_DIM)
    )


def minimax_h3_vae_qknorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused unaffine RMSNorm + partial NeoX RoPE for the H3 VAE ViT.

    ``query``/``key`` are the ``[1, S, H, 64]`` strided chunks of the fused
    QKV projection (head stride 192). ``cache`` is the ``[S, 48]``
    ``[cos_half(24) | sin_half(24)]`` table, ``positions`` is ``arange(S)``.
    Returns contiguous ``(query, key)``, or ``None`` when the caller should
    fall back to the split RMSNorm + rotary chain (call
    ``can_use_minimax_h3_vae_qknorm_rope`` first; this function does not
    re-validate every condition).
    """
    if query.shape[-1] != _HEAD_DIM or cache.shape[-1] != _ROTARY_DIM:
        return None
    qout = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    kout = torch.empty_like(qout)
    grid = (query.shape[1], query.shape[2])
    _minimax_h3_vae_qknorm_rope_kernel[grid](
        query,
        key,
        qout,
        kout,
        cache,
        positions,
        query.stride(1),
        query.stride(2),
        qout.stride(1),
        qout.stride(2),
        eps,
        num_warps=4,
    )
    return qout, kout


__all__ = ["can_use_minimax_h3_vae_qknorm_rope", "minimax_h3_vae_qknorm_rope"]
