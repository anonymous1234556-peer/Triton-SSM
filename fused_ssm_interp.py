import triton
import triton.language as tl
import torch


@triton.jit
def _combine_fwd(aL, xL, aR, xR):
    return aR * aL, aR * xL + xR


@triton.jit
def fused_interp_ssm_step_kernel(
    # interp inputs
    V_ptr, M_ptr, tau_ptr,
    # ssm params (discretized, per feature)
    A_bar_ptr,    # [F] diagonal A-bar
    B_bar_ptr,    # [F] diagonal B-bar projected
    # carry state  h_{t-1} for this (b, f) tile
    h_in_ptr,     # [B, F]
    # outputs
    Vhat_ptr,     # [B, T, F]
    h_out_ptr,    # [B, F]
    B, T, F,
    stride_vb, stride_vt, stride_vf,
    stride_mb, stride_mt, stride_mf,
    stride_tb, stride_tt,
    stride_ob, stride_ot, stride_of,
    stride_hib, stride_hif,
    stride_hob, stride_hof,
    t_cur,        # which time step we're computing (scalar)
    K: tl.constexpr,
    FTILE: tl.constexpr,
):
    b    = tl.program_id(0)
    fblk = tl.program_id(1)

    fng  = fblk * FTILE + tl.arange(0, FTILE)
    fmsk = fng < F

    # ---- interpolation (same as standalone kernel) ----
    m_cur = tl.load(M_ptr + b*stride_mb + t_cur*stride_mt + fng,
                    mask=fmsk, other=0.0)
    v_cur = tl.load(V_ptr + b*stride_vb + t_cur*stride_vt + fng,
                    mask=fmsk, other=0.0)
    tau_t = tl.load(tau_ptr + b*stride_tb + t_cur*stride_tt)

    vhat       = v_cur
    prev_idx   = -1
    found_prev = 0

    for k in range(1, K+1):
        tk = t_cur - k
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk >= 0)), other=0.0)
        prev_idx   = tl.where((~found_prev) & (tl.sum(pm) > 0.5) & (tk >= 0), tk, prev_idx)
        found_prev = tl.where((~found_prev) & (tl.sum(pm) > 0.5) & (tk >= 0), 1, found_prev)

    next_idx   = -1
    found_next = 0

    for k in range(1, K+1):
        tk = t_cur + k
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk < T)), other=0.0)
        next_idx   = tl.where((~found_next) & (tl.sum(pm) > 0.5) & (tk < T), tk, next_idx)
        found_next = tl.where((~found_next) & (tl.sum(pm) > 0.5) & (tk < T), 1, found_next)

    have_both = (prev_idx >= 0) & (next_idx >= 0)
    have_prev = (prev_idx >= 0) & (~have_both)
    have_next = (next_idx >= 0) & (~have_both) & (~have_prev)

    tau_prev = tl.load(tau_ptr + b*stride_tb + prev_idx*stride_tt,
                       mask=(prev_idx >= 0), other=0.0)
    tau_next = tl.load(tau_ptr + b*stride_tb + next_idx*stride_tt,
                       mask=(next_idx >= 0), other=0.0)

    w = tl.where((tau_next - tau_prev) > 1e-9,
                 (tau_t - tau_prev) / (tau_next - tau_prev), 0.5)
    w = tl.minimum(tl.maximum(w, 0.0), 1.0)

    v_prev = tl.load(V_ptr + b*stride_vb + prev_idx*stride_vt + fng,
                     mask=(fmsk & (prev_idx >= 0)), other=0.0)
    v_next = tl.load(V_ptr + b*stride_vb + next_idx*stride_vt + fng,
                     mask=(fmsk & (next_idx >= 0)), other=0.0)

    interp = (1.0 - w)*v_prev + w*v_next
    vhat = tl.where(m_cur > 0.5, v_cur,  vhat)
    vhat = tl.where(have_both,   interp, vhat)
    vhat = tl.where(have_prev,   v_prev, vhat)
    vhat = tl.where(have_next,   v_next, vhat)

    tl.store(Vhat_ptr + b*stride_ob + t_cur*stride_ot + fng, vhat, mask=fmsk)

    # ---- SSM step fused: h_t = A_bar * h_{t-1} + B_bar * vhat ----
    A_bar = tl.load(A_bar_ptr + fng, mask=fmsk, other=0.0)
    B_bar = tl.load(B_bar_ptr + fng, mask=fmsk, other=0.0)
    h_prev = tl.load(h_in_ptr + b*stride_hib + fng,
                     mask=(fmsk & (b*stride_hib + fng < B*F)), other=0.0)

    h_new = A_bar * h_prev + B_bar * vhat
    tl.store(h_out_ptr + b*stride_hob + fng, h_new, mask=fmsk)


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, L, H,
    stride_b, stride_l, stride_h,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # per-token layernorm, one block per (b, l) pair
    b = tl.program_id(0)
    l = tl.program_id(1)

    offs = tl.arange(0, BLOCK_H)
    msk  = offs < H
    base = b*stride_b + l*stride_l

    x   = tl.load(x_ptr + base + offs, mask=msk, other=0.0)
    mu  = tl.sum(x, axis=0) / H
    x_c = x - mu
    var = tl.sum(x_c * x_c, axis=0) / H
    inv = 1.0 / tl.sqrt(var + EPS)

    w  = tl.load(w_ptr    + offs, mask=msk, other=1.0)
    b_ = tl.load(bias_ptr + offs, mask=msk, other=0.0)

    out = x_c * inv * w + b_
    tl.store(out_ptr + base + offs, out, mask=msk)


@triton.jit
def layernorm_bwd_kernel(
    dout_ptr, x_ptr, w_ptr,
    dx_ptr, dw_ptr, db_ptr,
    B, L, H,
    stride_b, stride_l,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    b   = tl.program_id(0)
    l   = tl.program_id(1)
    offs = tl.arange(0, BLOCK_H)
    msk  = offs < H
    base = b*stride_b + l*stride_l

    x    = tl.load(x_ptr    + base + offs, mask=msk, other=0.0)
    dout = tl.load(dout_ptr + base + offs, mask=msk, other=0.0)
    w    = tl.load(w_ptr    + offs, mask=msk, other=1.0)

    mu   = tl.sum(x, axis=0) / H
    xc   = x - mu
    var  = tl.sum(xc * xc, axis=0) / H
    inv  = 1.0 / tl.sqrt(var + EPS)

    xhat = xc * inv
    dxhat = dout * w

    # dvar
    dvar  = tl.sum(dxhat * xc, axis=0) * (-0.5) * (var + EPS)**(-1.5)
    # dmu
    dmu   = tl.sum(dxhat * (-inv), axis=0) + dvar * tl.sum(-2.0 * xc, axis=0) / H

    dx    = dxhat * inv + dvar * 2.0 * xc / H + dmu / H

    tl.store(dx_ptr + base + offs, dx, mask=msk)
    # weight/bias grads -- accumulated atomically (one token at a time, not ideal)
    tl.atomic_add(dw_ptr + offs, dout * xhat, mask=msk)
    tl.atomic_add(db_ptr + offs, dout, mask=msk)


@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, out_ptr,
    B, L, H,
    stride_b, stride_l,
    BLOCK_H: tl.constexpr,
):
    # SwiGLU: out = silu(gate) * up
    b = tl.program_id(0)
    l = tl.program_id(1)
    offs = tl.arange(0, BLOCK_H)
    msk  = offs < H
    base = b*stride_b + l*stride_l

    gate = tl.load(gate_ptr + base + offs, mask=msk, other=0.0)
    up   = tl.load(up_ptr   + base + offs, mask=msk, other=0.0)
    silu = gate * tl.sigmoid(gate)
    tl.store(out_ptr + base + offs, silu * up, mask=msk)


@triton.jit
def swiglu_bwd_kernel(
    dout_ptr, gate_ptr, up_ptr,
    dgate_ptr, dup_ptr,
    B, L, H,
    stride_b, stride_l,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    offs = tl.arange(0, BLOCK_H)
    msk  = offs < H
    base = b*stride_b + l*stride_l

    gate = tl.load(gate_ptr + base + offs, mask=msk, other=0.0)
    up   = tl.load(up_ptr   + base + offs, mask=msk, other=0.0)
    dout = tl.load(dout_ptr + base + offs, mask=msk, other=0.0)

    sig  = tl.sigmoid(gate)
    silu = gate * sig
    # d silu / d gate = sig + gate * sig * (1 - sig) = sig*(1 + gate*(1-sig))
    dsilu_dgate = sig * (1.0 + gate * (1.0 - sig))

    tl.store(dup_ptr   + base + offs, dout * silu,          mask=msk)
    tl.store(dgate_ptr + base + offs, dout * up * dsilu_dgate, mask=msk)
