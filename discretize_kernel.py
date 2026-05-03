"""
ZOH discretization + selective-scan parameter projection kernels
given time deltas dt that come from the irregular timestamps.

A_bar = exp(dt * A)     (diagonal A assumed)
B_bar = (A_bar - I) * A^{-1} * B   -- ZOH for diagonal case simplifies to
        B_bar_i = (exp(dt*A_i) - 1) / A_i  * B_i
"""

import triton
import triton.language as tl
import torch


@triton.jit
def zoh_discretize_diag_kernel(
    A_ptr,       # [H]   continuous diagonal A (log domain -- stored as log(-a) so a<0)
    B_ptr,       # [B, T, H] continuous B
    dt_ptr,      # [B, T]    time deltas
    Abar_ptr,    # [B, T, H] output A_bar
    Bbar_ptr,    # [B, T, H] output B_bar
    B, T, H,
    stride_Bb, stride_Bt, stride_Bh,
    stride_db, stride_dt,
    stride_ab, stride_at, stride_ah,
    stride_bb, stride_bt, stride_bh,
    BLOCK_H: tl.constexpr,
):
    b    = tl.program_id(0)
    t    = tl.program_id(1)
    hblk = tl.program_id(2)

    hoffset = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask   = hoffset < H

    dt_val = tl.load(dt_ptr + b*stride_db + t*stride_dt)

    # A stored as log(-a) to ensure negativity
    log_neg_a = tl.load(A_ptr + hoffset, mask=hmask, other=-1.0)
    neg_a     = tl.exp(log_neg_a)   # = -a  (positive)
    a_val     = -neg_a

    # A_bar = exp(dt * a)  = exp(-dt * neg_a)
    Abar = tl.exp(dt_val * a_val)

    # B_bar = (exp(dt*a) - 1) / a * b   for diagonal ZOH
    B_cont = tl.load(B_ptr + b*stride_Bb + t*stride_Bt + hoffset,
                     mask=hmask, other=0.0)
    # stable computation: (exp(x)-1)/x ~ 1 + x/2 for small x
    x       = dt_val * a_val
    expm1_x = tl.where(tl.abs(x) > 1e-5,
                       tl.exp(x) - 1.0,
                       x * (1.0 + x * 0.5))
    Bbar    = expm1_x / (a_val + 1e-12) * B_cont

    tl.store(Abar_ptr + b*stride_ab + t*stride_at + hoffset, Abar, mask=hmask)
    tl.store(Bbar_ptr + b*stride_bb + t*stride_bt + hoffset, Bbar, mask=hmask)


@triton.jit
def zoh_discretize_diag_bf16_kernel(
    A_ptr, B_ptr, dt_ptr,
    Abar_ptr, Bbar_ptr,
    B, T, H,
    stride_Bb, stride_Bt, stride_Bh,
    stride_db, stride_dt,
    stride_ab, stride_at, stride_ah,
    stride_bb, stride_bt, stride_bh,
    BLOCK_H: tl.constexpr,
):
    b    = tl.program_id(0)
    t    = tl.program_id(1)
    hblk = tl.program_id(2)

    hoffset = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask   = hoffset < H

    dt_val    = tl.load(dt_ptr + b*stride_db + t*stride_dt).to(tl.float32)
    log_neg_a = tl.load(A_ptr + hoffset, mask=hmask, other=-1.0).to(tl.float32)
    neg_a     = tl.exp(log_neg_a)
    a_val     = -neg_a

    Abar   = tl.exp(dt_val * a_val)
    B_cont = tl.load(B_ptr + b*stride_Bb + t*stride_Bt + hoffset,
                     mask=hmask, other=tl.zeros([BLOCK_H], dtype=tl.bfloat16)).to(tl.float32)

    x      = dt_val * a_val
    expm1  = tl.where(tl.abs(x) > 1e-5, tl.exp(x) - 1.0, x + x*x*0.5)
    Bbar   = expm1 / (a_val + 1e-12) * B_cont

    tl.store(Abar_ptr + b*stride_ab + t*stride_at + hoffset,
             Abar.to(tl.bfloat16), mask=hmask)
    tl.store(Bbar_ptr + b*stride_bb + t*stride_bt + hoffset,
             Bbar.to(tl.bfloat16), mask=hmask)


@triton.jit
def zoh_discretize_bwd_kernel(
    # saved from forward
    A_ptr, B_ptr, dt_ptr, Abar_ptr,
    # incoming grads
    dAbar_ptr, dBbar_ptr,
    # outputs
    dA_ptr, dB_ptr, ddt_ptr,
    B, T, H,
    stride_Bb, stride_Bt, stride_Bh,
    stride_db, stride_dt,
    stride_ab, stride_at, stride_ah,
    stride_bb, stride_bt, stride_bh,
    BLOCK_H: tl.constexpr,
):
    b    = tl.program_id(0)
    t    = tl.program_id(1)
    hblk = tl.program_id(2)

    hoffset = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask   = hoffset < H

    dt_val    = tl.load(dt_ptr  + b*stride_db + t*stride_dt)
    log_neg_a = tl.load(A_ptr   + hoffset, mask=hmask, other=-1.0)
    neg_a     = tl.exp(log_neg_a)
    a_val     = -neg_a
    B_cont    = tl.load(B_ptr   + b*stride_Bb + t*stride_Bt + hoffset, mask=hmask, other=0.0)
    Abar      = tl.load(Abar_ptr + b*stride_ab + t*stride_at + hoffset, mask=hmask, other=0.0)

    dAbar = tl.load(dAbar_ptr + b*stride_ab + t*stride_at + hoffset, mask=hmask, other=0.0)
    dBbar = tl.load(dBbar_ptr + b*stride_bb + t*stride_bt + hoffset, mask=hmask, other=0.0)

    # d Abar / d dt = a_val * Abar
    ddt_from_Abar = tl.sum(dAbar * a_val * Abar)

    # d Bbar / d dt:
    x      = dt_val * a_val
    expm1  = tl.where(tl.abs(x) > 1e-5, tl.exp(x) - 1.0, x + x*x*0.5)
    dexpm1_dx = tl.exp(x)
    # Bbar = expm1/a * B  so d Bbar / d dt = (d expm1 / d dt) / a * B
    #                                       = dexpm1_dx * a / a * B = dexpm1_dx * B
    ddt_from_Bbar = tl.sum(dBbar * dexpm1_dx * B_cont)

    ddt_total = ddt_from_Abar + ddt_from_Bbar

    # d Abar / d a via chain through log_neg_a:  d(exp(dt*a))/d_a = dt*Abar, da/d_log_neg_a = -neg_a
    da_from_Abar = dAbar * dt_val * Abar
    # d Bbar / d a: Bbar = (exp(dt*a)-1)/a * B
    # d/da [(exp(x)-1)/a] = [dt*exp(x)*a - (exp(x)-1)] / a^2  where x=dt*a
    num_deriv = (dexpm1_dx * dt_val * a_val - expm1) / (a_val * a_val + 1e-18)
    da_from_Bbar = dBbar * num_deriv * B_cont
    da_total = da_from_Abar + da_from_Bbar
    # chain through log_neg_a: d_log_neg_a = da * da/d_neg_a * d_neg_a/d_log_neg_a
    #                                      = da * (-1) * neg_a
    d_log_neg_a = da_total * (-neg_a)

    # d Bbar / d B_cont = expm1 / a
    dB_cont = dBbar * expm1 / (a_val + 1e-12)

    tl.atomic_add(dA_ptr  + hoffset, d_log_neg_a, mask=hmask)
    tl.store(dB_ptr  + b*stride_Bb + t*stride_Bt + hoffset, dB_cont, mask=hmask)
    # dt grad is scalar per (b,t) -- store then reduce outside if needed
    if hblk == 0:
        tl.store(ddt_ptr + b*stride_db + t*stride_dt, ddt_total)
    else:
        tl.atomic_add(ddt_ptr + b*stride_db + t*stride_dt, ddt_total)


@triton.jit
def dt_project_softplus_kernel(
    # projects linear dt_raw to positive dt via softplus then clamps
    dt_raw_ptr,  # [B, T, dt_rank]
    W_ptr,       # [dt_rank, H]
    bias_ptr,    # [H]
    dt_out_ptr,  # [B, T, H]
    B, T, H, dt_rank,
    stride_rb, stride_rt, stride_rr,
    stride_wb, stride_wh,   # W strides
    stride_ob, stride_ot, stride_oh,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    b    = tl.program_id(0)
    t    = tl.program_id(1)
    hblk = tl.program_id(2)

    hoffset = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask   = hoffset < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for r_start in range(0, dt_rank, BLOCK_R):
        r_offs = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offs < dt_rank

        dt_raw = tl.load(dt_raw_ptr + b*stride_rb + t*stride_rt + r_offs,
                         mask=r_mask, other=0.0)

        # dt_raw: [BLOCK_R],  W[r, h]: need outer product and sum over r
        # simplified: for each h, sum over r: dt_raw[r] * W[r, h]
        for r_i in tl.static_range(BLOCK_R):
            if r_start + r_i < dt_rank:
                w_row = tl.load(W_ptr + (r_start + r_i)*stride_wb + hoffset,
                                mask=hmask, other=0.0)
                acc = acc + dt_raw[r_i] * w_row

    bias = tl.load(bias_ptr + hoffset, mask=hmask, other=0.0)
    acc  = acc + bias

    # softplus: log(1 + exp(x))
    dt_val = tl.log(1.0 + tl.exp(acc))
    # clamp to [DT_MIN, DT_MAX]
    dt_val = tl.minimum(tl.maximum(dt_val, DT_MIN), DT_MAX)

    tl.store(dt_out_ptr + b*stride_ob + t*stride_ot + hoffset, dt_val, mask=hmask)
