import triton
import triton.language as tl
import torch


# K=10, Ftile=128 as per paper -- dont change these without re-tuning
# grid is (B, T, ceil(F/128)) -- one block per (batch, time, feature-tile)

@triton.jit
def fused_time_aware_interp_kernel(
    V_ptr,       # [B, T, F]  raw observed values
    M_ptr,       # [B, T, F]  binary mask  1=observed 0=missing
    tau_ptr,     # [B, T]     timestamps  (any unit, must be consistent)
    Vhat_ptr,    # [B, T, F]  output interpolated
    B,
    T,
    F,
    stride_vb, stride_vt, stride_vf,
    stride_mb, stride_mt, stride_mf,
    stride_tb, stride_tt,
    stride_ob, stride_ot, stride_of,
    K: tl.constexpr,       # lookback window size
    FTILE: tl.constexpr,   # feature tile = 128
):
    b    = tl.program_id(0)
    t    = tl.program_id(1)
    fblk = tl.program_id(2)

    fng  = fblk * FTILE + tl.arange(0, FTILE)
    fmsk = fng < F

    # coalesced load of mask + values for current (b,t)
    m_cur = tl.load(M_ptr + b*stride_mb + t*stride_mt + fng,
                    mask=fmsk, other=0.0)
    v_cur = tl.load(V_ptr + b*stride_vb + t*stride_vt + fng,
                    mask=fmsk, other=0.0)

    tau_t = tl.load(tau_ptr + b*stride_tb + t*stride_tt)

    # if fully observed just copy -- early out via store & the rest is masked
    vhat = v_cur  # default: copy observed

    # backward neighbor search -- no break in triton, use flag
    prev_idx   = -1
    found_prev = 0

    for k in range(1, K+1):
        tk = t - k
        # load scalar mask for feature-0 as proxy for "any valid"  (paper: per-feature but
        # for the published pseudocode we use the feature-tile mask aggregate)
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk >= 0)), other=0.0)
        # any observed in tile?
        any_obs = tl.sum(pm) > 0.5
        prev_idx  = tl.where((~found_prev) & any_obs & (tk >= 0), tk, prev_idx)
        found_prev = tl.where((~found_prev) & any_obs & (tk >= 0), 1, found_prev)

    # forward neighbor search
    next_idx   = -1
    found_next = 0

    for k in range(1, K+1):
        tk = t + k
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk < T)), other=0.0)
        any_obs = tl.sum(pm) > 0.5
        next_idx   = tl.where((~found_next) & any_obs & (tk < T), tk, next_idx)
        found_next = tl.where((~found_next) & any_obs & (tk < T), 1, found_next)

    # time-weighted interpolation  eq(1) from paper
    have_both = (prev_idx >= 0) & (next_idx >= 0)
    have_prev = (prev_idx >= 0) & (~have_both)
    have_next = (next_idx >= 0) & (~have_both) & (~have_prev)

    tau_prev = tl.load(tau_ptr + b*stride_tb + prev_idx*stride_tt,
                       mask=(prev_idx >= 0), other=0.0)
    tau_next = tl.load(tau_ptr + b*stride_tb + next_idx*stride_tt,
                       mask=(next_idx >= 0), other=0.0)

    denom = tau_next - tau_prev
    w  = tl.where(denom > 1e-9, (tau_t - tau_prev) / denom, 0.5)
    w  = tl.where(w < 0.0, 0.0, w)
    w  = tl.where(w > 1.0, 1.0, w)

    v_prev = tl.load(V_ptr + b*stride_vb + prev_idx*stride_vt + fng,
                     mask=(fmsk & (prev_idx >= 0)), other=0.0)
    v_next = tl.load(V_ptr + b*stride_vb + next_idx*stride_vt + fng,
                     mask=(fmsk & (next_idx >= 0)), other=0.0)

    interp_both = (1.0 - w) * v_prev + w * v_next
    interp_prev = v_prev
    interp_next = v_next

    vhat = tl.where(m_cur > 0.5,  v_cur,        vhat)
    vhat = tl.where(have_both,     interp_both,  vhat)
    vhat = tl.where(have_prev,     interp_prev,  vhat)
    vhat = tl.where(have_next,     interp_next,  vhat)

    tl.store(Vhat_ptr + b*stride_ob + t*stride_ot + fng,
             vhat, mask=fmsk)


@triton.jit
def fused_interp_bf16_kernel(
    V_ptr, M_ptr, tau_ptr, Vhat_ptr,
    B, T, F,
    stride_vb, stride_vt, stride_vf,
    stride_mb, stride_mt, stride_mf,
    stride_tb, stride_tt,
    stride_ob, stride_ot, stride_of,
    K: tl.constexpr,
    FTILE: tl.constexpr,
):
    # identical logic, inputs loaded as bf16 and accumulated in fp32
    b    = tl.program_id(0)
    t    = tl.program_id(1)
    fblk = tl.program_id(2)

    fng  = fblk * FTILE + tl.arange(0, FTILE)
    fmsk = fng < F

    m_cur = tl.load(M_ptr + b*stride_mb + t*stride_mt + fng,
                    mask=fmsk, other=0.0).to(tl.float32)
    v_cur = tl.load(V_ptr + b*stride_vb + t*stride_vt + fng,
                    mask=fmsk, other=tl.zeros([FTILE], dtype=tl.bfloat16)).to(tl.float32)

    tau_t = tl.load(tau_ptr + b*stride_tb + t*stride_tt).to(tl.float32)

    vhat = v_cur
    prev_idx = -1
    found_prev = 0

    for k in range(1, K+1):
        tk = t - k
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk >= 0)), other=0.0).to(tl.float32)
        any_obs    = tl.sum(pm) > 0.5
        prev_idx   = tl.where((~found_prev) & any_obs & (tk >= 0), tk, prev_idx)
        found_prev = tl.where((~found_prev) & any_obs & (tk >= 0), 1, found_prev)

    next_idx   = -1
    found_next = 0

    for k in range(1, K+1):
        tk = t + k
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk < T)), other=0.0).to(tl.float32)
        any_obs    = tl.sum(pm) > 0.5
        next_idx   = tl.where((~found_next) & any_obs & (tk < T), tk, next_idx)
        found_next = tl.where((~found_next) & any_obs & (tk < T), 1, found_next)

    have_both = (prev_idx >= 0) & (next_idx >= 0)
    have_prev = (prev_idx >= 0) & (~have_both)
    have_next = (next_idx >= 0) & (~have_both) & (~have_prev)

    tau_prev = tl.load(tau_ptr + b*stride_tb + prev_idx*stride_tt,
                       mask=(prev_idx >= 0), other=0.0).to(tl.float32)
    tau_next = tl.load(tau_ptr + b*stride_tb + next_idx*stride_tt,
                       mask=(next_idx >= 0), other=0.0).to(tl.float32)

    denom = tau_next - tau_prev
    w = tl.where(denom > 1e-9, (tau_t - tau_prev) / denom, 0.5)
    w = tl.where(w < 0.0, 0.0, w)
    w = tl.where(w > 1.0, 1.0, w)

    v_prev = tl.load(V_ptr + b*stride_vb + prev_idx*stride_vt + fng,
                     mask=(fmsk & (prev_idx >= 0)),
                     other=tl.zeros([FTILE], dtype=tl.bfloat16)).to(tl.float32)
    v_next = tl.load(V_ptr + b*stride_vb + next_idx*stride_vt + fng,
                     mask=(fmsk & (next_idx >= 0)),
                     other=tl.zeros([FTILE], dtype=tl.bfloat16)).to(tl.float32)

    interp_both = (1.0 - w) * v_prev + w * v_next

    vhat = tl.where(m_cur > 0.5,  v_cur,        vhat)
    vhat = tl.where(have_both,     interp_both,  vhat)
    vhat = tl.where(have_prev,     v_prev,       vhat)
    vhat = tl.where(have_next,     v_next,       vhat)

    tl.store(Vhat_ptr + b*stride_ob + t*stride_ot + fng,
             vhat.to(tl.bfloat16), mask=fmsk)


# fp16 variant -- same deal, watch for NaN on extreme vitals (heart rate > 220 etc)
# paper recommends BF16 over FP16 for clinical data
@triton.jit
def fused_interp_fp16_kernel(
    V_ptr, M_ptr, tau_ptr, Vhat_ptr,
    B, T, F,
    stride_vb, stride_vt, stride_vf,
    stride_mb, stride_mt, stride_mf,
    stride_tb, stride_tt,
    stride_ob, stride_ot, stride_of,
    K: tl.constexpr,
    FTILE: tl.constexpr,
):
    b    = tl.program_id(0)
    t    = tl.program_id(1)
    fblk = tl.program_id(2)
    fng  = fblk * FTILE + tl.arange(0, FTILE)
    fmsk = fng < F

    m_cur = tl.load(M_ptr + b*stride_mb + t*stride_mt + fng,
                    mask=fmsk, other=0.0).to(tl.float32)
    v_cur = tl.load(V_ptr + b*stride_vb + t*stride_vt + fng,
                    mask=fmsk, other=tl.zeros([FTILE], dtype=tl.float16)).to(tl.float32)
    tau_t = tl.load(tau_ptr + b*stride_tb + t*stride_tt).to(tl.float32)

    vhat       = v_cur
    prev_idx   = -1
    found_prev = 0

    for k in range(1, K+1):
        tk = t - k
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk >= 0)), other=0.0).to(tl.float32)
        prev_idx   = tl.where((~found_prev) & (tl.sum(pm) > 0.5) & (tk >= 0), tk, prev_idx)
        found_prev = tl.where((~found_prev) & (tl.sum(pm) > 0.5) & (tk >= 0), 1, found_prev)

    next_idx   = -1
    found_next = 0

    for k in range(1, K+1):
        tk = t + k
        pm = tl.load(M_ptr + b*stride_mb + tk*stride_mt + fng,
                     mask=(fmsk & (tk < T)), other=0.0).to(tl.float32)
        next_idx   = tl.where((~found_next) & (tl.sum(pm) > 0.5) & (tk < T), tk, next_idx)
        found_next = tl.where((~found_next) & (tl.sum(pm) > 0.5) & (tk < T), 1, found_next)

    have_both = (prev_idx >= 0) & (next_idx >= 0)
    have_prev = (prev_idx >= 0) & (~have_both)
    have_next = (next_idx >= 0) & (~have_both) & (~have_prev)

    tau_prev = tl.load(tau_ptr + b*stride_tb + prev_idx*stride_tt,
                       mask=(prev_idx >= 0), other=0.0).to(tl.float32)
    tau_next = tl.load(tau_ptr + b*stride_tb + next_idx*stride_tt,
                       mask=(next_idx >= 0), other=0.0).to(tl.float32)

    w = tl.where((tau_next - tau_prev) > 1e-9,
                 (tau_t - tau_prev) / (tau_next - tau_prev), 0.5)
    w = tl.minimum(tl.maximum(w, 0.0), 1.0)

    v_prev = tl.load(V_ptr + b*stride_vb + prev_idx*stride_vt + fng,
                     mask=(fmsk & (prev_idx >= 0)),
                     other=tl.zeros([FTILE], dtype=tl.float16)).to(tl.float32)
    v_next = tl.load(V_ptr + b*stride_vb + next_idx*stride_vt + fng,
                     mask=(fmsk & (next_idx >= 0)),
                     other=tl.zeros([FTILE], dtype=tl.float16)).to(tl.float32)

    interp = (1.0 - w)*v_prev + w*v_next
    vhat = tl.where(m_cur > 0.5, v_cur,  vhat)
    vhat = tl.where(have_both,   interp, vhat)
    vhat = tl.where(have_prev,   v_prev, vhat)
    vhat = tl.where(have_next,   v_next, vhat)

    tl.store(Vhat_ptr + b*stride_ob + t*stride_ot + fng,
             vhat.to(tl.float16), mask=fmsk)
