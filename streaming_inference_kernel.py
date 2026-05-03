import triton
import triton.language as tl
import torch


@triton.jit
def streaming_ssm_inplace_kernel(
    vhat_ptr,       # [B, H]   interpolated observation at t
    h_ptr,          # [B, H]   state -- READ and WRITE in place
    Abar_ptr,       # [H]      diagonal A_bar (precomputed for this dt)
    Bbar_ptr,       # [H]      diagonal B_bar (precomputed for this dt)
    C_ptr,          # [B, H]   output matrix row for this timestep
    D_ptr,          # [H]      skip connection
    y_ptr,          # [B, H]   output
    B, H,
    stride_vb, stride_vh,
    stride_hb, stride_hh,
    stride_cb, stride_ch,
    BLOCK_H: tl.constexpr,
):
    b    = tl.program_id(0)
    hblk = tl.program_id(1)
    hoff = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmsk = hoff < H

    vhat   = tl.load(vhat_ptr + b*stride_vb + hoff, mask=hmsk, other=0.0)
    h_prev = tl.load(h_ptr    + b*stride_hb + hoff, mask=hmsk, other=0.0)
    Abar   = tl.load(Abar_ptr + hoff, mask=hmsk, other=0.0)
    Bbar   = tl.load(Bbar_ptr + hoff, mask=hmsk, other=0.0)
    C_t    = tl.load(C_ptr    + b*stride_cb + hoff, mask=hmsk, other=0.0)
    D_v    = tl.load(D_ptr    + hoff, mask=hmsk, other=0.0)

    h_new = Abar * h_prev + Bbar * vhat
    y_t   = C_t * h_new + D_v * vhat

    tl.store(h_ptr  + b*stride_hb + hoff, h_new, mask=hmsk)
    tl.store(y_ptr  + b*stride_vb + hoff, y_t,   mask=hmsk)


@triton.jit
def streaming_ssm_inplace_bf16_kernel(
    vhat_ptr, h_ptr, Abar_ptr, Bbar_ptr, C_ptr, D_ptr, y_ptr,
    B, H,
    stride_vb, stride_vh,
    stride_hb, stride_hh,
    stride_cb, stride_ch,
    BLOCK_H: tl.constexpr,
):
    b    = tl.program_id(0)
    hblk = tl.program_id(1)
    hoff = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmsk = hoff < H

    vhat   = tl.load(vhat_ptr + b*stride_vb + hoff, mask=hmsk,
                     other=tl.zeros([BLOCK_H], dtype=tl.bfloat16)).to(tl.float32)
    h_prev = tl.load(h_ptr    + b*stride_hb + hoff, mask=hmsk,
                     other=tl.zeros([BLOCK_H], dtype=tl.bfloat16)).to(tl.float32)
    Abar   = tl.load(Abar_ptr + hoff, mask=hmsk,
                     other=tl.zeros([BLOCK_H], dtype=tl.bfloat16)).to(tl.float32)
    Bbar   = tl.load(Bbar_ptr + hoff, mask=hmsk,
                     other=tl.zeros([BLOCK_H], dtype=tl.bfloat16)).to(tl.float32)
    C_t    = tl.load(C_ptr    + b*stride_cb + hoff, mask=hmsk,
                     other=tl.zeros([BLOCK_H], dtype=tl.bfloat16)).to(tl.float32)
    D_v    = tl.load(D_ptr    + hoff, mask=hmsk,
                     other=tl.zeros([BLOCK_H], dtype=tl.bfloat16)).to(tl.float32)

    h_new = Abar * h_prev + Bbar * vhat
    y_t   = C_t * h_new + D_v * vhat

    tl.store(h_ptr  + b*stride_hb + hoff, h_new.to(tl.bfloat16), mask=hmsk)
    tl.store(y_ptr  + b*stride_vb + hoff, y_t.to(tl.bfloat16),   mask=hmsk)


@triton.jit
def streaming_interp_then_ssm_kernel(
    # raw ring-buffer of last K+1 observations
    ring_V_ptr,    # [B, K+1, H]  circular
    ring_M_ptr,    # [B, K+1, H]
    ring_tau_ptr,  # [B, K+1]
    ring_head_ptr, # [B]  int32, current write position in ring
    # new observation to insert
    v_new_ptr,     # [B, H]
    m_new_ptr,     # [B, H]
    tau_new_ptr,   # [B]
    # ssm state
    h_ptr,         # [B, H]  in-place
    Abar_ptr,      # [H]
    Bbar_ptr,      # [H]
    C_ptr,         # [B, H]
    D_ptr,         # [H]
    y_ptr,         # [B, H] output
    B, H, K,
    stride_rb, stride_rk, stride_rh,
    stride_vb, stride_vh,
    stride_hb, stride_hh,
    stride_cb, stride_ch,
    BLOCK_H: tl.constexpr,
    KRING:   tl.constexpr,   # K+1, must be constexpr for ring indexing
):
    b    = tl.program_id(0)
    hblk = tl.program_id(1)
    hoff = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hmsk = hoff < H

    # 1. write new observation into ring buffer
    head = tl.load(ring_head_ptr + b).to(tl.int32)
    new_head = (head + 1) % KRING

    v_new = tl.load(v_new_ptr + b*stride_vb + hoff, mask=hmsk, other=0.0)
    m_new = tl.load(m_new_ptr + b*stride_vb + hoff, mask=hmsk, other=0.0)
    tau_n = tl.load(tau_new_ptr + b)

    tl.store(ring_V_ptr   + b*stride_rb + new_head*stride_rk + hoff, v_new, mask=hmsk)
    tl.store(ring_M_ptr   + b*stride_rb + new_head*stride_rk + hoff, m_new, mask=hmsk)
    if hblk == 0:
        tl.store(ring_tau_ptr + b*KRING + new_head, tau_n)
        tl.store(ring_head_ptr + b, new_head)

    # 2. interpolate from ring buffer (backward search only -- no forward in streaming)
    vhat     = v_new
    prev_k   = -1
    found_p  = 0

    for k in range(1, KRING):
        rk = (new_head - k + KRING) % KRING
        pm = tl.load(ring_M_ptr + b*stride_rb + rk*stride_rk + hoff,
                     mask=hmsk, other=0.0)
        any_v   = tl.sum(pm) > 0.5
        prev_k  = tl.where((~found_p) & any_v, rk, prev_k)
        found_p = tl.where((~found_p) & any_v, 1, found_p)

    # streaming: no future neighbor, forward-fill only
    have_prev = (prev_k >= 0) & (tl.sum(m_new) < 0.5)   # only fill if current is missing

    tau_prev = tl.load(ring_tau_ptr + b*KRING + prev_k,
                       mask=(prev_k >= 0), other=0.0)

    v_prev = tl.load(ring_V_ptr + b*stride_rb + prev_k*stride_rk + hoff,
                     mask=(hmsk & (prev_k >= 0)), other=0.0)

    # for streaming we only have prev, use forward fill (w=0 -> use prev)
    vhat = tl.where(tl.sum(m_new) > 0.5, v_new, vhat)
    vhat = tl.where(have_prev, v_prev, vhat)

    # 3. SSM step
    h_prev = tl.load(h_ptr  + b*stride_hb + hoff, mask=hmsk, other=0.0)
    Abar   = tl.load(Abar_ptr + hoff, mask=hmsk, other=0.0)
    Bbar   = tl.load(Bbar_ptr + hoff, mask=hmsk, other=0.0)
    C_t    = tl.load(C_ptr    + b*stride_cb + hoff, mask=hmsk, other=0.0)
    D_v    = tl.load(D_ptr    + hoff, mask=hmsk, other=0.0)

    h_new_state = Abar * h_prev + Bbar * vhat
    y_t         = C_t * h_new_state + D_v * vhat

    tl.store(h_ptr + b*stride_hb + hoff, h_new_state, mask=hmsk)
    tl.store(y_ptr + b*stride_vb + hoff, y_t, mask=hmsk)


@triton.jit
def mean_pool_logit_kernel(
    h_ptr,      # [B, T, H] hidden states
    W_ptr,      # [H]  mortality logit weight (scalar output so just dot)
    bias_ptr,   # []   scalar bias
    out_ptr,    # [B]  logits
    B, T, H,
    stride_b, stride_t, stride_h,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)

    acc = tl.zeros([1], dtype=tl.float32)

    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_msk  = t_offs < T

        for h_start in range(0, H, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_msk  = h_offs < H

            W = tl.load(W_ptr + h_offs, mask=h_msk, other=0.0)

            # load [BLOCK_T, BLOCK_H] slice of h
            # triton doesn't do 2d indexing nicely so flatten
            for t_i in tl.static_range(BLOCK_T):
                if t_start + t_i < T:
                    h_row = tl.load(h_ptr + b*stride_b + (t_start+t_i)*stride_t + h_offs,
                                    mask=h_msk, other=0.0)
                    acc += tl.sum(W * h_row)

    mean_val = acc[0] / (T * H + 1e-9)
    bias = tl.load(bias_ptr)
    tl.store(out_ptr + b, mean_val + bias)


@triton.jit
def masked_fill_zero_kernel(
    x_ptr, mask_ptr, out_ptr,
    B, T, H,
    BLOCK: tl.constexpr,
):
    # zero out positions where mask=0; used for padding in variable-length batches
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    N    = B * T * H
    msk  = offs < N

    # mask is [B, T] broadcast over H
    # get batch and time index from flat offset
    bt   = offs // H
    m_v  = tl.load(mask_ptr + bt, mask=msk, other=1)
    x    = tl.load(x_ptr    + offs, mask=msk, other=0.0)
    tl.store(out_ptr + offs, tl.where(m_v > 0, x, 0.0), mask=msk)
