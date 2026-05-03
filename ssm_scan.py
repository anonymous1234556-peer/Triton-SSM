import torch
import triton
import triton.language as tl

# parallel prefix scan for linear SSM recurrence
#   h_t = A_t * h_{t-1}  +  B_t * u_t
#   y_t = C_t * h_t  +  D * u_t
#
# associative pair representation: (a, x) s.t. h = a*h_init + x
# combine: (aR, xR) o (aL, xL) = (aR*aL,  aR*xL + xR)

@triton.jit
def _combine(aL, xL, aR, xR):
    return aR * aL, aR * xL + xR


@triton.jit
def ssm_fwd_local_kernel(
    u_ptr, A_ptr, B_ptr,
    loc_a_ptr, loc_x_ptr, carry_ptr,
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    BLK: tl.constexpr, NBLK: tl.constexpr,
):
    rid  = tl.program_id(0)   # B*H flattened
    blk  = tl.program_id(1)
    bi   = rid // H
    hi   = rid % H
    nrows = B * H

    a_r = 1.0
    x_r = 0.0

    ts = blk * BLK
    te = tl.where(ts + BLK > L, L, ts + BLK)

    for i in tl.static_range(BLK):
        p = ts + i
        if p < te:
            off  = bi*L*H + p*H + hi
            A_i  = tl.load(A_ptr + off)
            B_i  = tl.load(B_ptr + off)
            u_i  = tl.load(u_ptr + off)
            a_r, x_r = _combine(a_r, x_r, A_i, B_i * u_i)
            tl.store(loc_a_ptr + off, a_r)
            tl.store(loc_x_ptr + off, x_r)

    c_a = rid * NBLK + blk
    c_x = nrows * NBLK + c_a
    tl.store(carry_ptr + c_a, a_r)
    tl.store(carry_ptr + c_x, x_r)


@triton.jit
def ssm_carry_scan_kernel(
    carry_ptr,
    nrows: tl.constexpr,
    NBLK:  tl.constexpr,
):
    rid = tl.program_id(0)
    ks  = tl.arange(0, NBLK)
    a_in = tl.load(carry_ptr + rid*NBLK + ks)
    x_in = tl.load(carry_ptr + nrows*NBLK + rid*NBLK + ks)
    a_out, x_out = tl.associative_scan((a_in, x_in), 0, _combine)
    tl.store(carry_ptr + rid*NBLK + ks, a_out)
    tl.store(carry_ptr + nrows*NBLK + rid*NBLK + ks, x_out)


@triton.jit
def ssm_fwd_apply_kernel(
    loc_a_ptr, loc_x_ptr, carry_ptr,
    h_out_ptr,
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    BLK: tl.constexpr, NBLK: tl.constexpr,
):
    rid  = tl.program_id(0)
    blk  = tl.program_id(1)
    bi   = rid // H
    hi   = rid % H
    nrows = B * H

    if blk == 0:
        a_p, x_p = 1.0, 0.0
    else:
        c_a = rid * NBLK + (blk - 1)
        a_p = tl.load(carry_ptr + c_a)
        x_p = tl.load(carry_ptr + nrows*NBLK + c_a)

    ts = blk * BLK
    te = tl.where(ts + BLK > L, L, ts + BLK)

    for i in tl.static_range(BLK):
        p = ts + i
        if p < te:
            off    = bi*L*H + p*H + hi
            a_loc  = tl.load(loc_a_ptr + off)
            x_loc  = tl.load(loc_x_ptr + off)
            _, h_t = _combine(a_p, x_p, a_loc, x_loc)
            tl.store(h_out_ptr + off, h_t)


@triton.jit
def ssm_output_kernel(
    h_ptr, C_ptr, D_ptr, u_ptr, y_ptr,
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    BLK: tl.constexpr,
):
    # y_t = C_t * h_t  +  D * u_t
    rid = tl.program_id(0)
    blk = tl.program_id(1)
    bi  = rid // H
    hi  = rid % H

    ts = blk * BLK
    te = tl.where(ts + BLK > L, L, ts + BLK)

    for i in tl.static_range(BLK):
        p = ts + i
        if p < te:
            off = bi*L*H + p*H + hi
            h_t = tl.load(h_ptr + off)
            C_t = tl.load(C_ptr + off)
            D_v = tl.load(D_ptr + hi)      # D is per-dimension, not per-time
            u_t = tl.load(u_ptr + off)
            tl.store(y_ptr + off, C_t * h_t + D_v * u_t)


# --- backward kernels ---

@triton.jit
def _combine_bwd(aR, xR, aL, xL):
    # reverse time associativity for dH scan
    return aL * aR, aL * xR + xL


@triton.jit
def ssm_bwd_local_kernel(
    A_ptr, C_ptr, dY_ptr,
    loc_a_ptr, loc_dh_ptr, carry_ptr,
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    BLK: tl.constexpr, NBLK: tl.constexpr,
):
    rid  = tl.program_id(0)
    blk  = tl.program_id(1)
    bi   = rid // H
    hi   = rid % H
    nrows = B * H

    a_r = 1.0
    x_r = 0.0

    ts = blk * BLK
    te = tl.where(ts + BLK > L, L, ts + BLK)

    for i in tl.static_range(BLK):
        p = (te - 1) - i
        if p >= ts:
            off = bi*L*H + p*H + hi
            # need A_{t+1} for backward recurrence
            off_np1 = bi*L*H + (p+1)*H + hi
            A_tp1   = tl.load(A_ptr + off_np1, mask=(p+1 < L), other=0.0)
            C_t     = tl.load(C_ptr + off)
            dY_t    = tl.load(dY_ptr + off)
            a_r, x_r = _combine_bwd(a_r, x_r, A_tp1, C_t * dY_t)
            tl.store(loc_a_ptr  + off, a_r)
            tl.store(loc_dh_ptr + off, x_r)

    c_a = rid * NBLK + blk
    c_x = nrows * NBLK + c_a
    tl.store(carry_ptr + c_a, a_r)
    tl.store(carry_ptr + c_x, x_r)


@triton.jit
def ssm_bwd_carry_scan_kernel(
    carry_ptr,
    nrows: tl.constexpr,
    NBLK:  tl.constexpr,
):
    rid  = tl.program_id(0)
    ks   = tl.arange(0, NBLK)
    a_in = tl.load(carry_ptr + rid*NBLK + ks)
    x_in = tl.load(carry_ptr + nrows*NBLK + rid*NBLK + ks)
    a_out, x_out = tl.associative_scan((a_in, x_in), 0, _combine_bwd, reverse=True)
    tl.store(carry_ptr + rid*NBLK + ks, a_out)
    tl.store(carry_ptr + nrows*NBLK + rid*NBLK + ks, x_out)


@triton.jit
def ssm_bwd_apply_kernel(
    loc_a_ptr, loc_dh_ptr, carry_ptr,
    u_ptr, A_ptr, B_ptr, h_ptr, dY_ptr,
    du_ptr, dA_ptr, dB_ptr, dC_ptr,
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    BLK: tl.constexpr, NBLK: tl.constexpr,
):
    rid   = tl.program_id(0)
    blk   = tl.program_id(1)
    bi    = rid // H
    hi    = rid % H
    nrows = B * H

    if blk == NBLK - 1:
        a_p, x_p = 1.0, 0.0
    else:
        c_a = rid * NBLK + (blk + 1)
        a_p = tl.load(carry_ptr + c_a)
        x_p = tl.load(carry_ptr + nrows*NBLK + c_a)

    ts = blk * BLK
    te = tl.where(ts + BLK > L, L, ts + BLK)

    for i in tl.static_range(BLK):
        p = (te - 1) - i
        if p >= ts:
            off    = bi*L*H + p*H + hi
            a_loc  = tl.load(loc_a_ptr  + off)
            dh_loc = tl.load(loc_dh_ptr + off)
            _, dH  = _combine_bwd(a_p, x_p, a_loc, dh_loc)

            h_t    = tl.load(h_ptr  + off)
            u_t    = tl.load(u_ptr  + off)
            B_t    = tl.load(B_ptr  + off)
            dY_t   = tl.load(dY_ptr + off)

            tl.store(dC_ptr + off, dY_t * h_t)
            tl.store(du_ptr + off, B_t  * dH)
            tl.store(dB_ptr + off, u_t  * dH)

            h_prev = tl.load(h_ptr + bi*L*H + (p-1)*H + hi,
                             mask=(p > 0), other=0.0)
            tl.store(dA_ptr + off, h_prev * dH)


def _nxt_pow2(n):
    if n <= 1: return 1
    return 1 << (n-1).bit_length()


def ssm_forward(u, A, B, C, D, BLK=256):
    """
    u: [batch, seqlen, H]
    A, B, C: [batch, seqlen, H]  -- time-varying discretized params
    D: [H]
    returns y [batch, seqlen, H], h [batch, seqlen, H]
    """
    Bsz, L, H = u.shape
    nrows  = Bsz * H
    nblk_a = (L + BLK - 1) // BLK
    nblk   = _nxt_pow2(nblk_a)

    loc_a  = torch.empty_like(A)
    loc_x  = torch.empty_like(A)
    h_out  = torch.empty_like(u)
    y_out  = torch.empty_like(u)
    carry  = torch.empty(2 * nrows * nblk, dtype=u.dtype, device=u.device)

    g1 = (nrows, nblk)
    ssm_fwd_local_kernel[g1](u, A, B, loc_a, loc_x, carry,
                              Bsz, L, H, BLK, nblk)
    ssm_carry_scan_kernel[(nrows,)](carry, nrows, nblk)
    ssm_fwd_apply_kernel[g1](loc_a, loc_x, carry, h_out,
                              Bsz, L, H, BLK, nblk)
    ssm_output_kernel[g1](h_out, C, D, u, y_out, Bsz, L, H, BLK)
    return y_out, h_out


def ssm_backward(u, A, B, C, h, dY, BLK=256):
    Bsz, L, H = u.shape
    nrows  = Bsz * H
    nblk_a = (L + BLK - 1) // BLK
    nblk   = _nxt_pow2(nblk_a)

    loc_a   = torch.empty_like(A)
    loc_dh  = torch.empty_like(A)
    carry   = torch.empty(2 * nrows * nblk, dtype=u.dtype, device=u.device)
    du = torch.empty_like(u)
    dA = torch.empty_like(A)
    dB = torch.empty_like(B)
    dC = torch.empty_like(C)

    g1 = (nrows, nblk)
    ssm_bwd_local_kernel[g1](A, C, dY, loc_a, loc_dh, carry,
                              Bsz, L, H, BLK, nblk)
    ssm_bwd_carry_scan_kernel[(nrows,)](carry, nrows, nblk)
    ssm_bwd_apply_kernel[g1](loc_a, loc_dh, carry,
                              u, A, B, h, dY,
                              du, dA, dB, dC,
                              Bsz, L, H, BLK, nblk)
    return du, dA, dB, dC
