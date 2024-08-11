import triton
import triton.language as tl


@triton.jit
def naive_triton_matmul(out_ptr, a_ptr, b_ptr, M, N, K, stride_am, stride_ak,
                        stride_bk, stride_bn, stride_outm, stride_outn,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr):
    id_x = tl.program_id(0)
    id_y = tl.program_id(1)
    offset_am = (id_y * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_bn = (id_x * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offset_am[:, None] * stride_am +
                      offset_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk +
                      offset_bn[None, :] * stride_bn)
    res = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=offset_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offset_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        res += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offset_outm = id_y * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_outn = id_x * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (offset_outm[:, None] * stride_outm +
                          offset_outn[None, :] * stride_outn)
    out_mask = (offset_outm[:, None] < M) and (offset_outn[None, :] < N)
    tl.store(out_ptrs, res, mask=out_mask)


def launch_naive_triton_matmul(out_ptr, a_ptr, b_ptr, M, N, K, stride_am,
                               stride_ak, stride_bk, stride_bn, stride_outm,
                               stride_outn, BLOCK_SIZE):
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),
                         triton.cdiv(M, meta["BLOCK_SIZE_M"]))
    naive_triton_matmul[grid](out_ptr,
                              a_ptr,
                              b_ptr,
                              M,
                              N,
                              K,
                              stride_am,
                              stride_ak,
                              stride_bk,
                              stride_bn,
                              stride_outm,
                              stride_outn,
                              BLOCK_SIZE_M=BLOCK_SIZE,
                              BLOCK_SIZE_N=BLOCK_SIZE,
                              BLOCK_SIZE_K=BLOCK_SIZE)
