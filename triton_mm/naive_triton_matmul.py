import triton
import triton.language as tl


@triton.jit
def naive_triton_matmul(c_ptr, a_ptr, b_ptr, M, N, K, stride_am, stride_ak,
                        stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
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
    c_ptrs = c_ptr + (offset_am[:, None] * stride_cm +
                      offset_bn[None, :] * stride_cn)
    c_mask = (offset_am[:, None] < M) & (offset_bn[None, :] < N)
    tl.store(c_ptrs, res, mask=c_mask)


def launch_naive_triton_matmul(c_ptr, a_ptr, b_ptr, M, N, K, stride_am,
                               stride_ak, stride_bk, stride_bn, stride_cm,
                               stride_cn, BLOCK_SIZE):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                         triton.cdiv(N, meta["BLOCK_SIZE_N"]))
    naive_triton_matmul[grid](c_ptr,
                              a_ptr,
                              b_ptr,
                              M,
                              N,
                              K,
                              stride_am,
                              stride_ak,
                              stride_bk,
                              stride_bn,
                              stride_cm,
                              stride_cn,
                              BLOCK_SIZE_M=BLOCK_SIZE,
                              BLOCK_SIZE_N=BLOCK_SIZE,
                              BLOCK_SIZE_K=BLOCK_SIZE)
