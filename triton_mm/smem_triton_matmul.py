import triton
import triton.language as tl


@triton.jit
def smem_triton_matmul(c_ptr, a_ptr, b_ptr, M, N, K, stride_am, stride_ak,
                       stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                       BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    raw_pid_m = tl.program_id(0)
    raw_pid_n = tl.program_id(1)
    num_program_m = tl.num_programs(0)
    num_program_n = tl.num_programs(1)
    group_id = raw_pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_program_m - first_pid_m, GROUP_SIZE_M)
    linear_pid_in_group = (raw_pid_m - first_pid_m) * num_program_n + raw_pid_n
    pid_m = first_pid_m + linear_pid_in_group % group_size_m
    pid_n = linear_pid_in_group // group_size_m
    offset_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offset_am[:, None] * stride_am +
                      offset_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk +
                      offset_bn[None, :] * stride_bn)
    res = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=offset_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offset_k[None, :] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        res += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c_ptrs = c_ptr + (offset_am[:, None] * stride_cm +
                      offset_bn[None, :] * stride_cn)
    c_mask = (offset_am[:, None] < M) & (offset_bn[None, :] < N)
    tl.store(c_ptrs, res, mask=c_mask)


def launch_smem_triton_matmul(out_ptr, a_ptr, b_ptr, M, N, K, stride_am,
                              stride_ak, stride_bk, stride_bn, stride_outm,
                              stride_outn, BLOCK_SIZE, GROUP_SIZE):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),
                         triton.cdiv(N, meta["BLOCK_SIZE_N"]))
    smem_triton_matmul[grid](out_ptr,
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
                             BLOCK_SIZE_K=BLOCK_SIZE,
                             GROUP_SIZE_M=GROUP_SIZE)
