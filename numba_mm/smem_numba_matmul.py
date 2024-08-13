import numba as nb
from numba import cuda

BLOCK_SIZE = 32


@cuda.jit("void(float32[:,:], float32[:,:], float32[:,:],"
          "int64, int64, int64)")
def smem_numba_matmul(c, a, b, M, N, K):
    id_m = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    id_n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    bid_m = cuda.threadIdx.y
    bid_n = cuda.threadIdx.x

    a_shared = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=nb.float32)
    b_shared = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=nb.float32)

    res = 0.0
    for offset_k in range(0, K, BLOCK_SIZE):
        a_shared[bid_m, bid_n] = 0.0
        b_shared[bid_m, bid_n] = 0.0
        # load a
        if id_m < M and bid_n + offset_k < K:
            a_shared[bid_m, bid_n] = a[id_m, bid_n + offset_k]
        # load b
        if id_n < N and bid_m + offset_k < K:
            b_shared[bid_m, bid_n] = b[bid_m + offset_k, id_n]
        cuda.syncthreads()
        # vector dormul
        for k in range(BLOCK_SIZE):
            res += a_shared[bid_m, k] * b_shared[k, bid_n]
        cuda.syncthreads()
    if id_m < M and id_n < N:
        c[id_m, id_n] = res


def launch_smem_numba_matmul(c, a, b, M, N, K):
    block = (BLOCK_SIZE, BLOCK_SIZE)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) // BLOCK_SIZE)
    smem_numba_matmul[grid, block](c, a, b, M, N, K)
