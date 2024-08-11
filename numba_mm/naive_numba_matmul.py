from numba import cuda


@cuda.jit("void(float32[:,:], float32[:,:], float32[:,:],"
          "int64, int64, int64)")
def naive_numba_matmul(out, a, b, M, N, K):
    id_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    id_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if id_x > N or id_y > M:
        return

    res = 0
    for k in range(K):
        res += a[id_y, k] * b[k, id_x]
    out[id_y, id_x] = res


def launch_naive_numba_matmul(out, a, b, M, N, K, BLOCK_SIZE):
    block = (BLOCK_SIZE, BLOCK_SIZE)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) // BLOCK_SIZE)
    naive_numba_matmul[grid, block](out, a, b, M, N, K, BLOCK_SIZE)
