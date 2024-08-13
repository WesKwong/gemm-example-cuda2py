from numba import cuda


@cuda.jit("void(float32[:,:], float32[:,:], float32[:,:],"
          "int64, int64, int64)")
def naive_numba_matmul(c, a, b, M, N, K):
    id_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    id_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if id_x > M or id_y > N:
        return

    res = 0
    for k in range(K):
        res += a[id_x, k] * b[k, id_y]
    c[id_x, id_y] = res


def launch_naive_numba_matmul(c, a, b, M, N, K, BLOCK_SIZE):
    block = (BLOCK_SIZE, BLOCK_SIZE)
    grid = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    naive_numba_matmul[grid, block](c, a, b, M, N, K)
