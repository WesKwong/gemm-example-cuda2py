__global__ void smem_cuda_matmal(
    float *c_ptr,
    const float *a_ptr, const float *b_ptr,
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t stride_am, const int64_t stride_ak,
    const int64_t stride_bk, const int64_t stride_bn,
    const int64_t stride_cm, const int64_t stride_cn,
    const int64_t BLOCK_SIZE)
{
    const int id_m = blockIdx.y * blockDim.y + threadIdx.y;
    const int id_n = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid_m = threadIdx.y;
    const int bid_n = threadIdx.x;

    extern __shared__ float ab_shared[];
    float* a_shared = (float *) &ab_shared[0];
    float* b_shared = (float *) &ab_shared[BLOCK_SIZE * BLOCK_SIZE];
    const int shared_ptr = bid_m * BLOCK_SIZE + bid_n;

    float res = 0.0;
    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_SIZE)
    {
        a_shared[shared_ptr] = 0.0;
        b_shared[shared_ptr] = 0.0;
        // load a
        if (id_m < M and bid_n + offset_k < K)
            a_shared[shared_ptr] = a_ptr[id_m * stride_am + (bid_n + offset_k) * stride_ak];
        // load b
        if (id_n < N and bid_m + offset_k < K)
            b_shared[shared_ptr] = b_ptr[(bid_m + offset_k) * stride_bk + id_n * stride_bn];
        __syncthreads();
        // vector dotmul
        for (int k = 0; k < BLOCK_SIZE; k++)
            res += a_shared[bid_m * BLOCK_SIZE + k] * b_shared[k * BLOCK_SIZE + bid_n];
        __syncthreads();
    }
    if (id_m < M and id_n < N)
        c_ptr[id_m * stride_cm + id_n * stride_cn] = res;
}

void launch_smem_cuda_matmul(
    float *c_ptr,
    const float *a_ptr, const float *b_ptr,
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t stride_am, const int64_t stride_ak,
    const int64_t stride_bk, const int64_t stride_bn,
    const int64_t stride_cm, const int64_t stride_cn,
    const int64_t BLOCK_SIZE)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    smem_cuda_matmal<<<grid, block, 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float)>>>(
        c_ptr, a_ptr, b_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE);
}