__global__ void naive_cuda_matmal(
    float *c_ptr,
    const float *a_ptr, const float *b_ptr,
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t stride_am, const int64_t stride_ak,
    const int64_t stride_bk, const int64_t stride_bn,
    const int64_t stride_cm, const int64_t stride_cn,
    const int64_t BLOCK_SIZE)
{
    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (id_x > M or id_y > N)
        return;

    float res = 0;
    for (int k = 0; k < K; k++)
    {
        res += a_ptr[id_x * stride_am + k * stride_ak] * b_ptr[k * stride_bk + id_y * stride_bn];
    }
    c_ptr[id_x * stride_cm + id_y * stride_cn] = res;
}

void launch_naive_cuda_matmul(
    float *c_ptr,
    const float *a_ptr, const float *b_ptr,
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t stride_am, const int64_t stride_ak,
    const int64_t stride_bk, const int64_t stride_bn,
    const int64_t stride_cm, const int64_t stride_cn,
    const int64_t BLOCK_SIZE)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    naive_cuda_matmal<<<grid, block>>>(
        c_ptr, a_ptr, b_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE);
}