__global__ void naive_cuda_matmal(
    float *out_ptr,
    const float *a_ptr, const float *b_ptr,
    int M, int N, int K,
    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn)
{
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (id_x > N or id_y > M)
        return;

    float res = 0;
    for (int i = 0; i < K; i++)
    {
        res += a_ptr[id_y * stride_am + i * stride_ak] * b_ptr[i * stride_bk + id_x * stride_bn];
    }
    out_ptr[id_y * stride_cm + id_x * stride_cn] = res;
}

void launch_naive_cuda_matmul(
    float *out_ptr,
    const float *a_ptr, const float *b_ptr,
    int M, int N, int K,
    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,
    int BLOCK_SIZE)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_cuda_matmal<<<grid, block>>>(out_ptr, a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn);
}