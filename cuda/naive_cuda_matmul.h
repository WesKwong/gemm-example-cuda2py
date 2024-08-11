void launch_naive_cuda_matmul(
    float *out_ptr,
    const float *a_ptr, const float *b_ptr,
    int M, int N, int K,
    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,
    int BLOCK_SIZE);