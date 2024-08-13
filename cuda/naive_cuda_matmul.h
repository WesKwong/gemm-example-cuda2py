void launch_naive_cuda_matmul(
    float *c_ptr,
    const float *a_ptr, const float *b_ptr,
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t stride_am, const int64_t stride_ak,
    const int64_t stride_bk, const int64_t stride_bn,
    const int64_t stride_cm, const int64_t stride_cn,
    const int64_t BLOCK_SIZE);