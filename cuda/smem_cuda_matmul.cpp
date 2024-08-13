#include <torch/extension.h>
#include "smem_cuda_matmul.h"

void torch_launch_smem_cuda_matmul(
    torch::Tensor &c,
    const torch::Tensor &a,
    const torch::Tensor &b,
    const int64_t M, const int64_t N, const int64_t K,
    const int64_t stride_am, const int64_t stride_ak,
    const int64_t stride_bk, const int64_t stride_bn,
    const int64_t stride_cm, const int64_t stride_cn,
    const int64_t BLOCK_SIZE)
{
    launch_smem_cuda_matmul(
        (float *)c.data_ptr(),
        (const float *)a.data_ptr(),
        (const float *)b.data_ptr(),
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("torch_launch_smem_cuda_matmul",
          &torch_launch_smem_cuda_matmul,
          "smem matmul warpper");
}

TORCH_LIBRARY(torch_launch_naive_cuda_matmul, m)
{
    m.def("torch_launch_smem_cuda_matmul",
          &torch_launch_smem_cuda_matmul);
}