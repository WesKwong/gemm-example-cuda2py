#include <torch/extension.h>
#include "naive_cuda_matmul.h"

void torch_launch_naive_cuda_matmul(
    torch::Tensor &out,
    const torch::Tensor &a,
    const torch::Tensor &b,
    int64_t M, int64_t N, int64_t K,
    int64_t stride_am, int64_t stride_ak,
    int64_t stride_bk, int64_t stride_bn,
    int64_t stride_cm, int64_t stride_cn,
    int64_t BLOCK_SIZE)
{
    launch_naive_cuda_matmul(
        (float *)out.data_ptr(),
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
    m.def("torch_launch_naive_cuda_matmul",
          &torch_launch_naive_cuda_matmul,
          "naive matmul warpper");
}

TORCH_LIBRARY(torch_launch_naive_cuda_matmul, m)
{
    m.def("torch_launch_naive_cuda_matmul",
          &torch_launch_naive_cuda_matmul);
}
