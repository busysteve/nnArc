#pragma once

#include "nn/context.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace nn {

void launch_add_bias(sycl::queue& q, float* y, const float* bias, int rows, int cols);
void launch_bias_grad(sycl::queue& q, const float* grad_y, float* grad_bias, int rows, int cols);
void launch_add2(sycl::queue& q, float* dst, const float* a, const float* b, int64_t n);
void launch_mul(sycl::queue& q, float* y, const float* a, const float* b, int64_t n);
void launch_gelu(sycl::queue& q, float* y, const float* x, int64_t n);
void launch_gelu_backward(sycl::queue& q, float* grad_x, const float* x, const float* grad_y, int64_t n);
void launch_swiglu(sycl::queue& q, float* y, const float* x, int rows, int64_t inner);
void launch_swiglu_backward(sycl::queue& q, float* grad_x, const float* x, const float* grad_y, int rows,
                            int64_t inner);
void launch_tanh(sycl::queue& q, float* y, const float* x, int64_t n);
void launch_tanh_backward(sycl::queue& q, float* grad_x, const float* y, const float* grad_y, int64_t n);
void launch_gather_time(sycl::queue& q, float* dst, const float* src, int batch, int time, int features,
                        int index);
void launch_scatter_time(sycl::queue& q, float* dst, const float* src, int batch, int time, int features,
                         int index);
void launch_sgd(sycl::queue& q, float* param, float* velocity, const float* grad, int64_t n,
                float learning_rate, float momentum);
void launch_adamw(sycl::queue& q, float* param, float* m, float* v, const float* grad, int64_t n, float lr,
                  float beta1, float beta2, float eps, float weight_decay, float bc1, float bc2);
double launch_sum_squares(Context& ctx, const float* x, int64_t n);
float launch_sum(Context& ctx, const float* x, int64_t n);
void launch_scale(sycl::queue& q, float* x, int64_t n, float scale);
void launch_fill(sycl::queue& q, float* x, int64_t n, float value);

}  // namespace nn
