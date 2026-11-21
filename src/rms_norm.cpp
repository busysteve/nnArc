#include "nn/layers/rms_norm.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

#include <cmath>
#include <vector>

namespace nn {
namespace {

int normalized_rows(const Tensor& input, int normalized_shape) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() < 2 || shape.back() != normalized_shape) {
    throw Error("RMSNorm input shape");
  }
  return static_cast<int>(input.numel() / normalized_shape);
}

Context& require_rms_norm(Context& ctx, int normalized_shape, float eps) {
  if (normalized_shape < 1) throw Error("RMSNorm size");
  if (!(eps > 0.f) || !std::isfinite(eps)) throw Error("RMSNorm eps");
  return ctx;
}

}  // namespace

RMSNorm::RMSNorm(Context& ctx, int normalized_shape, float eps)
    : ctx_(&ctx),
      normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(require_rms_norm(ctx, normalized_shape, eps), {normalized_shape}),
      grad_weight_(ctx, {normalized_shape}) {
  weight_.fill(1.f);
  grad_weight_.zero();
}

std::vector<NamedParam> RMSNorm::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)}};
}

void RMSNorm::zero_grad() { grad_weight_.zero(); }

const Tensor& RMSNorm::forward(const Tensor& input) {
  const int rows = normalized_rows(input, normalized_shape_);
  const std::vector<int>& shape = input.shape();
  ensure_tensor(cache_x_, *ctx_, shape);
  ensure_tensor(output_, *ctx_, shape);
  ensure_tensor(inv_rms_, *ctx_, std::vector<int>{rows});
  cache_x_->copy_from(input);
  const int cols = normalized_shape_;
  const float eps = eps_;
  const float* x = cache_x_->data();
  const float* weight = weight_.data();
  float* y = output_->data();
  float* inv_rms_out = inv_rms_->data();
  for_groups(queue_of(*ctx_), "rms_norm_forward", rows,
             [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
               const int row = static_cast<int>(item.get_group(0));
               const int tid = static_cast<int>(item.get_local_id(0));
               const int64_t base = static_cast<int64_t>(row) * cols;
               const float* row_x = x + base;
               float* row_y = y + base;
               float sq = 0.f;
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(row_x + 4 * c);
                   sq += t[0] * t[0] + t[1] * t[1] + t[2] * t[2] + t[3] * t[3];
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   const float value = row_x[col];
                   sq += value * value;
                 }
               }
               sq = block_sum(item, sq, scratch);
               const float inv_rms = sycl::rsqrt(sq / static_cast<float>(cols) + eps);
               if (tid == 0) inv_rms_out[row] = inv_rms;
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(row_x + 4 * c);
                   const f4 w = load4(weight + 4 * c);
                   store4(row_y + 4 * c, make4(t[0] * inv_rms * w[0], t[1] * inv_rms * w[1],
                                               t[2] * inv_rms * w[2], t[3] * inv_rms * w[3]));
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   row_y[col] = row_x[col] * inv_rms * weight[col];
                 }
               }
             });
  return *output_;
}

const Tensor& RMSNorm::backward(const Tensor& grad_output) {
  if (!cache_x_ || !inv_rms_ || grad_output.shape() != cache_x_->shape()) {
    throw Error("RMSNorm backward shape");
  }
  const int rows = static_cast<int>(inv_rms_->numel());
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  const int cols = normalized_shape_;
  float* dx = grad_input_->data();
  const float* dy = grad_output.data();
  const float* x = cache_x_->data();
  const float* inv_rms = inv_rms_->data();
  const float* weight = weight_.data();
  float* grad_weight = grad_weight_.data();
  sycl::queue& q = queue_of(*ctx_);
  for_groups(q, "rms_norm_backward_input", rows,
             [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
               const int row = static_cast<int>(item.get_group(0));
               const int tid = static_cast<int>(item.get_local_id(0));
               const int64_t base = static_cast<int64_t>(row) * cols;
               const float inv = inv_rms[row];
               float sum_dhat_xhat = 0.f;
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(x + base + 4 * c);
                   const f4 d = load4(dy + base + 4 * c);
                   const f4 w = load4(weight + 4 * c);
                   const float x0 = t[0] * inv;
                   const float x1 = t[1] * inv;
                   const float x2 = t[2] * inv;
                   const float x3 = t[3] * inv;
                   sum_dhat_xhat += d[0] * w[0] * x0 + d[1] * w[1] * x1 + d[2] * w[2] * x2 + d[3] * w[3] * x3;
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   const float xhat = x[base + col] * inv;
                   const float dhat = dy[base + col] * weight[col];
                   sum_dhat_xhat += dhat * xhat;
                 }
               }
               const float mean_dhat_xhat = block_sum(item, sum_dhat_xhat, scratch) / static_cast<float>(cols);
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(x + base + 4 * c);
                   const f4 d = load4(dy + base + 4 * c);
                   const f4 w = load4(weight + 4 * c);
                   store4(dx + base + 4 * c,
                          make4(inv * (d[0] * w[0] - t[0] * inv * mean_dhat_xhat),
                                inv * (d[1] * w[1] - t[1] * inv * mean_dhat_xhat),
                                inv * (d[2] * w[2] - t[2] * inv * mean_dhat_xhat),
                                inv * (d[3] * w[3] - t[3] * inv * mean_dhat_xhat)));
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   const float xhat = x[base + col] * inv;
                   const float dhat = dy[base + col] * weight[col];
                   dx[base + col] = inv * (dhat - xhat * mean_dhat_xhat);
                 }
               }
             });
  for_each(q, "rms_norm_backward_param", cols, [=](int64_t col) {
    float sum = 0.f;
    for (int row = 0; row < rows; ++row) {
      const int64_t index = static_cast<int64_t>(row) * cols + col;
      const float xhat = x[index] * inv_rms[row];
      sum += dy[index] * xhat;
    }
    grad_weight[col] += sum;
  });
  return *grad_input_;
}

}  // namespace nn
