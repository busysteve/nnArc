#include "nn/layers/layer_norm.hpp"

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
  if ((shape.size() != 2 && shape.size() != 3) || shape.back() != normalized_shape) {
    throw Error("LayerNorm input shape");
  }
  return static_cast<int>(input.numel() / normalized_shape);
}

Context& require_layer_norm(Context& ctx, int normalized_shape, float eps) {
  if (normalized_shape < 1) throw Error("LayerNorm size");
  if (!(eps > 0.f) || !std::isfinite(eps)) throw Error("LayerNorm eps");
  return ctx;
}

}  // namespace

LayerNorm::LayerNorm(Context& ctx, int normalized_shape, float eps)
    : ctx_(&ctx),
      normalized_shape_(normalized_shape),
      eps_(eps),
      weight_(require_layer_norm(ctx, normalized_shape, eps), {normalized_shape}),
      grad_weight_(ctx, {normalized_shape}),
      bias_(ctx, {normalized_shape}),
      grad_bias_(ctx, {normalized_shape}) {
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
}

std::vector<NamedParam> LayerNorm::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

void LayerNorm::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

const Tensor& LayerNorm::forward(const Tensor& input) {
  const int rows = normalized_rows(input, normalized_shape_);
  const std::vector<int>& shape = input.shape();
  ensure_tensor(cache_x_, *ctx_, shape);
  ensure_tensor(output_, *ctx_, shape);
  ensure_tensor(mean_, *ctx_, std::vector<int>{rows});
  ensure_tensor(inv_std_, *ctx_, std::vector<int>{rows});
  cache_x_->copy_from(input);
  const int cols = normalized_shape_;
  const float eps = eps_;
  const float* x = cache_x_->data();
  const float* gamma = weight_.data();
  const float* beta = bias_.data();
  float* y = output_->data();
  float* mean_out = mean_->data();
  float* inv_std_out = inv_std_->data();
  for_groups(queue_of(*ctx_), "layer_norm_forward", rows,
             [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
               const int row = static_cast<int>(item.get_group(0));
               const int tid = static_cast<int>(item.get_local_id(0));
               const int64_t base = static_cast<int64_t>(row) * cols;
               const float* row_x = x + base;
               float* row_y = y + base;
               float sum = 0.f;
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(row_x + 4 * c);
                   sum += t[0] + t[1] + t[2] + t[3];
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) sum += row_x[col];
               }
               const float mean = block_sum(item, sum, scratch) / static_cast<float>(cols);
               float var_sum = 0.f;
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(row_x + 4 * c);
                   const float d0 = t[0] - mean;
                   const float d1 = t[1] - mean;
                   const float d2 = t[2] - mean;
                   const float d3 = t[3] - mean;
                   var_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   const float diff = row_x[col] - mean;
                   var_sum += diff * diff;
                 }
               }
               var_sum = block_sum(item, var_sum, scratch);
               const float inv_std = sycl::rsqrt(var_sum / static_cast<float>(cols) + eps);
               if (tid == 0) {
                 mean_out[row] = mean;
                 inv_std_out[row] = inv_std;
               }
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(row_x + 4 * c);
                   const f4 g = load4(gamma + 4 * c);
                   const f4 b = load4(beta + 4 * c);
                   store4(row_y + 4 * c, make4((t[0] - mean) * inv_std * g[0] + b[0],
                                               (t[1] - mean) * inv_std * g[1] + b[1],
                                               (t[2] - mean) * inv_std * g[2] + b[2],
                                               (t[3] - mean) * inv_std * g[3] + b[3]));
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   const float xhat = (row_x[col] - mean) * inv_std;
                   row_y[col] = xhat * gamma[col] + beta[col];
                 }
               }
             });
  return *output_;
}

const Tensor& LayerNorm::backward(const Tensor& grad_output) {
  if (!cache_x_ || !mean_ || !inv_std_ || grad_output.shape() != cache_x_->shape()) {
    throw Error("LayerNorm backward shape");
  }
  const int rows = static_cast<int>(mean_->numel());
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  const int cols = normalized_shape_;
  float* dx = grad_input_->data();
  const float* dy = grad_output.data();
  const float* x = cache_x_->data();
  const float* mean = mean_->data();
  const float* inv_std = inv_std_->data();
  const float* gamma = weight_.data();
  float* grad_gamma = grad_weight_.data();
  float* grad_beta = grad_bias_.data();
  sycl::queue& q = queue_of(*ctx_);
  for_groups(q, "layer_norm_backward_input", rows,
             [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
               const int row = static_cast<int>(item.get_group(0));
               const int tid = static_cast<int>(item.get_local_id(0));
               const int64_t base = static_cast<int64_t>(row) * cols;
               const float mean_v = mean[row];
               const float inv = inv_std[row];
               float sum_dhat = 0.f;
               float sum_dhat_xhat = 0.f;
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(x + base + 4 * c);
                   const f4 g = load4(gamma + 4 * c);
                   const f4 d = load4(dy + base + 4 * c);
                   const float x0 = (t[0] - mean_v) * inv;
                   const float x1 = (t[1] - mean_v) * inv;
                   const float x2 = (t[2] - mean_v) * inv;
                   const float x3 = (t[3] - mean_v) * inv;
                   const float d0 = d[0] * g[0];
                   const float d1 = d[1] * g[1];
                   const float d2 = d[2] * g[2];
                   const float d3 = d[3] * g[3];
                   sum_dhat += d0 + d1 + d2 + d3;
                   sum_dhat_xhat += d0 * x0 + d1 * x1 + d2 * x2 + d3 * x3;
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   const float xhat = (x[base + col] - mean_v) * inv;
                   const float dhat = dy[base + col] * gamma[col];
                   sum_dhat += dhat;
                   sum_dhat_xhat += dhat * xhat;
                 }
               }
               sum_dhat = block_sum(item, sum_dhat, scratch);
               sum_dhat_xhat = block_sum(item, sum_dhat_xhat, scratch);
               const float mean_dhat = sum_dhat / static_cast<float>(cols);
               const float mean_dhat_xhat = sum_dhat_xhat / static_cast<float>(cols);
               if ((cols % 4) == 0) {
                 const int n4 = cols / 4;
                 for (int c = tid; c < n4; c += kThreads) {
                   const f4 t = load4(x + base + 4 * c);
                   const f4 g = load4(gamma + 4 * c);
                   const f4 d = load4(dy + base + 4 * c);
                   const float x0 = (t[0] - mean_v) * inv;
                   const float x1 = (t[1] - mean_v) * inv;
                   const float x2 = (t[2] - mean_v) * inv;
                   const float x3 = (t[3] - mean_v) * inv;
                   store4(dx + base + 4 * c,
                          make4(inv * (d[0] * g[0] - mean_dhat - x0 * mean_dhat_xhat),
                                inv * (d[1] * g[1] - mean_dhat - x1 * mean_dhat_xhat),
                                inv * (d[2] * g[2] - mean_dhat - x2 * mean_dhat_xhat),
                                inv * (d[3] * g[3] - mean_dhat - x3 * mean_dhat_xhat)));
                 }
               } else {
                 for (int col = tid; col < cols; col += kThreads) {
                   const float xhat = (x[base + col] - mean_v) * inv;
                   const float dhat = dy[base + col] * gamma[col];
                   dx[base + col] = inv * (dhat - mean_dhat - xhat * mean_dhat_xhat);
                 }
               }
             });
  for_each(q, "layer_norm_backward_param", cols, [=](int64_t col) {
    float sum_gamma = 0.f;
    float sum_beta = 0.f;
    for (int row = 0; row < rows; ++row) {
      const int64_t index = static_cast<int64_t>(row) * cols + col;
      const float xhat = (x[index] - mean[row]) * inv_std[row];
      const float dy_v = dy[index];
      sum_gamma += dy_v * xhat;
      sum_beta += dy_v;
    }
    grad_gamma[col] += sum_gamma;
    grad_beta[col] += sum_beta;
  });
  return *grad_input_;
}

}  // namespace nn
