#include "nn/layers/batch_norm2d.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

namespace nn {

struct BatchNorm2d::Impl {};

BatchNorm2d::BatchNorm2d(Context& ctx, int channels)
    : ctx_(&ctx),
      channels_(channels),
      weight_(ctx, {channels}),
      grad_weight_(ctx, {channels}),
      bias_(ctx, {channels}),
      grad_bias_(ctx, {channels}),
      running_mean_(ctx, {channels}),
      running_var_(ctx, {channels}),
      saved_mean_(ctx, {channels}),
      saved_inv_variance_(ctx, {channels}),
      impl_(std::make_unique<Impl>()) {
  if (channels < 1) throw Error("BatchNorm2d channels");
  weight_.fill(1.f);
  bias_.zero();
  grad_weight_.zero();
  grad_bias_.zero();
  running_mean_.zero();
  running_var_.fill(1.f);
  saved_mean_.zero();
  saved_inv_variance_.zero();
}

BatchNorm2d::~BatchNorm2d() = default;

std::vector<NamedParam> BatchNorm2d::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)},
          NamedParam{"bias", const_cast<Tensor*>(&bias_), const_cast<Tensor*>(&grad_bias_)}};
}

std::vector<NamedBuffer> BatchNorm2d::buffers() const {
  auto* self = const_cast<BatchNorm2d*>(this);
  return {NamedBuffer{"running_mean", &self->running_mean_, &self->batches_},
          NamedBuffer{"running_var", &self->running_var_, &self->batches_}};
}

void BatchNorm2d::zero_grad() {
  grad_weight_.zero();
  grad_bias_.zero();
}

void BatchNorm2d::set_training(bool training) { training_ = training; }

void BatchNorm2d::bind(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != channels_) throw Error("BatchNorm2d input shape");
  ensure_tensor(output_, *ctx_, input.shape());
  ensure_tensor(grad_input_, *ctx_, input.shape());
}

const Tensor& BatchNorm2d::forward(const Tensor& input) {
  bind(input);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  const int n = input.shape()[0];
  const int spatial = input.shape()[2] * input.shape()[3];
  const int channels = channels_;
  const float* x = cache_x_->data();
  float* y = output_->data();
  const float* gamma = weight_.data();
  const float* beta = bias_.data();
  if (training_) {
    const int64_t count = static_cast<int64_t>(n) * spatial;
    if (count <= 1) throw Error("BatchNorm train mode needs more than one value per channel");
    ++batches_;
    const float factor = 1.f / static_cast<float>(batches_);
    float* running_mean = running_mean_.data();
    float* running_var = running_var_.data();
    float* saved_mean = saved_mean_.data();
    float* saved_inv = saved_inv_variance_.data();
    for_groups(queue_of(*ctx_), "batch_norm_forward", channels,
               [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
                 const int ch = static_cast<int>(item.get_group(0));
                 const int tid = static_cast<int>(item.get_local_id(0));
                 const int m = n * spatial;
                 float sum = 0.f;
                 for (int i = 0; i < n; ++i) {
                   const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
                   if ((spatial % 4) == 0) {
                     for (int s = tid; s < spatial / 4; s += kThreads) {
                       const f4 t = load4(plane + 4 * s);
                       sum += t[0] + t[1] + t[2] + t[3];
                     }
                   } else {
                     for (int s = tid; s < spatial; s += kThreads) sum += plane[s];
                   }
                 }
                 const float mean = block_sum(item, sum, scratch) / static_cast<float>(m);
                 float var_sum = 0.f;
                 for (int i = 0; i < n; ++i) {
                   const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
                   if ((spatial % 4) == 0) {
                     for (int s = tid; s < spatial / 4; s += kThreads) {
                       const f4 t = load4(plane + 4 * s);
                       const float d0 = t[0] - mean;
                       const float d1 = t[1] - mean;
                       const float d2 = t[2] - mean;
                       const float d3 = t[3] - mean;
                       var_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                     }
                   } else {
                     for (int s = tid; s < spatial; s += kThreads) {
                       const float d = plane[s] - mean;
                       var_sum += d * d;
                     }
                   }
                 }
                 var_sum = block_sum(item, var_sum, scratch);
                 const float var_b = var_sum / static_cast<float>(m);
                 const float inv = sycl::rsqrt(var_b + 1e-5f);
                 if (tid == 0) {
                   saved_mean[ch] = mean;
                   saved_inv[ch] = inv;
                   const float var_u = var_b * static_cast<float>(m) / static_cast<float>(m - 1);
                   running_mean[ch] = (1.f - factor) * running_mean[ch] + factor * mean;
                   running_var[ch] = (1.f - factor) * running_var[ch] + factor * var_u;
                 }
                 const float scale = gamma[ch];
                 const float shift = beta[ch];
                 for (int i = 0; i < n; ++i) {
                   const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
                   float* out = y + (static_cast<int64_t>(i) * channels + ch) * spatial;
                   for (int s = tid; s < spatial; s += kThreads) {
                     out[s] = scale * (plane[s] - mean) * inv + shift;
                   }
                 }
               });
  } else {
    const float* running_mean = running_mean_.data();
    const float* running_var = running_var_.data();
    for_groups(queue_of(*ctx_), "batch_norm_forward", channels,
               [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1>) {
                 const int ch = static_cast<int>(item.get_group(0));
                 const int tid = static_cast<int>(item.get_local_id(0));
                 const float mean = running_mean[ch];
                 const float inv = sycl::rsqrt(running_var[ch] + 1e-5f);
                 const float scale = gamma[ch];
                 const float shift = beta[ch];
                 for (int i = 0; i < n; ++i) {
                   const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
                   float* out = y + (static_cast<int64_t>(i) * channels + ch) * spatial;
                   for (int s = tid; s < spatial; s += kThreads) {
                     out[s] = scale * (plane[s] - mean) * inv + shift;
                   }
                 }
               });
  }
  return *output_;
}

const Tensor& BatchNorm2d::backward(const Tensor& grad_output) {
  if (!training_) throw Error("BatchNorm backward requires train mode");
  if (!cache_x_ || grad_output.shape() != cache_x_->shape()) throw Error("BatchNorm2d backward shape");
  const int n = cache_x_->shape()[0];
  const int spatial = cache_x_->shape()[2] * cache_x_->shape()[3];
  const int channels = channels_;
  float* dx = grad_input_->data();
  const float* x = cache_x_->data();
  const float* dy = grad_output.data();
  const float* gamma = weight_.data();
  float* grad_gamma = grad_weight_.data();
  float* grad_beta = grad_bias_.data();
  const float* saved_mean = saved_mean_.data();
  const float* saved_inv = saved_inv_variance_.data();
  for_groups(queue_of(*ctx_), "batch_norm_backward", channels,
             [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
               const int ch = static_cast<int>(item.get_group(0));
               const int tid = static_cast<int>(item.get_local_id(0));
               const int m = n * spatial;
               const float mean = saved_mean[ch];
               const float inv = saved_inv[ch];
               float sum_dy = 0.f;
               float sum_dy_xhat = 0.f;
               for (int i = 0; i < n; ++i) {
                 const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
                 const float* gplane = dy + (static_cast<int64_t>(i) * channels + ch) * spatial;
                 for (int s = tid; s < spatial; s += kThreads) {
                   const float xhat = (plane[s] - mean) * inv;
                   const float g = gplane[s];
                   sum_dy += g;
                   sum_dy_xhat += g * xhat;
                 }
               }
               sum_dy = block_sum(item, sum_dy, scratch);
               sum_dy_xhat = block_sum(item, sum_dy_xhat, scratch);
               const float mean_dy = sum_dy / static_cast<float>(m);
               const float mean_dy_xhat = sum_dy_xhat / static_cast<float>(m);
               if (tid == 0) {
                 grad_gamma[ch] += sum_dy_xhat;
                 grad_beta[ch] += sum_dy;
               }
               const float scale = gamma[ch] * inv;
               for (int i = 0; i < n; ++i) {
                 const float* plane = x + (static_cast<int64_t>(i) * channels + ch) * spatial;
                 const float* gplane = dy + (static_cast<int64_t>(i) * channels + ch) * spatial;
                 float* out = dx + (static_cast<int64_t>(i) * channels + ch) * spatial;
                 for (int s = tid; s < spatial; s += kThreads) {
                   const float xhat = (plane[s] - mean) * inv;
                   out[s] = scale * (gplane[s] - mean_dy - xhat * mean_dy_xhat);
                 }
               }
             });
  return *grad_input_;
}

}  // namespace nn
