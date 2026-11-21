#include "nn/layers/global_avg_pool2d.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

namespace nn {

struct GlobalAvgPool2d::Impl {};

GlobalAvgPool2d::GlobalAvgPool2d(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

GlobalAvgPool2d::~GlobalAvgPool2d() = default;

std::vector<NamedParam> GlobalAvgPool2d::parameters() const { return {}; }

void GlobalAvgPool2d::zero_grad() {}

const Tensor& GlobalAvgPool2d::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("GlobalAvgPool2d input rank");
  const int n = input.shape()[0];
  const int c = input.shape()[1];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  const int spatial = h * w;
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, {n, c, 1, 1});
  ensure_tensor(cache_y_, *ctx_, output_->shape());
  cache_x_->copy_from(input);
  const float* x = cache_x_->data();
  float* y = output_->data();
  const int planes = n * c;
  for_groups(queue_of(*ctx_), "global_avg_pool_forward", planes,
            [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
    const int plane = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const float* src = x + static_cast<int64_t>(plane) * spatial;
    float sum = 0.f;
    if ((spatial % 4) == 0) {
      for (int s = tid; s < spatial / 4; s += kThreads) {
        const f4 t = load4(src + 4 * s);
        sum += t[0] + t[1] + t[2] + t[3];
      }
    } else {
      for (int s = tid; s < spatial; s += kThreads) sum += src[s];
    }
    sum = block_sum(item, sum, scratch);
    if (tid == 0) y[plane] = sum / static_cast<float>(spatial);
  });
  cache_y_->copy_from(*output_);
  return *output_;
}

const Tensor& GlobalAvgPool2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !cache_y_ || grad_output.shape() != cache_y_->shape()) {
    throw Error("GlobalAvgPool2d backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  const int n = cache_x_->shape()[0];
  const int c = cache_x_->shape()[1];
  const int spatial = cache_x_->shape()[2] * cache_x_->shape()[3];
  const float* dy = grad_output.data();
  float* dx = grad_input_->data();
  for_groups(queue_of(*ctx_), "global_avg_pool_backward", n * c,
            [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1>) {
    const int plane = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const float scale = dy[plane] / static_cast<float>(spatial);
    float* dst = dx + static_cast<int64_t>(plane) * spatial;
    if ((spatial % 4) == 0) {
      const f4 filled = make4(scale, scale, scale, scale);
      for (int s = tid; s < spatial / 4; s += kThreads) store4(dst + 4 * s, filled);
    } else {
      for (int s = tid; s < spatial; s += kThreads) dst[s] = scale;
    }
  });
  return *grad_input_;
}

}  // namespace nn
