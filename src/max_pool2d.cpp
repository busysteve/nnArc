#include "nn/layers/max_pool2d.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "nn/config.hpp"
#include "nn/error.hpp"

namespace nn {

struct MaxPool2d::Impl {
  int* index = nullptr;
  int64_t capacity = 0;
};

MaxPool2d::MaxPool2d(Context& ctx, int kernel, int stride)
    : ctx_(&ctx), kernel_(kernel), stride_(stride), impl_(std::make_unique<Impl>()) {
  conv_out_size(kernel, kernel, stride, 0);
}

MaxPool2d::~MaxPool2d() {
  if (impl_ && impl_->index && ctx_) {
    try {
      queue_of(*ctx_).wait();
    } catch (...) {
    }
    device_free(queue_of(*ctx_), impl_->index);
  }
}

std::vector<NamedParam> MaxPool2d::parameters() const { return {}; }

void MaxPool2d::zero_grad() {}

const Tensor& MaxPool2d::forward(const Tensor& input) {
  if (input.rank() != 4) throw Error("MaxPool2d input rank");
  const int n = input.shape()[0];
  const int c = input.shape()[1];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  const int ho = conv_out_size(h, kernel_, stride_, 0);
  const int wo = conv_out_size(w, kernel_, stride_, 0);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, {n, c, ho, wo});
  ensure_tensor(cache_y_, *ctx_, output_->shape());
  cache_x_->copy_from(input);
  const int64_t outs = output_->numel();
  if (impl_->capacity < outs) {
    if (impl_->index) {
      queue_of(*ctx_).wait();
      device_free(queue_of(*ctx_), impl_->index);
      impl_->index = nullptr;
      impl_->capacity = 0;
    }
    impl_->index = device_alloc<int>(queue_of(*ctx_), static_cast<size_t>(outs), "sycl::malloc_device");
    impl_->capacity = outs;
  }
  float* y = output_->data();
  int* index = impl_->index;
  const float* x = cache_x_->data();
  const int k = kernel_;
  const int stride = stride_;
  for_each(queue_of(*ctx_), "max_pool_forward", outs, [=](int64_t idx) {
    const int ow = static_cast<int>(idx % wo);
    const int oh = static_cast<int>((idx / wo) % ho);
    const int ch = static_cast<int>((idx / (static_cast<int64_t>(wo) * ho)) % c);
    const int ni = static_cast<int>(idx / (static_cast<int64_t>(wo) * ho * c));
    const int ih0 = oh * stride;
    const int iw0 = ow * stride;
    float best = x[(((static_cast<int64_t>(ni) * c + ch) * h + ih0) * w) + iw0];
    int best_s = ih0 * w + iw0;
    for (int ky = 0; ky < k; ++ky) {
      const int ih = ih0 + ky;
      if (ih >= h) continue;
      for (int kx = 0; kx < k; ++kx) {
        const int iw = iw0 + kx;
        if (iw >= w) continue;
        const float value = x[(((static_cast<int64_t>(ni) * c + ch) * h + ih) * w) + iw];
        if (value > best) {
          best = value;
          best_s = ih * w + iw;
        }
      }
    }
    y[idx] = best;
    index[idx] = best_s;
  });
  cache_y_->copy_from(*output_);
  return *output_;
}

const Tensor& MaxPool2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !cache_y_ || grad_output.shape() != cache_y_->shape()) {
    throw Error("MaxPool2d backward shape");
  }
  ensure_tensor(grad_input_, *ctx_, cache_x_->shape());
  grad_input_->zero();
  const int n = cache_x_->shape()[0];
  const int c = cache_x_->shape()[1];
  const int h = cache_x_->shape()[2];
  const int w = cache_x_->shape()[3];
  const int ho = cache_y_->shape()[2];
  const int wo = cache_y_->shape()[3];
  float* dx = grad_input_->data();
  const float* dy = grad_output.data();
  const int* index = impl_->index;
  for_each(queue_of(*ctx_), "max_pool_backward", grad_output.numel(), [=](int64_t idx) {
    const int ch = static_cast<int>((idx / (static_cast<int64_t>(wo) * ho)) % c);
    const int ni = static_cast<int>(idx / (static_cast<int64_t>(wo) * ho * c));
    const int spatial = index[idx];
    const int ih = spatial / w;
    const int iw = spatial - ih * w;
    atomic_add(dx + (((static_cast<int64_t>(ni) * c + ch) * h + ih) * w) + iw, dy[idx]);
  });
  return *grad_input_;
}

}  // namespace nn
