#include "nn/layers/relu.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"

namespace nn {

struct ReLU::Impl {};

namespace {

void require_rank(const std::vector<int>& shape) {
  if (shape.size() != 2 && shape.size() != 3 && shape.size() != 4) throw Error("ReLU rank");
}

}  // namespace

ReLU::ReLU(Context& ctx) : ctx_(&ctx), impl_(std::make_unique<Impl>()) {}

ReLU::~ReLU() = default;

std::vector<NamedParam> ReLU::parameters() const { return {}; }

void ReLU::zero_grad() {}

const Tensor& ReLU::forward(const Tensor& input) {
  require_rank(input.shape());
  ensure_tensor(cache_x_, *ctx_, input.shape());
  ensure_tensor(output_, *ctx_, input.shape());
  cache_x_->copy_from(input);
  const int64_t n = cache_x_->numel();
  sycl::queue& q = queue_of(*ctx_);
  float* y = output_->data();
  const float* x = cache_x_->data();
  if (aligned16(y) && aligned16(x) && (n % 4) == 0) {
    for_each(q, "relu", n / 4, [=](int64_t i) {
      const f4 v = load4(x + 4 * i);
      store4(y + 4 * i, make4(sycl::fmax(v[0], 0.f), sycl::fmax(v[1], 0.f), sycl::fmax(v[2], 0.f),
                              sycl::fmax(v[3], 0.f)));
    });
  } else {
    for_each(q, "relu", n, [=](int64_t i) { y[i] = sycl::fmax(x[i], 0.f); });
  }
  return *output_;
}

const Tensor& ReLU::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) throw Error("ReLU backward shape");
  ensure_tensor(grad_input_, *ctx_, grad_output.shape());
  const int64_t n = cache_x_->numel();
  sycl::queue& q = queue_of(*ctx_);
  float* dx = grad_input_->data();
  const float* x = cache_x_->data();
  const float* dy = grad_output.data();
  if (aligned16(dx) && aligned16(x) && aligned16(dy) && (n % 4) == 0) {
    for_each(q, "relu_backward", n / 4, [=](int64_t i) {
      const f4 xv = load4(x + 4 * i);
      const f4 gv = load4(dy + 4 * i);
      store4(dx + 4 * i, make4(xv[0] > 0.f ? gv[0] : 0.f, xv[1] > 0.f ? gv[1] : 0.f, xv[2] > 0.f ? gv[2] : 0.f,
                               xv[3] > 0.f ? gv[3] : 0.f));
    });
  } else {
    for_each(q, "relu_backward", n, [=](int64_t i) { dx[i] = x[i] > 0.f ? dy[i] : 0.f; });
  }
  return *grad_input_;
}

}  // namespace nn
