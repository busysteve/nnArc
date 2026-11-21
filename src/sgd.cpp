#include "kernels.hpp"

#include "device.hpp"

namespace nn {
namespace {

inline void sgd_one(float& param, float& velocity, float grad, float lr, float momentum) {
  const float updated = momentum * velocity + grad;
  velocity = updated;
  param = param - lr * updated;
}

}  // namespace

void launch_sgd(sycl::queue& q, float* param, float* velocity, const float* grad, int64_t n,
                float learning_rate, float momentum) {
  if (n <= 0) return;
  if ((n % 4) == 0 && aligned16(param) && aligned16(velocity) && aligned16(grad)) {
    for_each(q, "sgd", n / 4, [=](int64_t i) {
      f4 p = load4(param + 4 * i);
      f4 vel = load4(velocity + 4 * i);
      const f4 g = load4(grad + 4 * i);
      float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
      float v0 = vel[0], v1 = vel[1], v2 = vel[2], v3 = vel[3];
      sgd_one(p0, v0, g[0], learning_rate, momentum);
      sgd_one(p1, v1, g[1], learning_rate, momentum);
      sgd_one(p2, v2, g[2], learning_rate, momentum);
      sgd_one(p3, v3, g[3], learning_rate, momentum);
      store4(param + 4 * i, make4(p0, p1, p2, p3));
      store4(velocity + 4 * i, make4(v0, v1, v2, v3));
    });
  } else {
    for_each(q, "sgd", n, [=](int64_t i) { sgd_one(param[i], velocity[i], grad[i], learning_rate, momentum); });
  }
}

}  // namespace nn
