#include "kernels.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "nn/tensor.hpp"

namespace nn {
namespace {

inline float gelu_forward(float x) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  return 0.5f * x * (1.f + sycl::erf(x * kInvSqrt2));
}

inline float gelu_grad(float x) {
  constexpr float kInvSqrt2 = 0.7071067811865476f;
  constexpr float kInvSqrt2Pi = 0.3989422804014327f;
  const float cdf = 0.5f * (1.f + sycl::erf(x * kInvSqrt2));
  const float pdf = kInvSqrt2Pi * sycl::exp(-0.5f * x * x);
  return cdf + x * pdf;
}

inline float sigmoid_stable(float x) {
  if (x >= 0.f) {
    const float z = sycl::exp(-x);
    return 1.f / (1.f + z);
  }
  const float z = sycl::exp(x);
  return z / (1.f + z);
}

inline float silu(float x) { return x * sigmoid_stable(x); }

inline float silu_grad(float x) {
  const float s = sigmoid_stable(x);
  return s * (1.f + x * (1.f - s));
}

bool vec4(const void* a, const void* b, int64_t n) {
  return n >= 4 && aligned16(a) && aligned16(b) && (n % 4) == 0;
}

bool vec4(const void* a, const void* b, const void* c, int64_t n) {
  return n >= 4 && aligned16(a) && aligned16(b) && aligned16(c) && (n % 4) == 0;
}

}  // namespace

void launch_fill(sycl::queue& q, float* x, int64_t n, float value) {
  if (n <= 0) return;
  if (aligned16(x) && n >= 4) {
    const int64_t n4 = n / 4;
    for_each(q, "fill", n4, [=](int64_t i) { store4(x + 4 * i, make4(value, value, value, value)); });
    const int64_t tail = n - n4 * 4;
    if (tail) for_each(q, "fill", tail, [=](int64_t i) { x[n4 * 4 + i] = value; });
  } else {
    for_each(q, "fill", n, [=](int64_t i) { x[i] = value; });
  }
}

void launch_add_bias(sycl::queue& q, float* y, const float* bias, int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  if (n <= 0) return;
  if ((cols % 4) == 0 && aligned16(y) && aligned16(bias)) {
    for_each(q, "add_bias", n / 4, [=](int64_t i) {
      const int col = static_cast<int>((i * 4) % cols);
      const f4 v = load4(y + 4 * i);
      const f4 b = load4(bias + col);
      store4(y + 4 * i, make4(v[0] + b[0], v[1] + b[1], v[2] + b[2], v[3] + b[3]));
    });
  } else {
    for_each(q, "add_bias", n, [=](int64_t i) { y[i] += bias[static_cast<int>(i % cols)]; });
  }
}

void launch_bias_grad(sycl::queue& q, const float* grad_y, float* grad_bias, int rows, int cols) {
  if (cols <= 0) return;
  for_each(q, "bias_grad", cols, [=](int64_t col) {
    float sum = 0.f;
    for (int row = 0; row < rows; ++row) sum += grad_y[static_cast<int64_t>(row) * cols + col];
    grad_bias[col] += sum;
  });
}

void launch_add2(sycl::queue& q, float* dst, const float* a, const float* b, int64_t n) {
  if (n <= 0) return;
  if (vec4(dst, a, b, n)) {
    for_each(q, "add2", n / 4, [=](int64_t i) {
      const f4 av = load4(a + 4 * i);
      const f4 bv = load4(b + 4 * i);
      store4(dst + 4 * i, make4(av[0] + bv[0], av[1] + bv[1], av[2] + bv[2], av[3] + bv[3]));
    });
  } else {
    for_each(q, "add2", n, [=](int64_t i) { dst[i] = a[i] + b[i]; });
  }
}

void launch_mul(sycl::queue& q, float* y, const float* a, const float* b, int64_t n) {
  if (n <= 0) return;
  if (vec4(y, a, b, n)) {
    for_each(q, "mul", n / 4, [=](int64_t i) {
      const f4 av = load4(a + 4 * i);
      const f4 bv = load4(b + 4 * i);
      store4(y + 4 * i, make4(av[0] * bv[0], av[1] * bv[1], av[2] * bv[2], av[3] * bv[3]));
    });
  } else {
    for_each(q, "mul", n, [=](int64_t i) { y[i] = a[i] * b[i]; });
  }
}

void Tensor::scale(float factor) { launch_scale(queue_of(*ctx_), data_, numel_, factor); }

void launch_scale(sycl::queue& q, float* x, int64_t n, float scale) {
  if (n <= 0) return;
  if (aligned16(x) && n >= 4) {
    const int64_t n4 = n / 4;
    for_each(q, "scale", n4, [=](int64_t i) {
      const f4 v = load4(x + 4 * i);
      store4(x + 4 * i, make4(v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale));
    });
    const int64_t tail = n - n4 * 4;
    if (tail) for_each(q, "scale", tail, [=](int64_t i) { x[n4 * 4 + i] *= scale; });
  } else {
    for_each(q, "scale", n, [=](int64_t i) { x[i] *= scale; });
  }
}

void launch_gelu(sycl::queue& q, float* y, const float* x, int64_t n) {
  if (n <= 0) return;
  if (vec4(y, x, n)) {
    for_each(q, "gelu", n / 4, [=](int64_t i) {
      const f4 v = load4(x + 4 * i);
      store4(y + 4 * i, make4(gelu_forward(v[0]), gelu_forward(v[1]), gelu_forward(v[2]), gelu_forward(v[3])));
    });
  } else {
    for_each(q, "gelu", n, [=](int64_t i) { y[i] = gelu_forward(x[i]); });
  }
}

void launch_gelu_backward(sycl::queue& q, float* grad_x, const float* x, const float* grad_y, int64_t n) {
  if (n <= 0) return;
  if (vec4(grad_x, x, grad_y, n)) {
    for_each(q, "gelu_backward", n / 4, [=](int64_t i) {
      const f4 xv = load4(x + 4 * i);
      const f4 gv = load4(grad_y + 4 * i);
      store4(grad_x + 4 * i, make4(gv[0] * gelu_grad(xv[0]), gv[1] * gelu_grad(xv[1]), gv[2] * gelu_grad(xv[2]),
                                   gv[3] * gelu_grad(xv[3])));
    });
  } else {
    for_each(q, "gelu_backward", n, [=](int64_t i) { grad_x[i] = grad_y[i] * gelu_grad(x[i]); });
  }
}

void launch_swiglu(sycl::queue& q, float* y, const float* x, int rows, int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  if (n <= 0) return;
  if ((inner % 4) == 0 && aligned16(y) && aligned16(x)) {
    for_each(q, "swiglu", n / 4, [=](int64_t i) {
      const int64_t elem = i * 4;
      const int64_t row = elem / inner;
      const int64_t col = elem - row * inner;
      const int64_t base = row * (inner * 2) + col;
      const f4 a = load4(x + base);
      const f4 b = load4(x + base + inner);
      store4(y + 4 * i, make4(silu(a[0]) * b[0], silu(a[1]) * b[1], silu(a[2]) * b[2], silu(a[3]) * b[3]));
    });
  } else {
    for_each(q, "swiglu", n, [=](int64_t i) {
      const int64_t row = i / inner;
      const int64_t col = i - row * inner;
      const int64_t base = row * (inner * 2) + col;
      y[i] = silu(x[base]) * x[base + inner];
    });
  }
}

void launch_swiglu_backward(sycl::queue& q, float* grad_x, const float* x, const float* grad_y, int rows,
                            int64_t inner) {
  const int64_t n = static_cast<int64_t>(rows) * inner;
  if (n <= 0) return;
  if ((inner % 4) == 0 && aligned16(grad_x) && aligned16(x) && aligned16(grad_y)) {
    for_each(q, "swiglu_backward", n / 4, [=](int64_t i) {
      const int64_t elem = i * 4;
      const int64_t row = elem / inner;
      const int64_t col = elem - row * inner;
      const int64_t base = row * (inner * 2) + col;
      const f4 a = load4(x + base);
      const f4 b = load4(x + base + inner);
      const f4 dy = load4(grad_y + 4 * i);
      store4(grad_x + base, make4(dy[0] * b[0] * silu_grad(a[0]), dy[1] * b[1] * silu_grad(a[1]),
                                  dy[2] * b[2] * silu_grad(a[2]), dy[3] * b[3] * silu_grad(a[3])));
      store4(grad_x + base + inner, make4(dy[0] * silu(a[0]), dy[1] * silu(a[1]), dy[2] * silu(a[2]),
                                          dy[3] * silu(a[3])));
    });
  } else {
    for_each(q, "swiglu_backward", n, [=](int64_t i) {
      const int64_t row = i / inner;
      const int64_t col = i - row * inner;
      const int64_t base = row * (inner * 2) + col;
      const float gate = x[base];
      const float value = x[base + inner];
      const float dy = grad_y[i];
      grad_x[base] = dy * value * silu_grad(gate);
      grad_x[base + inner] = dy * silu(gate);
    });
  }
}

void launch_tanh(sycl::queue& q, float* y, const float* x, int64_t n) {
  if (n <= 0) return;
  if (vec4(y, x, n)) {
    for_each(q, "tanh", n / 4, [=](int64_t i) {
      const f4 v = load4(x + 4 * i);
      store4(y + 4 * i, make4(sycl::tanh(v[0]), sycl::tanh(v[1]), sycl::tanh(v[2]), sycl::tanh(v[3])));
    });
  } else {
    for_each(q, "tanh", n, [=](int64_t i) { y[i] = sycl::tanh(x[i]); });
  }
}

void launch_tanh_backward(sycl::queue& q, float* grad_x, const float* y, const float* grad_y, int64_t n) {
  if (n <= 0) return;
  if (vec4(grad_x, y, grad_y, n)) {
    for_each(q, "tanh_backward", n / 4, [=](int64_t i) {
      const f4 hv = load4(y + 4 * i);
      const f4 gv = load4(grad_y + 4 * i);
      store4(grad_x + 4 * i, make4(gv[0] * (1.f - hv[0] * hv[0]), gv[1] * (1.f - hv[1] * hv[1]),
                                   gv[2] * (1.f - hv[2] * hv[2]), gv[3] * (1.f - hv[3] * hv[3])));
    });
  } else {
    for_each(q, "tanh_backward", n, [=](int64_t i) {
      const float h = y[i];
      grad_x[i] = grad_y[i] * (1.f - h * h);
    });
  }
}

void launch_gather_time(sycl::queue& q, float* dst, const float* src, int batch, int time, int features,
                        int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  if (n <= 0) return;
  for_each(q, "gather_time", n, [=](int64_t i) {
    const int64_t row = i / features;
    const int64_t col = i - row * features;
    dst[i] = src[(row * time + index) * features + col];
  });
}

void launch_scatter_time(sycl::queue& q, float* dst, const float* src, int batch, int time, int features,
                         int index) {
  const int64_t n = static_cast<int64_t>(batch) * features;
  if (n <= 0) return;
  for_each(q, "scatter_time", n, [=](int64_t i) {
    const int64_t row = i / features;
    const int64_t col = i - row * features;
    dst[(row * time + index) * features + col] = src[i];
  });
}

}  // namespace nn
