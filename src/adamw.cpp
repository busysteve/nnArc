#include "kernels.hpp"

#include "context_impl.hpp"
#include "device.hpp"

namespace nn {
namespace {

inline void adamw_one(float& param, float& m, float& v, float g, float lr, float beta1, float beta2, float eps,
                      float weight_decay, float bc1, float bc2) {
  const float m_new = beta1 * m + (1.f - beta1) * g;
  const float v_new = beta2 * v + (1.f - beta2) * g * g;
  m = m_new;
  v = v_new;
  const float m_hat = m_new / bc1;
  const float v_hat = v_new / bc2;
  const float update = m_hat / (sycl::sqrt(v_hat) + eps) + weight_decay * param;
  param = param - lr * update;
}

void partial_reduce(sycl::queue& q, const char* call, const float* x, int64_t n, int blocks, double* partials,
                    bool squares) {
  submit(q, call, [&](sycl::queue& queue) {
    queue.submit([&](sycl::handler& h) {
      sycl::local_accessor<double, 1> buffer(sycl::range<1>(kThreads), h);
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(blocks) * kThreads),
                                       sycl::range<1>(kThreads)),
                     [=](sycl::nd_item<1> item) {
                       const int tid = static_cast<int>(item.get_local_id(0));
                       const int bid = static_cast<int>(item.get_group(0));
                       double sum = 0.0;
                       const int64_t stride = static_cast<int64_t>(kThreads) * blocks;
                       for (int64_t i = static_cast<int64_t>(bid) * kThreads + tid; i < n; i += stride) {
                         const double value = static_cast<double>(x[i]);
                         sum += squares ? value * value : value;
                       }
                       buffer[tid] = sum;
                       item.barrier(sycl::access::fence_space::local_space);
                       for (int offset = 128; offset > 0; offset >>= 1) {
                         if (tid < offset) buffer[tid] += buffer[tid + offset];
                         item.barrier(sycl::access::fence_space::local_space);
                       }
                       if (tid == 0) partials[bid] = buffer[0];
                     });
    });
  });
}

double reduce_partials(Context& ctx, int blocks, double* scratch) {
  sycl::queue& q = queue_of(ctx);
  double* partials = scratch;
  double* out = scratch + blocks;
  submit(q, "sum_partials", [&](sycl::queue& queue) {
    queue.submit([&](sycl::handler& h) {
      sycl::local_accessor<double, 1> buffer(sycl::range<1>(kThreads), h);
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(kThreads), sycl::range<1>(kThreads)),
                     [=](sycl::nd_item<1> item) {
                       const int tid = static_cast<int>(item.get_local_id(0));
                       double sum = 0.0;
                       for (int i = tid; i < blocks; i += kThreads) sum += partials[i];
                       buffer[tid] = sum;
                       item.barrier(sycl::access::fence_space::local_space);
                       for (int offset = 128; offset > 0; offset >>= 1) {
                         if (tid < offset) buffer[tid] += buffer[tid + offset];
                         item.barrier(sycl::access::fence_space::local_space);
                       }
                       if (tid == 0) *out = buffer[0];
                     });
    });
  });
  ctx.synchronize();
  double host = 0.0;
  try {
    q.memcpy(&host, out, sizeof(double)).wait();
  } catch (const sycl::exception& ex) {
    backend_error("sycl::memcpy", ex.what());
  }
  return host;
}

double reduce_sum(Context& ctx, const float* x, int64_t n, bool squares, const char* call) {
  int64_t blocks64 = (n + kThreads - 1) / kThreads;
  if (blocks64 > 1024) blocks64 = 1024;
  if (blocks64 < 1) blocks64 = 1;
  const int blocks = static_cast<int>(blocks64);
  double* scratch = device_alloc<double>(queue_of(ctx), static_cast<size_t>(blocks + 1), "sycl::malloc_device");
  try {
    partial_reduce(queue_of(ctx), call, x, n, blocks, scratch, squares);
    const double host = reduce_partials(ctx, blocks, scratch);
    device_free(queue_of(ctx), scratch);
    return host;
  } catch (...) {
    device_free(queue_of(ctx), scratch);
    throw;
  }
}

}  // namespace

void launch_adamw(sycl::queue& q, float* param, float* m, float* v, const float* grad, int64_t n, float lr,
                  float beta1, float beta2, float eps, float weight_decay, float bc1, float bc2) {
  if (n <= 0) return;
  if ((n % 4) == 0 && aligned16(param) && aligned16(m) && aligned16(v) && aligned16(grad)) {
    for_each(q, "adamw", n / 4, [=](int64_t i) {
      f4 p = load4(param + 4 * i);
      f4 mm = load4(m + 4 * i);
      f4 vv = load4(v + 4 * i);
      const f4 g = load4(grad + 4 * i);
      float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
      float m0 = mm[0], m1 = mm[1], m2 = mm[2], m3 = mm[3];
      float v0 = vv[0], v1 = vv[1], v2 = vv[2], v3 = vv[3];
      adamw_one(p0, m0, v0, g[0], lr, beta1, beta2, eps, weight_decay, bc1, bc2);
      adamw_one(p1, m1, v1, g[1], lr, beta1, beta2, eps, weight_decay, bc1, bc2);
      adamw_one(p2, m2, v2, g[2], lr, beta1, beta2, eps, weight_decay, bc1, bc2);
      adamw_one(p3, m3, v3, g[3], lr, beta1, beta2, eps, weight_decay, bc1, bc2);
      store4(param + 4 * i, make4(p0, p1, p2, p3));
      store4(m + 4 * i, make4(m0, m1, m2, m3));
      store4(v + 4 * i, make4(v0, v1, v2, v3));
    });
  } else {
    for_each(q, "adamw", n, [=](int64_t i) {
      adamw_one(param[i], m[i], v[i], grad[i], lr, beta1, beta2, eps, weight_decay, bc1, bc2);
    });
  }
}

double launch_sum_squares(Context& ctx, const float* x, int64_t n) {
  if (n <= 0) return 0.0;
  return reduce_sum(ctx, x, n, true, "sum_squares");
}

float launch_sum(Context& ctx, const float* x, int64_t n) {
  if (n <= 0) return 0.f;
  return static_cast<float>(reduce_sum(ctx, x, n, false, "sum"));
}

}  // namespace nn
