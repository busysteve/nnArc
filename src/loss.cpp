#include "nn/loss.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"

#include <utility>

namespace nn {
namespace {

void copy_int(sycl::queue& q, int* host, const int* device) {
  try {
    q.memcpy(host, device, sizeof(int)).wait();
  } catch (const sycl::exception& ex) {
    backend_error("sycl::memcpy", ex.what());
  }
}

}  // namespace

DeviceIndices::DeviceIndices(Context& ctx, int64_t n) : ctx_(&ctx), n_(n) {
  if (n <= 0) throw Error("empty labels");
  data_ = device_alloc<int64_t>(queue_of(*ctx_), static_cast<size_t>(n), "sycl::malloc_device");
}

DeviceIndices::~DeviceIndices() {
  if (!data_) return;
  if (ctx_) {
    try {
      queue_of(*ctx_).wait();
    } catch (...) {
    }
    device_free(queue_of(*ctx_), data_);
  }
  data_ = nullptr;
}

DeviceIndices::DeviceIndices(DeviceIndices&& other) noexcept
    : ctx_(other.ctx_), data_(other.data_), n_(other.n_) {
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
}

DeviceIndices& DeviceIndices::operator=(DeviceIndices&& other) noexcept {
  if (this == &other) return *this;
  if (data_) {
    if (ctx_) {
      try {
        queue_of(*ctx_).wait();
      } catch (...) {
      }
      device_free(queue_of(*ctx_), data_);
    }
  }
  ctx_ = other.ctx_;
  data_ = other.data_;
  n_ = other.n_;
  other.ctx_ = nullptr;
  other.data_ = nullptr;
  other.n_ = 0;
  return *this;
}

void DeviceIndices::copy_from_host(const int64_t* src, int64_t n) {
  if (n != n_) throw Error("label count mismatch");
  memcpy_async(queue_of(*ctx_), data_, src, static_cast<size_t>(n) * sizeof(int64_t), "sycl::memcpy");
}

LossResult softmax_cross_entropy(Context& ctx, const Tensor& logits, const DeviceIndices& labels) {
  if (logits.rank() != 2) throw Error("logits rank");
  const int n = logits.shape()[0];
  const int classes = logits.shape()[1];
  if (n <= 0 || classes <= 1) throw Error("logits shape");
  if (labels.size() != n) throw Error("label count mismatch");

  LossResult result{0.f, Tensor(ctx, {n, classes})};
  sycl::queue& q = queue_of(ctx);
  float* per_sample = device_alloc<float>(q, static_cast<size_t>(n), "sycl::malloc_device");
  int* bad = device_alloc<int>(q, 1, "sycl::malloc_device");
  try {
    memset_async(q, bad, 0, sizeof(int), "sycl::memset");
    const float* row_logits_base = logits.data();
    const int64_t* label_base = labels.data();
    float* grad = result.grad_logits.data();
    for_groups(q, "softmax_cross_entropy", n, [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
      const int row = static_cast<int>(item.get_group(0));
      const int tid = static_cast<int>(item.get_local_id(0));
      const float* row_logits = row_logits_base + static_cast<int64_t>(row) * classes;
      float* row_grad = grad + static_cast<int64_t>(row) * classes;
      float thread_max = -1e30f;
      for (int c = tid; c < classes; c += kThreads) thread_max = sycl::fmax(thread_max, row_logits[c]);
      const float max_logit = block_max(item, thread_max, scratch);
      float thread_sum = 0.f;
      for (int c = tid; c < classes; c += kThreads) thread_sum += sycl::exp(row_logits[c] - max_logit);
      const float sum = block_sum(item, thread_sum, scratch);
      const int64_t label = label_base[row];
      if (label < 0 || label >= classes) {
        if (tid == 0) atomic_exchange(bad, 1);
        return;
      }
      const float inv_sum = 1.f / sum;
      const float inv_n = 1.f / static_cast<float>(n);
      for (int c = tid; c < classes; c += kThreads) {
        const float softmax = sycl::exp(row_logits[c] - max_logit) * inv_sum;
        const float indicator = (static_cast<int64_t>(c) == label) ? 1.f : 0.f;
        row_grad[c] = (softmax - indicator) * inv_n;
      }
      if (tid == 0) per_sample[row] = -(row_logits[label] - (max_logit + sycl::log(sum)));
    });
    const float sum = launch_sum(ctx, per_sample, n);
    ctx.synchronize();
    int bad_host = 0;
    copy_int(q, &bad_host, bad);
    device_free(q, per_sample);
    device_free(q, bad);
    per_sample = nullptr;
    bad = nullptr;
    if (bad_host) throw Error("label out of range");
    result.loss = sum / static_cast<float>(n);
    return result;
  } catch (...) {
    device_free(q, per_sample);
    device_free(q, bad);
    throw;
  }
}

LossResult sequence_cross_entropy(Context& ctx, const Tensor& logits, const IndexTensor& labels,
                                  int64_t ignore_index) {
  int classes = 0;
  int rows = 0;
  if (logits.rank() == 2) {
    rows = logits.shape()[0];
    classes = logits.shape()[1];
    if (labels.shape().size() != 1 || labels.shape()[0] != rows) throw Error("label shape");
  } else if (logits.rank() == 3) {
    if (labels.shape().size() != 2 || labels.shape()[0] != logits.shape()[0] ||
        labels.shape()[1] != logits.shape()[1]) {
      throw Error("label shape");
    }
    classes = logits.shape()[2];
    rows = static_cast<int>(static_cast<int64_t>(logits.shape()[0]) * logits.shape()[1]);
  } else {
    throw Error("logits rank");
  }
  if (classes <= 1) throw Error("logits shape");

  LossResult result{0.f, Tensor(ctx, logits.shape())};
  sycl::queue& q = queue_of(ctx);
  int* counts = device_alloc<int>(q, 2, "sycl::malloc_device");
  int* bad = counts;
  int* n_valid_dev = counts + 1;
  float* per_sample = nullptr;
  try {
    memset_async(q, counts, 0, 2 * sizeof(int), "sycl::memset");
    const int64_t* label_base = labels.data();
    for_each(q, "sequence_cross_entropy", rows, [=](int64_t row) {
      const int64_t label = label_base[row];
      if (label == ignore_index) return;
      if (label < 0 || label >= static_cast<int64_t>(classes)) {
        atomic_exchange(bad, 1);
        return;
      }
      atomic_add(n_valid_dev, 1);
    });
    ctx.synchronize();
    int bad_host = 0;
    int n_valid = 0;
    copy_int(q, &bad_host, bad);
    copy_int(q, &n_valid, n_valid_dev);
    if (bad_host) {
      device_free(q, counts);
      counts = nullptr;
      throw Error("label out of range");
    }
    if (n_valid == 0) {
      device_free(q, counts);
      counts = nullptr;
      result.grad_logits.zero();
      ctx.synchronize();
      return result;
    }

    per_sample = device_alloc<float>(q, static_cast<size_t>(rows), "sycl::malloc_device");
    const float* logits_base = logits.data();
    float* grad = result.grad_logits.data();
    for_groups(q, "sequence_cross_entropy", rows,
               [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
                 const int row = static_cast<int>(item.get_group(0));
                 const int tid = static_cast<int>(item.get_local_id(0));
                 const float* row_logits = logits_base + static_cast<int64_t>(row) * classes;
                 float* row_grad = grad + static_cast<int64_t>(row) * classes;
                 const int64_t label = label_base[row];
                 if (label == ignore_index) {
                   for (int c = tid; c < classes; c += kThreads) row_grad[c] = 0.f;
                   if (tid == 0) per_sample[row] = 0.f;
                   return;
                 }
                 if (label < 0 || label >= static_cast<int64_t>(classes)) {
                   for (int c = tid; c < classes; c += kThreads) row_grad[c] = 0.f;
                   if (tid == 0) {
                     atomic_exchange(bad, 1);
                     per_sample[row] = 0.f;
                   }
                   return;
                 }
                 float thread_max = -1e30f;
                 for (int c = tid; c < classes; c += kThreads) thread_max = sycl::fmax(thread_max, row_logits[c]);
                 const float max_logit = block_max(item, thread_max, scratch);
                 float thread_sum = 0.f;
                 for (int c = tid; c < classes; c += kThreads) thread_sum += sycl::exp(row_logits[c] - max_logit);
                 const float sum = block_sum(item, thread_sum, scratch);
                 const float inv_sum = 1.f / sum;
                 const float inv_n = 1.f / static_cast<float>(n_valid);
                 for (int c = tid; c < classes; c += kThreads) {
                   const float softmax = sycl::exp(row_logits[c] - max_logit) * inv_sum;
                   const float indicator = (static_cast<int64_t>(c) == label) ? 1.f : 0.f;
                   row_grad[c] = (softmax - indicator) * inv_n;
                 }
                 if (tid == 0) per_sample[row] = -(row_logits[label] - (max_logit + sycl::log(sum)));
               });
    const float sum = launch_sum(ctx, per_sample, rows);
    ctx.synchronize();
    copy_int(q, &bad_host, bad);
    device_free(q, per_sample);
    device_free(q, counts);
    per_sample = nullptr;
    counts = nullptr;
    if (bad_host) throw Error("label out of range");
    result.loss = sum / static_cast<float>(n_valid);
    return result;
  } catch (...) {
    device_free(q, per_sample);
    device_free(q, counts);
    throw;
  }
}

}  // namespace nn
