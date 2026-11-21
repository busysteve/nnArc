#include "nn/index_tensor.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "nn/error.hpp"

#include <climits>
#include <utility>

namespace nn {
namespace {

int64_t checked_numel(const std::vector<int>& shape) {
  if (shape.size() != 1 && shape.size() != 2) throw Error("index tensor rank out of range");
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) throw Error("index tensor dimension out of range");
    if (n > static_cast<int64_t>(INT32_MAX) / dim) throw Error("index tensor numel out of range");
    n *= dim;
  }
  return n;
}

}  // namespace

IndexTensor::IndexTensor(Context& ctx, std::vector<int> shape)
    : ctx_(&ctx), shape_(std::move(shape)) {
  numel_ = checked_numel(shape_);
  data_ = device_alloc<int64_t>(queue_of(*ctx_), static_cast<size_t>(numel_), "sycl::malloc_device");
}

void IndexTensor::release() noexcept {
  if (!data_) return;
  if (ctx_) {
    try {
      queue_of(*ctx_).wait();
    } catch (...) {
    }
    device_free(queue_of(*ctx_), data_);
  }
  data_ = nullptr;
  numel_ = 0;
}

IndexTensor::~IndexTensor() { release(); }

IndexTensor::IndexTensor(IndexTensor&& other) noexcept
    : ctx_(other.ctx_), shape_(std::move(other.shape_)), numel_(other.numel_), data_(other.data_) {
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
}

IndexTensor& IndexTensor::operator=(IndexTensor&& other) noexcept {
  if (this == &other) return *this;
  release();
  ctx_ = other.ctx_;
  shape_ = std::move(other.shape_);
  numel_ = other.numel_;
  data_ = other.data_;
  other.ctx_ = nullptr;
  other.numel_ = 0;
  other.data_ = nullptr;
  return *this;
}

void IndexTensor::copy_from_host(const int64_t* src, int64_t n) {
  if (n != numel_) throw Error("index tensor host copy numel mismatch");
  memcpy_async(queue_of(*ctx_), data_, src, static_cast<size_t>(n) * sizeof(int64_t), "sycl::memcpy");
}

std::vector<int64_t> IndexTensor::to_host() const {
  ctx_->synchronize();
  std::vector<int64_t> host(static_cast<size_t>(numel_));
  try {
    queue_of(*ctx_).memcpy(host.data(), data_, host.size() * sizeof(int64_t)).wait();
  } catch (const sycl::exception& ex) {
    backend_error("sycl::memcpy", ex.what());
  }
  return host;
}

}  // namespace nn
