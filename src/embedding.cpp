#include "nn/layers/embedding.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <cstdint>
#include <vector>

namespace nn {
namespace {

std::vector<int> embedding_weight_shape(int num_embeddings, int embedding_dim) {
  if (num_embeddings < 1 || embedding_dim < 1) throw Error("Embedding size");
  return {num_embeddings, embedding_dim};
}

void copy_indices(sycl::queue& q, IndexTensor& dst, const IndexTensor& src) {
  memcpy_async(q, const_cast<int64_t*>(dst.data()), src.data(),
               static_cast<size_t>(src.numel()) * sizeof(int64_t), "sycl::memcpy");
}

void require_indices(Context& ctx, const int64_t* indices, int64_t rows, int vocab) {
  sycl::queue& q = queue_of(ctx);
  int* bad = device_alloc<int>(q, 1, "sycl::malloc_device");
  try {
    memset_async(q, bad, 0, sizeof(int), "sycl::memset");
    for_each(q, "embedding_check", rows, [=](int64_t i) {
      const int64_t id = indices[i];
      if (id < 0 || id >= vocab) atomic_exchange(bad, 1);
    });
    ctx.synchronize();
    int bad_host = 0;
    try {
      q.memcpy(&bad_host, bad, sizeof(int)).wait();
    } catch (const sycl::exception& ex) {
      backend_error("sycl::memcpy", ex.what());
    }
    device_free(q, bad);
    bad = nullptr;
    if (bad_host) throw Error("index out of range");
  } catch (...) {
    device_free(q, bad);
    throw;
  }
}

}  // namespace

Embedding::Embedding(Context& ctx, int num_embeddings, int embedding_dim, std::mt19937& rng)
    : ctx_(&ctx),
      num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      weight_(ctx, embedding_weight_shape(num_embeddings, embedding_dim)),
      grad_weight_(ctx, {num_embeddings, embedding_dim}) {
  normal_(weight_, 0.f, 0.02f, rng);
  grad_weight_.zero();
}

std::vector<NamedParam> Embedding::parameters() const {
  return {NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)}};
}

void Embedding::zero_grad() { grad_weight_.zero(); }

const Tensor& Embedding::forward(const Tensor&) { throw Error("Embedding requires indices"); }

const Tensor& Embedding::forward_indices(const IndexTensor& indices) {
  if (&indices.context() != ctx_) throw Error("Embedding context");
  sycl::queue& q = queue_of(*ctx_);
  require_indices(*ctx_, indices.data(), indices.numel(), num_embeddings_);
  if (!cache_indices_ || cache_indices_->shape() != indices.shape()) {
    cache_indices_.emplace(*ctx_, indices.shape());
  }
  copy_indices(q, *cache_indices_, indices);

  std::vector<int> out_shape = indices.shape();
  out_shape.push_back(embedding_dim_);
  ensure_tensor(output_, *ctx_, out_shape);
  const int64_t rows = cache_indices_->numel();
  const int64_t n = rows * static_cast<int64_t>(embedding_dim_);
  const int64_t* ids = cache_indices_->data();
  const float* weight = weight_.data();
  float* out = output_->data();
  const int dim = embedding_dim_;
  if ((dim % 4) == 0 && aligned16(out) && aligned16(weight)) {
    const int dim4 = dim / 4;
    for_each(q, "embedding_gather", n / 4, [=](int64_t i) {
      const int64_t row = i / dim4;
      const int col = static_cast<int>(i - row * dim4);
      const int64_t id = ids[row];
      store4(out + 4 * i, load4(weight + (id * dim4 + col) * 4));
    });
  } else {
    for_each(q, "embedding_gather", n, [=](int64_t i) {
      const int64_t row = i / dim;
      const int col = static_cast<int>(i - row * dim);
      const int64_t id = ids[row];
      out[i] = weight[id * static_cast<int64_t>(dim) + col];
    });
  }
  return *output_;
}

const Tensor& Embedding::backward(const Tensor& grad_output) {
  if (!cache_indices_ || !output_) throw Error("Embedding backward without forward");
  if (grad_output.shape() != output_->shape()) throw Error("Embedding backward shape");
  const int64_t rows = cache_indices_->numel();
  const int64_t n = rows * static_cast<int64_t>(embedding_dim_);
  const int dim = embedding_dim_;
  const int64_t* ids = cache_indices_->data();
  float* grad_weight = grad_weight_.data();
  const float* grad = grad_output.data();
  for_each(queue_of(*ctx_), "embedding_scatter", n, [=](int64_t i) {
    const int64_t row = i / dim;
    const int col = static_cast<int>(i - row * dim);
    const int64_t id = ids[row];
    atomic_add(grad_weight + id * static_cast<int64_t>(dim) + col, grad[i]);
  });
  ensure_tensor(grad_input_, *ctx_, grad_output.shape());
  grad_input_->zero();
  return *grad_input_;
}

}  // namespace nn
