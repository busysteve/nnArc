#include "nn/layers/multi_head_attention.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "gemm.hpp"
#include "kernels.hpp"
#include "nn/error.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {
namespace {

constexpr float kMask = -1.0e9f;

inline float masked_score(float value, int col, int query, int causal) {
  if (causal && col > query) return kMask;
  return value;
}

void launch_split_heads(sycl::queue& q, float* dst, const float* src, int time, int heads, int dh, int64_t n) {
  if ((dh % 4) == 0 && aligned16(dst) && aligned16(src)) {
    const int dh4 = dh / 4;
    for_each(q, "mha_split_heads", n / 4, [=](int64_t i) {
      const int64_t d = i % dh4;
      const int64_t h = (i / dh4) % heads;
      const int64_t t = (i / (static_cast<int64_t>(dh4) * heads)) % time;
      const int64_t batch = i / (static_cast<int64_t>(dh4) * heads * time);
      const int64_t o = ((batch * heads + h) * time + t) * dh4 + d;
      store4(dst + 4 * o, load4(src + 4 * i));
    });
  } else {
    for_each(q, "mha_split_heads", n, [=](int64_t i) {
      const int64_t d = i % dh;
      const int64_t h = (i / dh) % heads;
      const int64_t t = (i / (static_cast<int64_t>(dh) * heads)) % time;
      const int64_t batch = i / (static_cast<int64_t>(dh) * heads * time);
      dst[((batch * heads + h) * time + t) * dh + d] = src[i];
    });
  }
}

void launch_merge_heads(sycl::queue& q, float* dst, const float* src, int time, int heads, int dh, int64_t n) {
  if ((dh % 4) == 0 && aligned16(dst) && aligned16(src)) {
    const int dh4 = dh / 4;
    for_each(q, "mha_merge_heads", n / 4, [=](int64_t i) {
      const int64_t d = i % dh4;
      const int64_t h = (i / dh4) % heads;
      const int64_t t = (i / (static_cast<int64_t>(dh4) * heads)) % time;
      const int64_t batch = i / (static_cast<int64_t>(dh4) * heads * time);
      const int64_t s = ((batch * heads + h) * time + t) * dh4 + d;
      store4(dst + 4 * i, load4(src + 4 * s));
    });
  } else {
    for_each(q, "mha_merge_heads", n, [=](int64_t i) {
      const int64_t d = i % dh;
      const int64_t h = (i / dh) % heads;
      const int64_t t = (i / (static_cast<int64_t>(dh) * heads)) % time;
      const int64_t batch = i / (static_cast<int64_t>(dh) * heads * time);
      dst[i] = src[((batch * heads + h) * time + t) * dh + d];
    });
  }
}

void launch_write_kv(sycl::queue& q, float* dst, const float* src, int t, int t_max, int dh, int64_t n) {
  for_each(q, "mha_write_kv", n, [=](int64_t i) {
    const int d = static_cast<int>(i % dh);
    const int64_t head = i / dh;
    dst[(head * static_cast<int64_t>(t_max) + t) * dh + d] = src[i];
  });
}

void launch_rope(sycl::queue& q, float* query, float* key, int time, int dh, int pos0, int64_t n, bool inverse,
                 float base_theta) {
  const int64_t npairs = n / 2;
  const int inv = inverse ? 1 : 0;
  for_each(q, "mha_rope", npairs, [=](int64_t pair_index) {
    const int pairs = dh / 2;
    const int pair = static_cast<int>(pair_index % pairs);
    const int64_t vec = pair_index / pairs;
    const int t = static_cast<int>(vec % time);
    const int64_t base = vec * dh + static_cast<int64_t>(pair) * 2;
    const float theta = static_cast<float>(pos0 + t) *
                        sycl::exp(-sycl::log(base_theta) * (2.f * static_cast<float>(pair) / static_cast<float>(dh)));
    const float c = sycl::cos(theta);
    const float s = sycl::sin(theta);
    const float q0 = query[base];
    const float q1 = query[base + 1];
    const float k0 = key[base];
    const float k1 = key[base + 1];
    if (inv) {
      query[base] = q0 * c + q1 * s;
      query[base + 1] = -q0 * s + q1 * c;
      key[base] = k0 * c + k1 * s;
      key[base + 1] = -k0 * s + k1 * c;
    } else {
      query[base] = q0 * c - q1 * s;
      query[base + 1] = q0 * s + q1 * c;
      key[base] = k0 * c - k1 * s;
      key[base + 1] = k0 * s + k1 * c;
    }
  });
}

void launch_repeat_kv(sycl::queue& q, float* dst, const float* src, int time, int kv_heads, int group, int dh,
                      int64_t n) {
  if ((dh % 4) == 0 && aligned16(dst) && aligned16(src)) {
    const int dh4 = dh / 4;
    for_each(q, "mha_repeat_kv", n / 4, [=](int64_t i) {
      const int heads = kv_heads * group;
      const int64_t d = i % dh4;
      const int64_t t = (i / dh4) % time;
      const int64_t h = (i / (static_cast<int64_t>(dh4) * time)) % heads;
      const int64_t batch = i / (static_cast<int64_t>(dh4) * time * heads);
      const int64_t hkv = h / group;
      const int64_t s = ((batch * kv_heads + hkv) * time + t) * dh4 + d;
      store4(dst + 4 * i, load4(src + 4 * s));
    });
  } else {
    for_each(q, "mha_repeat_kv", n, [=](int64_t i) {
      const int heads = kv_heads * group;
      const int64_t d = i % dh;
      const int64_t t = (i / dh) % time;
      const int64_t h = (i / (static_cast<int64_t>(dh) * time)) % heads;
      const int64_t batch = i / (static_cast<int64_t>(dh) * time * heads);
      const int64_t hkv = h / group;
      dst[i] = src[((batch * kv_heads + hkv) * time + t) * dh + d];
    });
  }
}

void launch_sum_kv(sycl::queue& q, float* dst, const float* src, int time, int kv_heads, int group, int dh,
                   int64_t n) {
  for_each(q, "mha_sum_kv", n, [=](int64_t i) {
    const int heads = kv_heads * group;
    const int64_t d = i % dh;
    const int64_t t = (i / dh) % time;
    const int64_t hkv = (i / (static_cast<int64_t>(dh) * time)) % kv_heads;
    const int64_t batch = i / (static_cast<int64_t>(dh) * time * kv_heads);
    float sum = 0.f;
    for (int g = 0; g < group; ++g) {
      const int64_t h = hkv * group + g;
      sum += src[((batch * heads + h) * time + t) * dh + d];
    }
    dst[i] = sum;
  });
}

void softmax_rows(sycl::queue& q, float* out, const float* in, int rows, int cols, int causal) {
  for_groups(q, "mha_softmax", rows, [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
    const int row = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int query = row % cols;
    const int64_t base = static_cast<int64_t>(row) * cols;
    const float* row_in = in + base;
    float* row_out = out + base;
    float max_v = -std::numeric_limits<float>::infinity();
    for (int col = tid; col < cols; col += kThreads) {
      max_v = sycl::fmax(max_v, masked_score(row_in[col], col, query, causal));
    }
    max_v = block_max(item, max_v, scratch);
    float sum = 0.f;
    for (int col = tid; col < cols; col += kThreads) {
      sum += sycl::exp(masked_score(row_in[col], col, query, causal) - max_v);
    }
    sum = block_sum(item, sum, scratch);
    const float inv = 1.f / sum;
    for (int col = tid; col < cols; col += kThreads) {
      row_out[col] = sycl::exp(masked_score(row_in[col], col, query, causal) - max_v) * inv;
    }
  });
}

void softmax_backward(sycl::queue& q, float* ds, const float* probs, const float* dp, int rows, int cols,
                      int causal) {
  for_groups(q, "mha_softmax_backward", rows, [=](sycl::nd_item<1> item, sycl::local_accessor<float, 1> scratch) {
    const int row = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int query = row % cols;
    const int64_t base = static_cast<int64_t>(row) * cols;
    float dot = 0.f;
    for (int col = tid; col < cols; col += kThreads) dot += dp[base + col] * probs[base + col];
    dot = block_sum(item, dot, scratch);
    for (int col = tid; col < cols; col += kThreads) {
      float grad = probs[base + col] * (dp[base + col] - dot);
      if (causal && col > query) grad = 0.f;
      ds[base + col] = grad;
    }
  });
}

void gemm_scores(sycl::queue& q, const float* query, const float* key, float* scores, int time, int dh,
                 int batch, float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_batch(q, true, false, time, time, dh, scale, key, dh, stride_qk, query, dh, stride_qk, 0.f, scores, time,
             stride_s, batch);
}

void gemm_dq(sycl::queue& q, const float* key, const float* ds, float* dq, int time, int dh, int batch,
             float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_batch(q, false, false, dh, time, time, scale, key, dh, stride_qk, ds, time, stride_s, 0.f, dq, dh,
             stride_qk, batch);
}

void gemm_dk(sycl::queue& q, const float* query, const float* ds, float* dk, int time, int dh, int batch,
             float scale) {
  const int64_t stride_qk = static_cast<int64_t>(time) * dh;
  const int64_t stride_s = static_cast<int64_t>(time) * time;
  gemm_batch(q, false, true, dh, time, time, scale, query, dh, stride_qk, ds, time, stride_s, 0.f, dk, dh,
             stride_qk, batch);
}

void append_params(std::vector<NamedParam>& out, const Layer& layer, const char* prefix) {
  for (NamedParam param : layer.parameters()) {
    param.name = std::string(prefix) + param.name;
    out.push_back(param);
  }
}

Context* attention_context(Context& ctx, int d_model, int num_heads, int head_dim, int num_kv_heads, bool use_rope,
                           float rope_theta, float rms_eps) {
  if (d_model < 1 || num_heads < 1) throw Error("MultiHeadAttention size");
  const int dh = head_dim == 0 ? d_model / num_heads : head_dim;
  const int kv = num_kv_heads == 0 ? num_heads : num_kv_heads;
  if (head_dim == 0 && d_model % num_heads != 0) throw Error("MultiHeadAttention size");
  if (dh < 1 || kv < 1 || kv > num_heads || num_heads % kv != 0) throw Error("MultiHeadAttention size");
  if (dh > 0x7fffffff / num_heads || dh > 0x7fffffff / kv) throw Error("MultiHeadAttention size");
  if (use_rope && dh % 2 != 0) throw std::invalid_argument("use_rope");
  if (!(rope_theta > 0.f) || !std::isfinite(rope_theta)) throw std::invalid_argument("rope_theta");
  if (!(rms_eps > 0.f) || !std::isfinite(rms_eps)) throw std::invalid_argument("rms_eps");
  return &ctx;
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(Context& ctx, int d_model, int num_heads, bool causal, bool use_bias,
                                       bool use_rope, std::mt19937& rng, int head_dim, int num_kv_heads,
                                       bool qk_norm, float rope_theta, float rms_eps)
    : ctx_(attention_context(ctx, d_model, num_heads, head_dim, num_kv_heads, use_rope, rope_theta, rms_eps)),
      d_model_(d_model),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads == 0 ? num_heads : num_kv_heads),
      head_dim_(head_dim == 0 ? d_model / num_heads : head_dim),
      q_dim_(num_heads * (head_dim == 0 ? d_model / num_heads : head_dim)),
      kv_dim_((num_kv_heads == 0 ? num_heads : num_kv_heads) *
              (head_dim == 0 ? d_model / num_heads : head_dim)),
      causal_(causal),
      use_rope_(use_rope),
      qk_norm_(qk_norm),
      rope_theta_(rope_theta),
      q_(std::make_unique<Linear>(*ctx_, d_model, num_heads * (head_dim == 0 ? d_model / num_heads : head_dim),
                                  use_bias, rng)),
      k_(std::make_unique<Linear>(
          *ctx_, d_model,
          (num_kv_heads == 0 ? num_heads : num_kv_heads) * (head_dim == 0 ? d_model / num_heads : head_dim),
          use_bias, rng)),
      v_(std::make_unique<Linear>(
          *ctx_, d_model,
          (num_kv_heads == 0 ? num_heads : num_kv_heads) * (head_dim == 0 ? d_model / num_heads : head_dim),
          use_bias, rng)),
      out_(std::make_unique<Linear>(*ctx_, num_heads * (head_dim == 0 ? d_model / num_heads : head_dim), d_model,
                                    use_bias, rng)) {
  if (qk_norm) {
    q_norm_ = std::make_unique<RMSNorm>(*ctx_, head_dim_, rms_eps);
    k_norm_ = std::make_unique<RMSNorm>(*ctx_, head_dim_, rms_eps);
  }
}

std::vector<NamedParam> MultiHeadAttention::parameters() const {
  std::vector<NamedParam> params;
  append_params(params, *q_, "q.");
  append_params(params, *k_, "k.");
  append_params(params, *v_, "v.");
  append_params(params, *out_, "out.");
  if (q_norm_) append_params(params, *q_norm_, "q_norm.");
  if (k_norm_) append_params(params, *k_norm_, "k_norm.");
  return params;
}

void MultiHeadAttention::zero_grad() {
  q_->zero_grad();
  k_->zero_grad();
  v_->zero_grad();
  out_->zero_grad();
  if (q_norm_) q_norm_->zero_grad();
  if (k_norm_) k_norm_->zero_grad();
}

void MultiHeadAttention::lay_qkv(const Tensor& q, const Tensor& k, const Tensor& v, int time) {
  const int n = q.shape()[0];
  const int heads = num_heads_;
  const int kv = num_kv_heads_;
  const int group = heads / kv;
  const int dh = head_dim_;
  const int64_t q_n = static_cast<int64_t>(n) * time * q_dim_;
  const int64_t kv_n = static_cast<int64_t>(n) * time * kv_dim_;
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;
  sycl::queue& queue = queue_of(*ctx_);
  launch_split_heads(queue, q_heads_->data(), q.data(), time, heads, dh, q_n);
  if (q_norm_) q_heads_->copy_from(q_norm_->forward(*q_heads_));
  if (group == 1) {
    launch_split_heads(queue, k_heads_->data(), k.data(), time, kv, dh, kv_n);
    launch_split_heads(queue, v_heads_->data(), v.data(), time, kv, dh, kv_n);
    if (k_norm_) k_heads_->copy_from(k_norm_->forward(*k_heads_));
  } else {
    ensure_tensor(k_kv_, *ctx_, {n, kv, time, dh});
    ensure_tensor(v_kv_, *ctx_, {n, kv, time, dh});
    launch_split_heads(queue, k_kv_->data(), k.data(), time, kv, dh, kv_n);
    launch_split_heads(queue, v_kv_->data(), v.data(), time, kv, dh, kv_n);
    if (k_norm_) k_kv_->copy_from(k_norm_->forward(*k_kv_));
    launch_repeat_kv(queue, k_heads_->data(), k_kv_->data(), time, kv, group, dh, full_n);
    launch_repeat_kv(queue, v_heads_->data(), v_kv_->data(), time, kv, group, dh, full_n);
  }
}

const Tensor& MultiHeadAttention::forward(const Tensor& input) {
  const std::vector<int>& shape = input.shape();
  if (shape.size() != 3 || shape[2] != d_model_) throw Error("MultiHeadAttention input shape");
  cached_forward_ = false;
  const int n = shape[0];
  const int time = shape[1];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};
  const std::vector<int> merged_shape = {n, time, q_dim_};
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;

  ensure_tensor(q_heads_, *ctx_, heads_shape);
  ensure_tensor(k_heads_, *ctx_, heads_shape);
  ensure_tensor(v_heads_, *ctx_, heads_shape);
  ensure_tensor(scores_, *ctx_, score_shape);
  ensure_tensor(attn_, *ctx_, score_shape);
  ensure_tensor(context_, *ctx_, heads_shape);
  ensure_tensor(merged_, *ctx_, merged_shape);
  ensure_tensor(output_, *ctx_, shape);

  sycl::queue& queue = queue_of(*ctx_);
  lay_qkv(q_->forward(input), k_->forward(input), v_->forward(input), time);
  if (use_rope_) {
    launch_rope(queue, q_heads_->data(), k_heads_->data(), time, dh, 0, full_n, false, rope_theta_);
  }
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  gemm_scores(queue, q_heads_->data(), k_heads_->data(), scores_->data(), time, dh, batch, scale);
  softmax_rows(queue, attn_->data(), scores_->data(), batch * time, time, causal_ ? 1 : 0);
  gemm_dq(queue, v_heads_->data(), attn_->data(), context_->data(), time, dh, batch, 1.f);
  launch_merge_heads(queue, merged_->data(), context_->data(), time, heads, dh, full_n);
  output_->copy_from(out_->forward(*merged_));
  return *output_;
}

void MultiHeadAttention::reset_cache(Cache& cache, int batch, int t_max) const {
  if (batch < 1) throw std::invalid_argument("batch");
  if (t_max < 1) throw std::invalid_argument("t_max");
  cache.k = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_});
  cache.v = Tensor(*ctx_, {batch, num_heads_, t_max, head_dim_});
  cache.k.zero();
  cache.v.zero();
  cache.t = 0;
}

const Tensor& MultiHeadAttention::forward_cached(const Tensor& x_t, Cache& cache) {
  if (&x_t.context() != ctx_ || x_t.rank() != 3 || x_t.shape()[1] != 1 || x_t.shape()[2] != d_model_) {
    throw std::invalid_argument("x_t");
  }
  const int n = x_t.shape()[0];
  const int heads = num_heads_;
  const int dh = head_dim_;
  const int64_t full_n = static_cast<int64_t>(n) * heads * dh;
  const std::vector<int>& k_shape = cache.k.shape();
  if (&cache.k.context() != ctx_ || &cache.v.context() != ctx_ || cache.k.rank() != 4 ||
      cache.v.shape() != k_shape || k_shape[0] != n || k_shape[1] != heads || k_shape[3] != dh ||
      cache.t < 0 || cache.t >= k_shape[2]) {
    throw std::invalid_argument("cache");
  }
  cached_forward_ = true;

  const int t = cache.t;
  const int t_max = k_shape[2];
  const int tk = t + 1;
  constexpr int tq = 1;
  const int batch = n * heads;
  const std::vector<int> q_shape = {n, heads, tq, dh};
  const std::vector<int> score_shape = {n, heads, tq, tk};
  const std::vector<int> flat_shape = x_t.shape();
  const std::vector<int> merged_shape = {n, tq, q_dim_};

  ensure_tensor(q_heads_, *ctx_, q_shape);
  ensure_tensor(k_heads_, *ctx_, q_shape);
  ensure_tensor(v_heads_, *ctx_, q_shape);
  ensure_tensor(scores_, *ctx_, score_shape);
  ensure_tensor(attn_, *ctx_, score_shape);
  ensure_tensor(context_, *ctx_, q_shape);
  ensure_tensor(merged_, *ctx_, merged_shape);
  ensure_tensor(output_, *ctx_, flat_shape);

  sycl::queue& queue = queue_of(*ctx_);
  lay_qkv(q_->forward(x_t), k_->forward(x_t), v_->forward(x_t), tq);
  if (use_rope_) {
    launch_rope(queue, q_heads_->data(), k_heads_->data(), tq, dh, t, full_n, false, rope_theta_);
  }
  const int64_t kv_n = static_cast<int64_t>(batch) * dh;
  launch_write_kv(queue, cache.k.data(), k_heads_->data(), t, t_max, dh, kv_n);
  launch_write_kv(queue, cache.v.data(), v_heads_->data(), t, t_max, dh, kv_n);

  const float scale = 1.f / std::sqrt(static_cast<float>(dh));
  const int64_t stride_k = static_cast<int64_t>(t_max) * dh;
  const int64_t stride_q = static_cast<int64_t>(tq) * dh;
  const int64_t stride_s = static_cast<int64_t>(tq) * tk;
  gemm_batch(queue, true, false, tk, tq, dh, scale, cache.k.data(), dh, stride_k, q_heads_->data(), dh, stride_q,
             0.f, scores_->data(), tk, stride_s, batch);
  softmax_rows(queue, attn_->data(), scores_->data(), batch, tk, 0);
  gemm_batch(queue, false, false, dh, tq, tk, 1.f, cache.v.data(), dh, stride_k, attn_->data(), tk, stride_s, 0.f,
             context_->data(), dh, stride_q, batch);
  launch_merge_heads(queue, merged_->data(), context_->data(), tq, heads, dh, full_n);
  output_->copy_from(out_->forward(*merged_));
  cache.t = tk;
  return *output_;
}

const Tensor& MultiHeadAttention::backward(const Tensor& grad_output) {
  if (cached_forward_) throw std::invalid_argument("forward_cached");
  if (!output_ || !attn_ || !q_heads_ || !k_heads_ || !v_heads_ || grad_output.shape() != output_->shape()) {
    throw Error("MultiHeadAttention backward shape");
  }
  const int n = grad_output.shape()[0];
  const int time = grad_output.shape()[1];
  const int heads = num_heads_;
  const int kv = num_kv_heads_;
  const int group = heads / kv;
  const int dh = head_dim_;
  const int batch = n * heads;
  const std::vector<int> heads_shape = {n, heads, time, dh};
  const std::vector<int> score_shape = {n, heads, time, time};
  const std::vector<int>& flat_shape = grad_output.shape();
  const int64_t full_n = static_cast<int64_t>(n) * heads * time * dh;
  const int64_t q_n = static_cast<int64_t>(n) * time * q_dim_;
  const int64_t kv_n = static_cast<int64_t>(n) * time * kv_dim_;

  ensure_tensor(d_context_, *ctx_, heads_shape);
  ensure_tensor(d_attn_, *ctx_, score_shape);
  ensure_tensor(d_scores_, *ctx_, score_shape);
  ensure_tensor(d_q_, *ctx_, heads_shape);
  ensure_tensor(d_k_, *ctx_, heads_shape);
  ensure_tensor(d_v_, *ctx_, heads_shape);
  ensure_tensor(q_flat_grad_, *ctx_, {n, time, q_dim_});
  ensure_tensor(kv_flat_grad_, *ctx_, {n, time, kv_dim_});
  ensure_tensor(grad_pair_, *ctx_, flat_shape);
  ensure_tensor(grad_input_, *ctx_, flat_shape);

  sycl::queue& queue = queue_of(*ctx_);
  const int64_t hidden = grad_output.numel();
  const float scale = 1.f / std::sqrt(static_cast<float>(dh));

  const Tensor& d_merged = out_->backward(grad_output);
  launch_split_heads(queue, d_context_->data(), d_merged.data(), time, heads, dh, q_n);
  gemm_scores(queue, d_context_->data(), v_heads_->data(), d_attn_->data(), time, dh, batch, 1.f);
  gemm_dk(queue, d_context_->data(), attn_->data(), d_v_->data(), time, dh, batch, 1.f);
  softmax_backward(queue, d_scores_->data(), attn_->data(), d_attn_->data(), batch * time, time, causal_ ? 1 : 0);
  gemm_dq(queue, k_heads_->data(), d_scores_->data(), d_q_->data(), time, dh, batch, scale);
  gemm_dk(queue, q_heads_->data(), d_scores_->data(), d_k_->data(), time, dh, batch, scale);
  if (use_rope_) {
    launch_rope(queue, d_q_->data(), d_k_->data(), time, dh, 0, full_n, true, rope_theta_);
  }

  const float* dq = d_q_->data();
  if (q_norm_) dq = q_norm_->backward(*d_q_).data();
  launch_merge_heads(queue, q_flat_grad_->data(), dq, time, heads, dh, q_n);
  const Tensor& gq = q_->backward(*q_flat_grad_);

  const float* dk = d_k_->data();
  if (group > 1) {
    ensure_tensor(k_kv_, *ctx_, {n, kv, time, dh});
    const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
    launch_sum_kv(queue, k_kv_->data(), d_k_->data(), time, kv, group, dh, kv_heads_n);
    dk = k_kv_->data();
    if (k_norm_) dk = k_norm_->backward(*k_kv_).data();
  } else if (k_norm_) {
    dk = k_norm_->backward(*d_k_).data();
  }
  launch_merge_heads(queue, kv_flat_grad_->data(), dk, time, kv, dh, kv_n);
  const Tensor& gk = k_->backward(*kv_flat_grad_);

  const float* dv = d_v_->data();
  if (group > 1) {
    ensure_tensor(v_kv_, *ctx_, {n, kv, time, dh});
    const int64_t kv_heads_n = static_cast<int64_t>(n) * kv * time * dh;
    launch_sum_kv(queue, v_kv_->data(), d_v_->data(), time, kv, group, dh, kv_heads_n);
    dv = v_kv_->data();
  }
  launch_merge_heads(queue, kv_flat_grad_->data(), dv, time, kv, dh, kv_n);
  const Tensor& gv = v_->backward(*kv_flat_grad_);

  launch_add2(queue, grad_pair_->data(), gq.data(), gk.data(), hidden);
  launch_add2(queue, grad_input_->data(), grad_pair_->data(), gv.data(), hidden);
  return *grad_input_;
}

}  // namespace nn
