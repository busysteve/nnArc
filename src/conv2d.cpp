#include "nn/layers/conv2d.hpp"

#include "context_impl.hpp"
#include "device.hpp"
#include "ensure.hpp"
#include "gemm.hpp"
#include "nn/config.hpp"
#include "nn/error.hpp"
#include "nn/init.hpp"

#include <algorithm>
#include <string>

namespace nn {

struct Conv2d::Impl {};

namespace {

constexpr size_t kMaxWorkspace = 64ull * 1024ull * 1024ull;

}  // namespace

Conv2d::Conv2d(Context& ctx, int in_channels, int out_channels, int kernel, int stride, int pad, bool use_bias,
               std::mt19937& rng)
    : ctx_(&ctx),
      in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_(kernel),
      stride_(stride),
      pad_(pad),
      use_bias_(use_bias),
      weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      grad_weight_(ctx, {out_channels, in_channels, kernel, kernel}),
      impl_(std::make_unique<Impl>()) {
  if (in_channels < 1 || out_channels < 1) throw Error("Conv2d channels");
  conv_out_size(1 + kernel, kernel, stride, pad);
  kaiming_normal(weight_, in_channels * kernel * kernel, rng);
  grad_weight_.zero();
  if (use_bias_) {
    bias_.emplace(ctx, std::vector<int>{out_channels});
    grad_bias_.emplace(ctx, std::vector<int>{out_channels});
    bias_->zero();
    grad_bias_->zero();
  }
}

Conv2d::~Conv2d() {
  if (workspace_ && ctx_) {
    try {
      queue_of(*ctx_).wait();
    } catch (...) {
    }
    device_free(queue_of(*ctx_), workspace_);
  }
}

std::vector<NamedParam> Conv2d::parameters() const {
  std::vector<NamedParam> params;
  params.push_back(NamedParam{"weight", const_cast<Tensor*>(&weight_), const_cast<Tensor*>(&grad_weight_)});
  if (bias_) {
    params.push_back(NamedParam{"bias", const_cast<Tensor*>(&*bias_), const_cast<Tensor*>(&*grad_bias_)});
  }
  return params;
}

void Conv2d::zero_grad() {
  grad_weight_.zero();
  if (grad_bias_) grad_bias_->zero();
}

void Conv2d::prepare(int n, int h, int w) {
  const int ho = conv_out_size(h, kernel_, stride_, pad_);
  const int wo = conv_out_size(w, kernel_, stride_, pad_);
  if (ho > 0x7fffffff / wo) throw Error("Conv2d output dim mismatch");
  const int spatial = ho * wo;
  const int k_col = in_channels_ * kernel_ * kernel_;
  if (k_col <= 0 || spatial <= 0) throw Error("Conv2d output dim mismatch");
  const bool direct = kernel_ == 1 && stride_ == 1 && pad_ == 0;
  const size_t col_per = direct ? 0 : static_cast<size_t>(k_col) * static_cast<size_t>(spatial) * sizeof(float);
  const size_t dw_per = static_cast<size_t>(out_channels_) * static_cast<size_t>(k_col) * sizeof(float);
  const size_t per = col_per + dw_per;
  if (per > kMaxWorkspace) {
    throw Error("conv workspace " + std::to_string(per) + " exceeds 64 MiB budget");
  }
  int chunk = n;
  if (per > 0) chunk = static_cast<int>(std::min<size_t>(n, kMaxWorkspace / per));
  if (chunk < 1) chunk = 1;
  const size_t need = per * static_cast<size_t>(chunk);
  if (need > workspace_bytes_) {
    if (workspace_) {
      queue_of(*ctx_).wait();
      device_free(queue_of(*ctx_), workspace_);
      workspace_ = nullptr;
      workspace_bytes_ = 0;
    }
    if (need > 0) {
      workspace_ = device_alloc(queue_of(*ctx_), need, "sycl::malloc_device");
      workspace_bytes_ = need;
    }
  }
  ensure_tensor(output_, *ctx_, {n, out_channels_, ho, wo});
  ensure_tensor(grad_input_, *ctx_, {n, in_channels_, h, w});
}

const Tensor& Conv2d::forward(const Tensor& input) {
  if (input.rank() != 4 || input.shape()[1] != in_channels_) throw Error("Conv2d input shape");
  const int n = input.shape()[0];
  const int h = input.shape()[2];
  const int w = input.shape()[3];
  prepare(n, h, w);
  ensure_tensor(cache_x_, *ctx_, input.shape());
  cache_x_->copy_from(input);

  const int ho = output_->shape()[2];
  const int wo = output_->shape()[3];
  const int spatial = ho * wo;
  const int k_col = in_channels_ * kernel_ * kernel_;
  const bool direct = kernel_ == 1 && stride_ == 1 && pad_ == 0;
  const size_t col_per = direct ? 0 : static_cast<size_t>(k_col) * static_cast<size_t>(spatial) * sizeof(float);
  const size_t dw_per = static_cast<size_t>(out_channels_) * static_cast<size_t>(k_col) * sizeof(float);
  const size_t per = col_per + dw_per;
  int chunk = n;
  if (per > 0) chunk = static_cast<int>(std::min<size_t>(n, kMaxWorkspace / per));
  if (chunk < 1) chunk = 1;

  sycl::queue& q = queue_of(*ctx_);
  float* col = static_cast<float*>(workspace_);
  const int ic = in_channels_;
  const int oc = out_channels_;
  const int k = kernel_;
  const int stride = stride_;
  const int pad = pad_;
  const float* x = cache_x_->data();
  const float* weight = weight_.data();
  float* y = output_->data();
  for (int n0 = 0; n0 < n; n0 += chunk) {
    const int nb = std::min(chunk, n - n0);
    const float* a = x + static_cast<int64_t>(n0) * k_col * spatial;
    if (!direct) {
      const int64_t total = static_cast<int64_t>(nb) * k_col * spatial;
      for_each(q, "im2col", total, [=](int64_t idx) {
        const int s = static_cast<int>(idx % spatial);
        const int64_t tmp = idx / spatial;
        const int kc = static_cast<int>(tmp % k_col);
        const int local_n = static_cast<int>(tmp / k_col);
        const int xw = s % wo;
        const int xh = s / wo;
        const int kx = kc % k;
        const int ky = (kc / k) % k;
        const int channel = kc / (k * k);
        const int ih = xh * stride - pad + ky;
        const int iw = xw * stride - pad + kx;
        float value = 0.f;
        if (static_cast<unsigned>(ih) < static_cast<unsigned>(h) &&
            static_cast<unsigned>(iw) < static_cast<unsigned>(w)) {
          const int image = n0 + local_n;
          value = x[(((static_cast<int64_t>(image) * ic + channel) * h + ih) * w) + iw];
        }
        col[idx] = value;
      });
      a = col;
    }
    gemm_batch(q, false, false, spatial, oc, k_col, 1.f, a, spatial, static_cast<int64_t>(k_col) * spatial,
               weight, k_col, 0, 0.f, y + static_cast<int64_t>(n0) * oc * spatial, spatial,
               static_cast<int64_t>(oc) * spatial, nb);
  }
  if (bias_) {
    const float* bias = bias_->data();
    for_each(q, "conv_bias", output_->numel(), [=](int64_t idx) {
      const int c = static_cast<int>((idx / spatial) % oc);
      y[idx] += bias[c];
    });
  }
  return *output_;
}

const Tensor& Conv2d::backward(const Tensor& grad_output) {
  if (!cache_x_ || !output_ || grad_output.shape() != output_->shape()) {
    throw Error("Conv2d backward shape");
  }
  const int n = cache_x_->shape()[0];
  const int h = cache_x_->shape()[2];
  const int w = cache_x_->shape()[3];
  const int ho = output_->shape()[2];
  const int wo = output_->shape()[3];
  const int spatial = ho * wo;
  const int k_col = in_channels_ * kernel_ * kernel_;
  const bool direct = kernel_ == 1 && stride_ == 1 && pad_ == 0;
  const size_t col_per = direct ? 0 : static_cast<size_t>(k_col) * static_cast<size_t>(spatial) * sizeof(float);
  const size_t dw_per = static_cast<size_t>(out_channels_) * static_cast<size_t>(k_col) * sizeof(float);
  const size_t per = col_per + dw_per;
  int chunk = n;
  if (per > 0) chunk = static_cast<int>(std::min<size_t>(n, kMaxWorkspace / per));
  if (chunk < 1) chunk = 1;

  sycl::queue& q = queue_of(*ctx_);
  float* col = static_cast<float*>(workspace_);
  const int dw_count = out_channels_ * k_col;
  const int ic = in_channels_;
  const int oc = out_channels_;
  const int k = kernel_;
  const int stride = stride_;
  const int pad = pad_;
  const float* x = cache_x_->data();
  const float* weight = weight_.data();
  const float* gy = grad_output.data();
  float* gw = grad_weight_.data();
  float* dx_all = grad_input_->data();
  for (int n0 = 0; n0 < n; n0 += chunk) {
    const int nb = std::min(chunk, n - n0);
    float* partial = col + (direct ? 0 : static_cast<int64_t>(nb) * k_col * spatial);
    const float* a = x + static_cast<int64_t>(n0) * k_col * spatial;
    if (!direct) {
      const int64_t total = static_cast<int64_t>(nb) * k_col * spatial;
      for_each(q, "im2col", total, [=](int64_t idx) {
        const int s = static_cast<int>(idx % spatial);
        const int64_t tmp = idx / spatial;
        const int kc = static_cast<int>(tmp % k_col);
        const int local_n = static_cast<int>(tmp / k_col);
        const int xw = s % wo;
        const int xh = s / wo;
        const int kx = kc % k;
        const int ky = (kc / k) % k;
        const int channel = kc / (k * k);
        const int ih = xh * stride - pad + ky;
        const int iw = xw * stride - pad + kx;
        float value = 0.f;
        if (static_cast<unsigned>(ih) < static_cast<unsigned>(h) &&
            static_cast<unsigned>(iw) < static_cast<unsigned>(w)) {
          const int image = n0 + local_n;
          value = x[(((static_cast<int64_t>(image) * ic + channel) * h + ih) * w) + iw];
        }
        col[idx] = value;
      });
      a = col;
    }
    gemm_batch(q, true, false, k_col, oc, spatial, 1.f, a, spatial, static_cast<int64_t>(k_col) * spatial,
               gy + static_cast<int64_t>(n0) * oc * spatial, spatial, static_cast<int64_t>(oc) * spatial, 0.f,
               partial, k_col, static_cast<int64_t>(dw_count), nb);
    for_each(q, "conv_dw", dw_count, [=](int64_t idx) {
      float sum = 0.f;
      for (int i = 0; i < nb; ++i) sum += partial[static_cast<int64_t>(i) * dw_count + idx];
      gw[idx] += sum;
    });

    float* dx = direct ? dx_all + static_cast<int64_t>(n0) * k_col * spatial : col;
    gemm_batch(q, false, true, spatial, k_col, oc, 1.f, gy + static_cast<int64_t>(n0) * oc * spatial, spatial,
               static_cast<int64_t>(oc) * spatial, weight, k_col, 0, 0.f, dx, spatial,
               static_cast<int64_t>(k_col) * spatial, nb);
    if (!direct) {
      const int64_t total = static_cast<int64_t>(nb) * ic * h * w;
      for_each(q, "col2im", total, [=](int64_t idx) {
        const int iw = static_cast<int>(idx % w);
        const int tmp1 = static_cast<int>(idx / w);
        const int ih = tmp1 % h;
        const int tmp2 = tmp1 / h;
        const int channel = tmp2 % ic;
        const int local_n = tmp2 / ic;
        float sum = 0.f;
        for (int ky = 0; ky < k; ++ky) {
          const int h_nom = ih + pad - ky;
          if (h_nom < 0 || (h_nom % stride) != 0) continue;
          const int xh = h_nom / stride;
          if (xh >= ho) continue;
          for (int kx = 0; kx < k; ++kx) {
            const int w_nom = iw + pad - kx;
            if (w_nom < 0 || (w_nom % stride) != 0) continue;
            const int xw = w_nom / stride;
            if (xw >= wo) continue;
            const int kc = (channel * k + ky) * k + kx;
            const int s = xh * wo + xw;
            sum += col[((static_cast<int64_t>(local_n) * k_col + kc) * spatial) + s];
          }
        }
        const int image = n0 + local_n;
        dx_all[(((static_cast<int64_t>(image) * ic + channel) * h + ih) * w) + iw] = sum;
      });
    }
  }
  if (grad_bias_) {
    float* gb = grad_bias_->data();
    for_each(q, "conv_bias_grad", oc, [=](int64_t c) {
      float sum = 0.f;
      for (int i = 0; i < n; ++i) {
        const float* plane = gy + (static_cast<int64_t>(i) * oc + c) * spatial;
        if ((spatial % 4) == 0) {
          for (int s = 0; s < spatial; s += 4) {
            const f4 t = load4(plane + s);
            sum += t[0] + t[1] + t[2] + t[3];
          }
        } else {
          for (int s = 0; s < spatial; ++s) sum += plane[s];
        }
      }
      gb[c] += sum;
    });
  }
  return *grad_input_;
}

}  // namespace nn
