#pragma once

#include "device.hpp"

#include <oneapi/mkl/blas.hpp>

#include <cstdint>
#include <exception>

namespace nn {
namespace detail {

inline oneapi::mkl::transpose to_transpose(bool trans) {
  return trans ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
}

inline void gemm_error(const char* call, const std::exception& ex) { backend_error(call, ex.what()); }

}  // namespace detail

// Column-major GEMM. Row-major y = x W^T is C = W x^T with these transposes.
inline void gemm(sycl::queue& q, bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k, float alpha,
                 const float* a, int64_t lda, const float* b, int64_t ldb, float beta, float* c,
                 int64_t ldc) {
  try {
    oneapi::mkl::blas::column_major::gemm(q, detail::to_transpose(trans_a), detail::to_transpose(trans_b), m, n, k,
                                          alpha, a, lda, b, ldb, beta, c, ldc);
  } catch (const std::exception& ex) {
    detail::gemm_error("gemm", ex);
  }
}

// A stride of 0 repeats that matrix. oneMKL's strided batch does not promise that, so it is a loop.
inline void gemm_batch(sycl::queue& q, bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k,
                       float alpha, const float* a, int64_t lda, int64_t stride_a, const float* b,
                       int64_t ldb, int64_t stride_b, float beta, float* c, int64_t ldc, int64_t stride_c,
                       int64_t batch) {
  if (batch <= 0) return;
  if (stride_a == 0 || stride_b == 0) {
    for (int64_t i = 0; i < batch; ++i) {
      gemm(q, trans_a, trans_b, m, n, k, alpha, a + (stride_a ? i * stride_a : 0), lda,
           b + (stride_b ? i * stride_b : 0), ldb, beta, c + i * stride_c, ldc);
    }
    return;
  }
  try {
    oneapi::mkl::blas::column_major::gemm_batch(q, detail::to_transpose(trans_a), detail::to_transpose(trans_b), m,
                                                n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                                stride_c, batch);
  } catch (const std::exception& ex) {
    detail::gemm_error("gemm_batch", ex);
  }
}

}  // namespace nn
