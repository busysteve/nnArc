#pragma once

#include "nn/context.hpp"

#include <sycl/sycl.hpp>

#include <memory>
#include <mutex>
#include <string>

namespace nn {

struct Context::Impl {
  sycl::device device;
  int index = 0;
  std::mutex error_mu;
  std::string async_error;
  std::unique_ptr<sycl::queue> queue;
};

struct ContextAccess {
  static Context::Impl& get(Context& ctx) { return *ctx.impl_; }
  static const Context::Impl& get(const Context& ctx) { return *ctx.impl_; }
};

inline sycl::queue& queue_of(Context& ctx) { return *ContextAccess::get(ctx).queue; }
inline sycl::queue& queue_of(const Context& ctx) { return *ContextAccess::get(ctx).queue; }
inline sycl::queue& stream_of(Context& ctx) { return queue_of(ctx); }
inline sycl::queue& stream_of(const Context& ctx) { return queue_of(ctx); }

}  // namespace nn
