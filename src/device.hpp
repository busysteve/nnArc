#pragma once

#include "check.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <limits>
#include <string>

namespace nn {

constexpr int kThreads = 256;
// One float per subgroup. A 256-wide group and a subgroup of at least 8 needs at most 32 slots.
constexpr int kScratch = 32;

inline bool aligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0;
}

using f4 = sycl::vec<float, 4>;

inline f4 load4(const float* p) { return *reinterpret_cast<const f4*>(p); }

inline void store4(float* p, f4 v) { *reinterpret_cast<f4*>(p) = v; }

inline f4 make4(float a, float b, float c, float d) { return f4(a, b, c, d); }

template <typename Fn>
inline void submit(sycl::queue& q, const char* call, Fn&& fn) {
  try {
    fn(q);
  } catch (const sycl::exception& ex) {
    backend_error(call, ex.what());
  }
}

// One work-item per element. n is the exact range.
template <typename Fn>
inline void for_each(sycl::queue& q, const char* call, int64_t n, Fn fn) {
  if (n <= 0) return;
  submit(q, call, [&](sycl::queue& queue) {
    queue.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> id) {
      fn(static_cast<int64_t>(id[0]));
    });
  });
}

// `groups` work-groups of kThreads. fn(item, scratch) may call block_sum / block_max.
template <typename Fn>
inline void for_groups(sycl::queue& q, const char* call, int groups, Fn fn) {
  if (groups <= 0) return;
  submit(q, call, [&](sycl::queue& queue) {
    queue.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> scratch(sycl::range<1>(static_cast<size_t>(kScratch)), h);
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * kThreads),
                                       sycl::range<1>(kThreads)),
                     [=](sycl::nd_item<1> item) { fn(item, scratch); });
    });
  });
}

inline float subgroup_sum(sycl::sub_group sg, float v) {
  return sycl::reduce_over_group(sg, v, sycl::plus<float>());
}

inline float subgroup_max(sycl::sub_group sg, float v) {
  return sycl::reduce_over_group(sg, v, sycl::maximum<float>());
}

// scratch holds one float per subgroup. The trailing barrier lets the next reduction reuse it.
inline float block_sum(sycl::nd_item<1> item, float v, sycl::local_accessor<float, 1> scratch) {
  sycl::sub_group sg = item.get_sub_group();
  v = subgroup_sum(sg, v);
  const int lane = static_cast<int>(sg.get_local_linear_id());
  const int wid = static_cast<int>(sg.get_group_linear_id());
  const int nsub = static_cast<int>(sg.get_group_linear_range());
  if (lane == 0) scratch[wid] = v;
  item.barrier(sycl::access::fence_space::local_space);
  v = (static_cast<int>(item.get_local_id(0)) < nsub) ? scratch[static_cast<size_t>(item.get_local_id(0))]
                                                      : 0.f;
  if (wid == 0) v = subgroup_sum(sg, v);
  if (item.get_local_id(0) == 0) scratch[0] = v;
  item.barrier(sycl::access::fence_space::local_space);
  v = scratch[0];
  item.barrier(sycl::access::fence_space::local_space);
  return v;
}

inline float block_max(sycl::nd_item<1> item, float v, sycl::local_accessor<float, 1> scratch) {
  sycl::sub_group sg = item.get_sub_group();
  v = subgroup_max(sg, v);
  const int lane = static_cast<int>(sg.get_local_linear_id());
  const int wid = static_cast<int>(sg.get_group_linear_id());
  const int nsub = static_cast<int>(sg.get_group_linear_range());
  if (lane == 0) scratch[wid] = v;
  item.barrier(sycl::access::fence_space::local_space);
  v = (static_cast<int>(item.get_local_id(0)) < nsub)
          ? scratch[static_cast<size_t>(item.get_local_id(0))]
          : -std::numeric_limits<float>::infinity();
  if (wid == 0) v = subgroup_max(sg, v);
  if (item.get_local_id(0) == 0) scratch[0] = v;
  item.barrier(sycl::access::fence_space::local_space);
  v = scratch[0];
  item.barrier(sycl::access::fence_space::local_space);
  return v;
}

inline void* device_alloc(sycl::queue& q, size_t bytes, const char* call) {
  if (bytes == 0) return nullptr;
  void* p = nullptr;
  try {
    p = sycl::malloc_device(bytes, q);
  } catch (const sycl::exception& ex) {
    backend_error(call, ex.what());
  }
  if (!p) backend_error(call, "null");
  return p;
}

template <typename T>
inline T* device_alloc(sycl::queue& q, size_t count, const char* call) {
  return static_cast<T*>(device_alloc(q, count * sizeof(T), call));
}

inline void device_free(sycl::queue& q, void* p) noexcept {
  if (!p) return;
  try {
    sycl::free(p, q);
  } catch (...) {
  }
}

inline void memcpy_async(sycl::queue& q, void* dst, const void* src, size_t bytes, const char* call) {
  try {
    q.memcpy(dst, src, bytes);
  } catch (const sycl::exception& ex) {
    backend_error(call, ex.what());
  }
}

inline void memset_async(sycl::queue& q, void* dst, int value, size_t bytes, const char* call) {
  try {
    q.memset(dst, value, bytes);
  } catch (const sycl::exception& ex) {
    backend_error(call, ex.what());
  }
}

template <typename T>
inline void atomic_add(T* addr, T value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*addr);
  ref.fetch_add(value);
}

inline void atomic_exchange(int* addr, int value) {
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*addr);
  ref.exchange(value);
}

}  // namespace nn
