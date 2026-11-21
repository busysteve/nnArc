#include "context_impl.hpp"
#include "check.hpp"
#include "device.hpp"

#include <cctype>
#include <mutex>
#include <string>
#include <vector>

namespace nn {
namespace {

std::string trimmed(std::string text) {
  auto space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!text.empty() && space(static_cast<unsigned char>(text.front()))) text.erase(text.begin());
  while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
  return text;
}

bool is_auto(const std::string& text) {
  if (text.size() != 4) return false;
  return std::tolower(static_cast<unsigned char>(text[0])) == 'a' &&
         std::tolower(static_cast<unsigned char>(text[1])) == 'u' &&
         std::tolower(static_cast<unsigned char>(text[2])) == 't' &&
         std::tolower(static_cast<unsigned char>(text[3])) == 'o';
}

bool parse_index(const std::string& text, int& index) {
  if (text.empty()) return false;
  int value = 0;
  for (char ch : text) {
    if (ch < '0' || ch > '9') return false;
    const int digit = ch - '0';
    if (value > (2147483647 - digit) / 10) return false;
    value = value * 10 + digit;
  }
  index = value;
  return true;
}

bool intel_gpu(const sycl::device& dev) {
  if (!dev.is_gpu()) return false;
  std::string vendor = dev.get_info<sycl::info::device::vendor>();
  for (char& ch : vendor) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return vendor.find("intel") != std::string::npos;
}

std::vector<sycl::device> intel_gpus() {
  std::vector<sycl::device> found;
  for (const sycl::device& dev : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (intel_gpu(dev)) found.push_back(dev);
  }
  return found;
}

bool subgroup_fits(const sycl::device& dev) {
  if (dev.get_info<sycl::info::device::max_work_group_size>() < static_cast<size_t>(kThreads)) return false;
  for (size_t size : dev.get_info<sycl::info::device::sub_group_sizes>()) {
    if (size >= 8 && size <= static_cast<size_t>(kScratch) && (static_cast<size_t>(kThreads) % size) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

void Context::open_handles(int index) {
  const std::vector<sycl::device> gpus = intel_gpus();
  if (index < 0 || index >= static_cast<int>(gpus.size())) throw Error("device");
  const sycl::device dev = gpus[static_cast<size_t>(index)];
  if (!subgroup_fits(dev)) throw Error("device");
  impl_->device = dev;
  impl_->index = index;
  Impl* impl = impl_.get();
  try {
    impl_->queue = std::make_unique<sycl::queue>(
        dev,
        [impl](sycl::exception_list errors) {
          for (std::exception_ptr error : errors) {
            try {
              if (error) std::rethrow_exception(error);
            } catch (const std::exception& ex) {
              std::lock_guard<std::mutex> lock(impl->error_mu);
              if (impl->async_error.empty()) impl->async_error = ex.what();
            }
          }
        },
        sycl::property_list{sycl::property::queue::in_order{}});
  } catch (const sycl::exception& ex) {
    backend_error("sycl::queue", ex.what());
  }
}

int device_count() { return static_cast<int>(intel_gpus().size()); }

std::string device_name(int device) {
  const std::vector<sycl::device> gpus = intel_gpus();
  if (device < 0 || device >= static_cast<int>(gpus.size())) throw Error("device");
  return gpus[static_cast<size_t>(device)].get_info<sycl::info::device::name>();
}

std::vector<DeviceInfo> detect_devices() {
  const std::vector<sycl::device> gpus = intel_gpus();
  std::vector<DeviceInfo> found;
  found.reserve(gpus.size());
  for (int device = 0; device < static_cast<int>(gpus.size()); ++device) {
    found.push_back(DeviceInfo{device, gpus[static_cast<size_t>(device)].get_info<sycl::info::device::name>()});
  }
  return found;
}

int select_device(const std::string& choice) {
  const std::vector<DeviceInfo> found = detect_devices();
  if (found.empty()) throw Error("device");
  const std::string text = trimmed(choice);
  if (is_auto(text)) return found.front().index;
  int index = 0;
  if (!parse_index(text, index) || index >= static_cast<int>(found.size())) throw Error("device");
  return index;
}

Context::Context() : impl_(std::make_unique<Impl>()) {
  if (device_count() < 1) throw Error("device");
  open_handles(0);
}

Context::Context(int device) : impl_(std::make_unique<Impl>()) {
  if (device < 0 || device >= device_count()) throw Error("device");
  open_handles(device);
}

Context::Context(const std::string& choice) : impl_(std::make_unique<Impl>()) {
  open_handles(select_device(choice));
}

Context::~Context() = default;

void Context::synchronize() const {
  try {
    impl_->queue->wait();
  } catch (const sycl::exception& ex) {
    backend_error("sycl::queue", ex.what());
  }
  std::string err;
  {
    std::lock_guard<std::mutex> lock(impl_->error_mu);
    err.swap(impl_->async_error);
  }
  if (!err.empty()) backend_error("sycl::queue", err);
}

}  // namespace nn
