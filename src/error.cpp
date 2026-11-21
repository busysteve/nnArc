#include "check.hpp"

namespace nn {

void backend_error(const char* call, const std::string& status) {
  throw Error(std::string(call) + ": " + status);
}

}  // namespace nn
