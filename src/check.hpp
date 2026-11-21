#pragma once

#include "nn/error.hpp"

#include <string>

namespace nn {

// Backend failures use the contract form `call: status`.
void backend_error(const char* call, const std::string& status);

}  // namespace nn
