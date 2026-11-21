cmake_minimum_required(VERSION 3.24)

# Standalone builds use Intel DPC++ when icpx is installed. A parent project
# (llm ./build.sh arc) sets the compiler before this file is added.
if(NOT DEFINED CMAKE_CXX_COMPILER)
  foreach(candidate
      /opt/intel/oneapi/compiler/latest/bin/icpx
      /opt/intel/oneapi/compiler/2026.0/bin/icpx)
    if(EXISTS ${candidate})
      set(CMAKE_CXX_COMPILER ${candidate} CACHE FILEPATH "SYCL compiler" FORCE)
      break()
    endif()
  endforeach()
endif()

project(nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED MKL_LINK)
  set(MKL_LINK dynamic)
endif()
if(NOT DEFINED MKL_INTERFACE)
  set(MKL_INTERFACE lp64)
endif()
if(NOT DEFINED MKL_THREADING)
  set(MKL_THREADING sequential)
endif()
if(NOT DEFINED MKL_SYCL_LINK)
  set(MKL_SYCL_LINK dynamic)
endif()
if(NOT DEFINED MKL_SYCL_INTERFACE_FULL)
  set(MKL_SYCL_INTERFACE_FULL intel_lp64)
endif()
if(NOT DEFINED MKL_SYCL_THREADING)
  set(MKL_SYCL_THREADING sequential)
endif()
set(ENABLE_SYCL_COMPILER ON CACHE STRING "oneMKL treats the C++ compiler as a SYCL compiler")

set(NN_MKL_ROOT "/opt/intel/oneapi/mkl/latest")
if(DEFINED ENV{MKLROOT} AND NOT "$ENV{MKLROOT}" STREQUAL "")
  set(NN_MKL_ROOT "$ENV{MKLROOT}")
endif()
find_package(MKL CONFIG REQUIRED HINTS ${NN_MKL_ROOT} /opt/intel/oneapi/mkl/latest)
if(NOT TARGET MKL::MKL_SYCL::BLAS)
  message(FATAL_ERROR
    "oneMKL SYCL BLAS was not found. Install intel-oneapi-mkl-sycl and intel-oneapi-dpcpp-cpp.")
endif()

add_library(nn STATIC
  src/context.cpp
  src/error.cpp
  src/tensor.cpp
  src/index_tensor.cpp
  src/init.cpp
  src/config.cpp
  src/linear.cpp
  src/gelu.cpp
  src/swiglu.cpp
  src/dropout.cpp
  src/transformer_block.cpp
  src/transformer.cpp
  src/rnn.cpp
  src/last_step.cpp
  src/flatten.cpp
  src/network.cpp
  src/optim.cpp
  src/checkpoint.cpp
  src/builders.cpp
  src/residual_block.cpp
  src/elementwise.cpp
  src/embedding.cpp
  src/layer_norm.cpp
  src/rms_norm.cpp
  src/multi_head_attention.cpp
  src/sgd.cpp
  src/adamw.cpp
  src/loss.cpp
  src/relu.cpp
  src/conv2d.cpp
  src/max_pool2d.cpp
  src/batch_norm2d.cpp
  src/global_avg_pool2d.cpp
)

target_include_directories(nn
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_options(nn PUBLIC -fsycl)
target_link_options(nn PUBLIC -fsycl)
target_link_libraries(nn PUBLIC MKL::MKL_SYCL::BLAS)
target_compile_options(nn PRIVATE -Wno-deprecated-declarations -Wno-unused-result)

set_target_properties(nn PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(nn_tests
  tests/test_main.cpp
  tests/test_shapes.cpp
  tests/test_gradients.cpp
  tests/test_checkpoint.cpp
  tests/test_mnist.cpp
  tests/mnist.cpp
)

target_link_libraries(nn_tests PRIVATE nn)
target_include_directories(nn_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(nn_tests PRIVATE -Wno-deprecated-declarations)

enable_testing()
add_test(NAME shapes COMMAND nn_tests shapes)
add_test(NAME grads COMMAND nn_tests grads)
add_test(NAME ckpt COMMAND nn_tests ckpt)
add_test(NAME mnist COMMAND nn_tests mnist)
set_tests_properties(shapes grads ckpt PROPERTIES TIMEOUT 900)
set_tests_properties(mnist PROPERTIES TIMEOUT 7200)
