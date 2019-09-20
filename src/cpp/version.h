// This library contains version check functions to ensure compatibility.
#ifndef EDGETPU_CPP_VERSION_H_
#define EDGETPU_CPP_VERSION_H_
#include <string>

#include "absl/base/attributes.h"

namespace coral {
// Version of edgetpu_cpp_wrapper library.
// The version is a string with format 'API(version) + TF(version)'.
// API version corresponds to this C++ source code while TF version is the
// commit number of tensorflow submodule when compiling `edgetpu-native` repo.
ABSL_CONST_INIT extern const char kEdgeTpuCppWrapperVersion[];

// Supported runtime version. This is the runtime version required by this
// edgetpu_cpp_wrapper library.
ABSL_CONST_INIT extern const char kSupportedRuntimeVersion[];

// Gets version of the EdgeTpu runtime(libedgetpu.so).
// The version is dynamically retrieved from shared object.
std::string GetRuntimeVersion();
}  // namespace coral

#endif  // EDGETPU_CPP_VERSION_H_
