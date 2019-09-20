#include "src/cpp/version.h"

#include "edgetpu.h"

namespace coral {
const char kEdgeTpuCppWrapperVersion[] =
    "API(2.0) TF(84c176726febd6f0b1eaae5b165af8b6a983b2f8)";
const char kSupportedRuntimeVersion[] = "RuntimeVersion(12)";
std::string GetRuntimeVersion() {
  return ::edgetpu::EdgeTpuManager::GetSingleton()->Version();
}
}  // namespace coral
