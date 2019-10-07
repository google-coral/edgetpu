#include "src/cpp/version.h"

#include "edgetpu.h"

namespace coral {
const char kEdgeTpuCppWrapperVersion[] =
    "API(2.1) TF(5d0b55dd4a00c74809e5b32217070a26ac6ef823)";
const char kSupportedRuntimeVersion[] = "RuntimeVersion(12)";
std::string GetRuntimeVersion() {
  return ::edgetpu::EdgeTpuManager::GetSingleton()->Version();
}
}  // namespace coral
