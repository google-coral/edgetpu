#include "src/cpp/version.h"

#include "edgetpu.h"

namespace coral {
const char kEdgeTpuCppWrapperVersion[] =
    "API(2.1) TF(d855adfc5a0195788bf5f92c3c7352e638aa1109)";
const char kSupportedRuntimeVersion[] = "RuntimeVersion(13)";
std::string GetRuntimeVersion() {
  return ::edgetpu::EdgeTpuManager::GetSingleton()->Version();
}
}  // namespace coral
