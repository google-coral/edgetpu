#include "benchmark/benchmark.h"
#include "edgetpu.h"
#include "glog/logging.h"
#include "src/cpp/basic/edgetpu_resource_manager.h"

namespace coral {

static void BM_EnumerateEdgeTpu(benchmark::State& state) {
  while (state.KeepRunning()) {
    edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  }
}
BENCHMARK(BM_EnumerateEdgeTpu);

static void BM_CreateEdgeTpuContextFromScratch(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::unique_ptr<EdgeTpuResource> edgetpu_resource;
    EdgeTpuResourceManager::GetSingleton()->GetEdgeTpuResource(
        &edgetpu_resource);
  }
}
BENCHMARK(BM_CreateEdgeTpuContextFromScratch);

static void BM_CreateEdgeTpuContextCached(benchmark::State& state) {
  const auto& tpu_devices =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  CHECK_GT(tpu_devices.size(), 0);
  std::unique_ptr<EdgeTpuResource> edgetpu_resource;
  const auto& device_path = tpu_devices[0].path;
  EdgeTpuResourceManager::GetSingleton()->GetEdgeTpuResource(device_path,
                                                             &edgetpu_resource);

  while (state.KeepRunning()) {
    std::unique_ptr<EdgeTpuResource> tmp;
    EdgeTpuResourceManager::GetSingleton()->GetEdgeTpuResource(device_path,
                                                               &tmp);
  }
}
BENCHMARK(BM_CreateEdgeTpuContextCached);
}  // namespace coral
