#include "src/cpp/basic/edgetpu_resource_manager.h"

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "glog/logging.h"

namespace coral {

EdgeTpuResource::~EdgeTpuResource() {
  EdgeTpuResourceManager::GetSingleton()->ReclaimEdgeTpuResource(path_);
}

EdgeTpuResourceManager* EdgeTpuResourceManager::GetSingleton() {
  // Such static local variable's initialization is thread-safe.
  // https://stackoverflow.com/questions/8102125/is-local-static-variable-initialization-thread-safe-in-c11.
  static auto* const manager = new EdgeTpuResourceManager();
  return manager;
}

EdgeTpuResourceManager::EdgeTpuResourceManager() {
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
}

EdgeTpuApiStatus EdgeTpuResourceManager::CreateEdgeTpuResource(
    const edgetpu::DeviceType type, const std::string& path,
    std::unique_ptr<EdgeTpuResource>* resource) {
  CHECK(resource);
  std::unordered_map<std::string, std::string> options = {
      {"Usb.MaxBulkInQueueLength", "8"},
  };
  auto tpu_context =
      edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(type, path, options);
  if (!tpu_context) {
    error_reporter_->Report(
        absl::Substitute("Error in device opening ($0)!", path));
    return kEdgeTpuApiError;
  }
  resource_map_[path].usage_count = 1;
  resource_map_[path].context = tpu_context;
  // Using `new` to access a non-public constructor.
  *resource =
      absl::WrapUnique(new EdgeTpuResource(path, resource_map_[path].context));
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus EdgeTpuResourceManager::GetEdgeTpuResource(
    std::unique_ptr<EdgeTpuResource>* resource) {
  const auto& tpu_devices =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();

  if (tpu_devices.empty()) {
    error_reporter_->Report("No Edge TPU device detected!");
    return kEdgeTpuApiError;
  } else if (tpu_devices.size() == 1) {
    return GetEdgeTpuResource(tpu_devices[0].path, resource);
  } else {
    return GetEdgeTpuResourceMultipleDetected(resource);
  }
}

EdgeTpuApiStatus EdgeTpuResourceManager::GetEdgeTpuResource(
    const std::string& path, std::unique_ptr<EdgeTpuResource>* resource) {
  absl::MutexLock lock(&mu_);
  // Call to EnumerateEdgeTpu() is relatively expensive (10+ ms), Search
  // `resource_map_` first to see if there's cache-hit.
  auto it = resource_map_.find(path);
  if (it != resource_map_.end()) {
    it->second.usage_count++;
    // Using `new` to access a non-public constructor.
    *resource = absl::WrapUnique(new EdgeTpuResource(path, it->second.context));
    return kEdgeTpuApiOk;
  }

  // No cache-hit, create EdgeTpuResource from scratch.
  const auto& tpu_devices =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  for (const auto& device : tpu_devices) {
    if (device.path == path) {
      CHECK(resource_map_.find(path) == resource_map_.end());
      return CreateEdgeTpuResource(device.type, device.path, resource);
    }
  }
  error_reporter_->Report(
      absl::Substitute("Path $0 does not map to an Edge TPU device.", path));
  return kEdgeTpuApiError;
}

EdgeTpuApiStatus EdgeTpuResourceManager::GetEdgeTpuResourceMultipleDetected(
    std::unique_ptr<EdgeTpuResource>* resource) {
  absl::MutexLock lock(&mu_);
  const auto& tpu_devices =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();

  // Always prefer to use PCIe version of EdgeTpu.
  for (const auto& device : tpu_devices) {
    if (device.type != edgetpu::DeviceType::kApexPci) continue;
    if (resource_map_.find(device.path) == resource_map_.end()) {
      return CreateEdgeTpuResource(device.type, device.path, resource);
    }
  }

  // Check USB version of EdgeTpu.
  for (const auto& device : tpu_devices) {
    if (device.type != edgetpu::DeviceType::kApexUsb) continue;
    if (resource_map_.find(device.path) == resource_map_.end()) {
      return CreateEdgeTpuResource(device.type, device.path, resource);
    }
  }
  error_reporter_->Report(
      "Multiple Edge TPUs detected and all have been mapped to at least one "
      "model. If you want to share one Edge TPU with multiple models, specify "
      "`device_path` name.");
  return kEdgeTpuApiError;
}

EdgeTpuApiStatus EdgeTpuResourceManager::ReclaimEdgeTpuResource(
    const std::string& path) {
  absl::MutexLock lock(&mu_);
  auto it = resource_map_.find(path);
  if (it != resource_map_.end()) {
    it->second.usage_count--;
    if (it->second.usage_count == 0) {
      it->second.context.reset();
      resource_map_.erase(it);
    }
  } else {
    error_reporter_->Report(
        absl::Substitute("Trying to reclaim unassigned device: $0.", path));
    return kEdgeTpuApiError;
  }
  return kEdgeTpuApiOk;
}

std::vector<std::string> EdgeTpuResourceManager::ListEdgeTpuPaths(
    const EdgeTpuState& state) {
  std::vector<std::string> result;
  const auto& edgetpu_devices =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  switch (state) {
    case EdgeTpuState::kNone: {
      for (const auto& device : edgetpu_devices) {
        result.push_back(device.path);
      }
      break;
    }
    case EdgeTpuState::kAssigned:
    case EdgeTpuState::kUnassigned: {
      // Return true when
      //  *) `state` is `kAssigned` and is recorded in `resource_map_`;
      //  *) `state` is `kUnassigned` and is NOT recorded in `resource_map_`;
      auto should_return = [this, &state](const std::string& path) -> bool {
        absl::ReaderMutexLock lock(&mu_);
        const bool assigned = (state == EdgeTpuState::kAssigned);
        const bool recorded = (resource_map_.find(path) != resource_map_.end());
        return (assigned && recorded) || ((!assigned) && (!recorded));
      };
      for (const auto& device : edgetpu_devices) {
        if (should_return(device.path)) {
          result.push_back(device.path);
        }
      }
      break;
    }
  }
  return result;
}
}  // namespace coral
