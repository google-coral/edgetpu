#ifndef EDGETPU_CPP_BASIC_EDGETPU_RESOURCE_MANAGER_H_
#define EDGETPU_CPP_BASIC_EDGETPU_RESOURCE_MANAGER_H_

#include <unordered_map>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "edgetpu.h"
#include "src/cpp/error_reporter.h"

namespace coral {

class EdgeTpuResourceManager;

// A thin wrapper around `EdgeTpuContext`.
//  *) It records the device path associated with this context.
//  *) It un-registers itself with `EdgeTpuResourceManager` during destruction.
//     This (un-register) is critical information for `EdgeTpuResourceManager`
//     in use case where multiple models need to share one EdgeTpuContext.
class EdgeTpuResource {
 public:
  ~EdgeTpuResource();

  // Gets underlying EdgeTpuContext.
  edgetpu::EdgeTpuContext* context() { return context_.get(); }

  // Gets associated device path for `EdgeTpuContext`.
  std::string path() const { return path_; }

 private:
  friend class EdgeTpuResourceManager;
  // Only allows `EdgeTpuResourceManager` to create `EdgeTpuResource`, such that
  // no one can mess up the ownership management of `EdgeTpuResourceManager`
  // through `EdgeTpuResource`.
  EdgeTpuResource(const std::string& path,
                  std::shared_ptr<edgetpu::EdgeTpuContext> context)
      : path_(path), context_(context) {}
  // Disallows copy constructor and assignment.
  EdgeTpuResource(const EdgeTpuResource&) = delete;
  EdgeTpuResource& operator=(const EdgeTpuResource&) = delete;

  // Path associated with `EdgeTpuContext`.
  std::string path_;
  std::shared_ptr<edgetpu::EdgeTpuContext> context_;
};

// This class manages `EdgeTpuResource`.
//
// It creates `EdgeTpuContext` and caches it, such that multiple models can
// share one `EdgeTpuContext` if they have to. When there is no more reference
// to the context, the cached context is released automatically.
//
// Example usage:
//     auto model = tflite::FlatBufferModel::BuildFromFile("model_name.tflite");
//     std::unique_ptr<EdgeTpuResource> edgetpu_resource;
//     CHECK_EQ(kEdgeTpuApiOk,
//              EdgeTpuResourceManager::GetSingleton()->GetEdgeTpuResource(
//                  &edgetpu_resource));
//     VLOG(1) << "Edge TPU device path: " << edgetpu_resource->path();
//     tflite::ops::builtin::BuiltinOpResolver resolver;
//     resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//     std::unique_ptr<tflite::Interpreter> interpreter;
//     CHECK(tflite::InterpreterBuilder(model, resolver)(&interpreter) ==
//           kTfLiteOk);
//     interpreter->SetExternalContext(kTfLiteEdgeTpuContext,
//                                     edgetpu_resource.context());
//
// Note: by default, use `GetEdgeTpuResource(&resource)`, which tries to do a
// 1-on-1 mapping between model and Edge TPU device. This allows each model to
// take advantage of parameter-caching mode. Use `GetEdgeTpuResource(path,
// &resource)` to share one Edge TPU among different models.
//
// This class is thread-safe.
class EdgeTpuResourceManager {
 public:
  static EdgeTpuResourceManager* GetSingleton();

  // Gets next available EdgeTpuResource.
  EdgeTpuApiStatus GetEdgeTpuResource(
      std::unique_ptr<EdgeTpuResource>* resource) ABSL_LOCKS_EXCLUDED(mu_);

  // Gets context associated with specified device (through `path`).
  //
  // This function is particular useful if one wants to share one
  // `EdgeTpuResource` among multiple models.
  //
  // Note: If multiple models share one Edge TPU context, concurrent access to
  // the context is handled under the hood. Edge TPU will buffer the request and
  // return the correct result to individual model.
  EdgeTpuApiStatus GetEdgeTpuResource(
      const std::string& path, std::unique_ptr<EdgeTpuResource>* resource);

  EdgeTpuApiStatus ReclaimEdgeTpuResource(const std::string& path)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Edge TPU assignment state.
  enum class EdgeTpuState {
    // Lists all Edge TPU devices (assigned or unassigned).
    kNone,
    // Lists all Edge TPU devices that has been assigned to at least one model.
    kAssigned,
    // Lists all Edge TPU devices that is available.
    kUnassigned,
  };
  // Lists path of Edge TPU devices.
  std::vector<std::string> ListEdgeTpuPaths(const EdgeTpuState& state);

  // Caller can use this function to retrieve error message when get
  // kEdgeTpuApiError.
  std::string get_error_message() { return error_reporter_->message(); }

 private:
  EdgeTpuResourceManager();
  ~EdgeTpuResourceManager() = default;

  EdgeTpuApiStatus CreateEdgeTpuResource(
      const edgetpu::DeviceType type, const std::string& path,
      std::unique_ptr<EdgeTpuResource>* resource)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Gets next available EdgeTpuResource when there are multiple detected.
  EdgeTpuApiStatus GetEdgeTpuResourceMultipleDetected(
      std::unique_ptr<EdgeTpuResource>* resource) ABSL_LOCKS_EXCLUDED(mu_);

  struct ResourceState {
    int usage_count = 0;
    std::shared_ptr<edgetpu::EdgeTpuContext> context;
  };
  absl::Mutex mu_;
  // Keeps track of assigned Edge TPU, e.g., how many references on it.
  // Keyed by device path.
  std::unordered_map<std::string, ResourceState> resource_map_
      ABSL_GUARDED_BY(mu_);
  // Data structure to stores error messages.
  std::unique_ptr<EdgeTpuErrorReporter> error_reporter_;
};
}  // namespace coral

#endif  // EDGETPU_CPP_BASIC_EDGETPU_RESOURCE_MANAGER_H_
