/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//
// This header file defines EdgeTpuManager, and EdgeTpuContext.
// EdgeTpuContext is an object associated with one or more tflite::Interpreter.
// Instances of this class should be allocated through
// EdgeTpuManager::NewEdgeTpuContext.
// More than one Interpreter instances can point to the same context. This means
// the tasks from both would be executed under the same TPU context.
// The lifetime of this context must be longer than all associated
// tflite::Interpreter instances.
//
// Typical usage with NNAPI:
//
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   auto model =
//   tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());
//
//   // Registers edge TPU custom op handler with Tflite resolver.
//   resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//
//   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//
//   interpreter->AllocateTensors();
//      .... (Prepare input tensors)
//   interpreter->Invoke();
//      .... (retrieving the result from output tensors)
//
//   // Releases interpreter instance to free up resources associated with
//   // this custom op.
//   interpreter.reset();
//
// Typical usage with Non-NNAPI:
//
//   // Sets up the tpu_context.
//   auto tpu_context =
//       edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
//
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   auto model =
//   tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());
//
//   // Registers edge TPU custom op handler with Tflite resolver.
//   resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//
//   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//
//   // Binds a context with a specific interpreter.
//   interpreter->SetExternalContext(kTfLiteEdgeTpuContext,
//     tpu_context.get());
//
//   // Note that all edge TPU context set ups should be done before this
//   // function is called.
//   interpreter->AllocateTensors();
//      .... (Prepare input tensors)
//   interpreter->Invoke();
//      .... (retrieving the result from output tensors)
//
//   // Releases interpreter instance to free up resources associated with
//   // this custom op.
//   interpreter.reset();
//
//   // Closes the edge TPU.
//   tpu_context.reset();

#ifndef TFLITE_PUBLIC_EDGETPU_H_
#define TFLITE_PUBLIC_EDGETPU_H_

// If the ABI changes in a backward-incompatible way, please increment the
// version number in the BUILD file.
// LINT.IfChange()

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/context.h"

#if defined(_WIN32)
#ifdef EDGETPU_COMPILE_LIBRARY
#define EDGETPU_EXPORT __declspec(dllexport)
#else
#define EDGETPU_EXPORT __declspec(dllimport)
#endif  // EDGETPU_COMPILE_LIBRARY
#else
#define EDGETPU_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

namespace edgetpu {

// EdgeTPU custom op.
static const char kCustomOp[] = "edgetpu-custom-op";

enum class DeviceType {
  kApexPci = 0,
  kApexUsb = 1,
};

class EdgeTpuContext;

// Singleton edge TPU manager for allocating new TPU contexts.
// Functions in this interface are thread-safe.
class EDGETPU_EXPORT EdgeTpuManager {
 public:
  using DeviceOptions = std::unordered_map<std::string, std::string>;
  struct DeviceEnumerationRecord {
    DeviceType type;
    std::string path;

    // Returns true if two enumeration records point to the same device.
    friend bool operator==(const DeviceEnumerationRecord& lhs,
                           const DeviceEnumerationRecord& rhs) {
      return (lhs.type == rhs.type) && (lhs.path == rhs.path);
    }

    // Returns true if two enumeration records point to defferent devices.
    friend bool operator!=(const DeviceEnumerationRecord& lhs,
                           const DeviceEnumerationRecord& rhs) {
      return !(lhs == rhs);
    }
  };

  // Returns pointer to the singleton object, or nullptr if not supported on
  // this platform.
  static EdgeTpuManager* GetSingleton();

  // NewEdgeTpuContext family functions has been deprecated and will be removed
  // in the future. Please use OpenDevice for new code.
  //
  // These functions return an unique_ptr to EdgeTpuContext, with
  // the intention that the device will be closed, and associate resources
  // released, when the unique_ptr leaves scope.
  //
  // These functions seek exclusive ownership of the opened devices. As they
  // cannot open devices already opened by OpenDevice, and vice versa.
  // Devices opened through these functions would have attribute
  // "ExclusiveOwnership", which can be queried through
  // #EdgeTpuContext::GetDeviceOptions().

  // Creates a new Edge TPU context to be assigned to Tflite::Interpreter. The
  // Edge TPU context is associated with the default TPU device. May be null
  // if underlying device cannot be found or open. Caller owns the returned new
  // context and should destroy the context either implicity or explicitly after
  // all interpreters sharing this context are destroyed.
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext() = 0;

  // Same as above, but the created context is associated with the specified
  // type.
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type) = 0;

  // Same as above, but the created context is associated with the specified
  // type and device path.
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path) = 0;

  // Same as above, but the created context is associated with the given device
  // type, path and options.
  //
  // Available options are:
  //  - "Performance": ["Low", "Medium", "High", "Max"] (Default is "Max")
  //  - "Usb.AlwaysDfu": ["True", "False"] (Default is "False")
  //  - "Usb.MaxBulkInQueueLength": ["0",.., "255"] (Default is "32")
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path,
      const DeviceOptions& options) = 0;

  // Enumerates all connected Edge TPU devices.
  virtual std::vector<DeviceEnumerationRecord> EnumerateEdgeTpu() const = 0;

  // OpenDevice family of functions return a shared_ptr to EdgeTpuContext, with
  // the intention that the device can be shared among multiple software
  // components.
  //
  // These functions seek shared ownership of the opened devices. As they
  // cannot open devices already opened by NewEdgeTpuContext, and vice versa.
  // The device would be closed after the last reference leaves scope.

  // Opens the default Edge TPU device.
  //
  // Multiple invocations of this function could return handle to the same
  // device, but there is no guarantee.
  //
  // Returns a shared pointer to Edge TPU device. The shared_ptr could point to
  // nullptr in case of error.
  virtual std::shared_ptr<EdgeTpuContext> OpenDevice() = 0;

  // Same as above, but the returned context is associated with the specified
  // type.
  virtual std::shared_ptr<EdgeTpuContext> OpenDevice(
      DeviceType device_type) = 0;

  // Same as above, but the returned context is associated with the specified
  // type and device path. If path is empty, any device of the specified type
  // could be returned.
  virtual std::shared_ptr<EdgeTpuContext> OpenDevice(
      DeviceType device_type, const std::string& device_path) = 0;

  // Same as above, but the specified options would used to create a new context
  // if no existing device is compatible with the specified type and path.
  //
  // If a device of compatible type and path can be found, the options could be
  // ignored. It is the caller's responsibility to verify if the returned
  // context is desirable, through #EdgeTpuContext::GetDeviceOptions().
  //
  // Available options are:
  //  - "Performance": ["Low", "Medium", "High", "Max"] (Default is "Max")
  //  - "Usb.AlwaysDfu": ["True", "False"] (Default is "False")
  //  - "Usb.MaxBulkInQueueLength": ["0",.., "255"] (Default is "32")
  virtual std::shared_ptr<EdgeTpuContext> OpenDevice(
      DeviceType device_type, const std::string& device_path,
      const DeviceOptions& options) = 0;

  // Returns a snapshot of currently opened shareable devices.
  // Exclusively owned Edge TPU devices cannot be returned here, as they're
  // owned by unique pointers.
  virtual std::vector<std::shared_ptr<EdgeTpuContext>> GetOpenedDevices()
      const = 0;

  // Sets verbosity of operating logs related to edge TPU.
  // Verbosity level can be set to [0-10], in which 10 is the most verbose.
  virtual TfLiteStatus SetVerbosity(int verbosity) = 0;

  // Returns the version of EdgeTPU runtime stack.
  virtual std::string Version() const = 0;

 protected:
  // No deletion for this singleton instance.
  virtual ~EdgeTpuManager() = default;
};

// External context to be assigned through
// tflite::Interpreter::SetExternalContext.
// One should get hold of either shared_ptr from EdgeTpuManager::OpenDevice, or
// unique_ptr from EdgeTpuManager::NewEdgeTpuContext, to ensure ownership, and
// avoid using this pointer directly.
// Functions in this interface are thread-safe.
class EdgeTpuContext : public TfLiteExternalContext {
 public:
  virtual ~EdgeTpuContext() = 0;

  // Returns a pointer to the device enumeration record for this device,
  // if available.
  virtual const EdgeTpuManager::DeviceEnumerationRecord& GetDeviceEnumRecord()
      const = 0;

  // Returns a snapshot of the options used to open this
  // device, and current state, if available.
  //
  // Supported attributes are:
  //  - "ExclusiveOwnership": present when it is under exclusive ownership
  //  (unique_ptr returned by NewEdgeTpuContext).
  //  - "IsReady": present when it is ready for further requests.
  virtual EdgeTpuManager::DeviceOptions GetDeviceOptions() const = 0;

  // Returns true if the device is most likely ready to accept requests.
  // When there are fatal errors, including unplugging of an USB device, the
  // state of this device would be changed.
  virtual bool IsReady() const = 0;
};

// Returns pointer to an instance of TfLiteRegistration to handle
// EdgeTPU custom ops, to be used with
// tflite::ops::builtin::BuiltinOpResolver::AddCustom
EDGETPU_EXPORT TfLiteRegistration* RegisterCustomOp();

// Inserts name of device type into ostream. Returns the modified ostream.
EDGETPU_EXPORT std::ostream& operator<<(std::ostream& out,
                                        DeviceType device_type);

}  // namespace edgetpu

// LINT.ThenChange(BUILD:version)

#endif  // TFLITE_PUBLIC_EDGETPU_H_
