/*
Copyright 2019 Google LLC

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
// This header defines C API to provide edge TPU support for TensorFlow Lite
// framework. It is only available for non-NNAPI use cases.
//
// Typical API usage from C++ code involves serveral steps:
//
// 1. Create tflite::FlatBufferModel which may contain edge TPU custom op.
//
// auto model =
//    tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());
//
// 2. Create tflite::Interpreter.
//
// tflite::ops::builtin::BuiltinOpResolver resolver;
// std::unique_ptr<tflite::Interpreter> interpreter;
// tflite::InterpreterBuilder(model, resolver)(&interpreter);
//
// 3. Enumerate edge TPU devices.
//
// size_t num_devices;
// std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
//     edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
//
// assert(num_devices > 0);
// const auto& device = devices.get()[0];
//
// 4. Modify interpreter with the delegate.
//
// auto* delegate =
//     edgetpu_create_delegate(device.type, device.path, nullptr, 0);
// interpreter->ModifyGraphWithDelegate({delegate, edgetpu_free_delegate});
//
// 5. Prepare input tensors and run inference.
//
// interpreter->AllocateTensors();
//   .... (Prepare input tensors)
// interpreter->Invoke();
//   .... (Retrieve the result from output tensors)

#ifndef TFLITE_PUBLIC_EDGETPU_C_H_
#define TFLITE_PUBLIC_EDGETPU_C_H_

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

#ifdef __cplusplus
extern "C" {
#endif

enum edgetpu_device_type {
  EDGETPU_APEX_PCI = 0,
  EDGETPU_APEX_USB = 1,
};

struct edgetpu_device {
  enum edgetpu_device_type type;
  const char* path;
};

struct edgetpu_option {
  const char* name;
  const char* value;
};

// Returns array of connected edge TPU devices.
EDGETPU_EXPORT struct edgetpu_device* edgetpu_list_devices(size_t* num_devices);

// Frees array returned by `edgetpu_list_devices`.
EDGETPU_EXPORT void edgetpu_free_devices(struct edgetpu_device* dev);

// Creates a delegate which handles all edge TPU custom ops inside
// `tflite::Interpreter`. Options must be available only during the call of this
// function.
EDGETPU_EXPORT TfLiteDelegate* edgetpu_create_delegate(
    enum edgetpu_device_type type, const char* name,
    const struct edgetpu_option* options, size_t num_options);

// Frees delegate returned by `edgetpu_create_delegate`.
EDGETPU_EXPORT void edgetpu_free_delegate(TfLiteDelegate* delegate);

// Sets verbosity of operating logs related to edge TPU.
// Verbosity level can be set to [0-10], in which 10 is the most verbose.
EDGETPU_EXPORT void edgetpu_verbosity(int verbosity);

// Returns the version of edge TPU runtime stack.
EDGETPU_EXPORT const char* edgetpu_version();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TFLITE_PUBLIC_EDGETPU_C_H_
