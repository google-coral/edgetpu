#ifndef EDGETPU_CPP_PIPELINE_UTILS_H_
#define EDGETPU_CPP_PIPELINE_UTILS_H_

#include <unordered_set>

#include "src/cpp/pipeline/allocator.h"
#include "src/cpp/pipeline/common.h"
#include "tensorflow/lite/interpreter.h"

namespace coral {

// Returns all input tensor names for the given tflite::Interpreter.
std::unordered_set<std::string> GetInputTensorNames(
    const tflite::Interpreter& interpreter);

// Deallocates the memory for the given tensors.
// Use this to free output tensors each time you process the results.
//
// @param tensors A vector of PipelineTensor objects to release.
// @param allocator The Allocator originally used to allocate the tensors.
void FreeTensors(const std::vector<PipelineTensor>& tensors,
                 Allocator* allocator);

// Returns the input tensor matching `name` in the given tflite::Interpreter.
// Returns nullptr if such tensor does not exist.
const TfLiteTensor* GetInputTensor(const tflite::Interpreter& interpreter,
                                   const char* name);

}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_UTILS_H_
