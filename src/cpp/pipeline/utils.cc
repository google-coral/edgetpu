#include "src/cpp/pipeline/utils.h"

#include <cstring>

#include "glog/logging.h"

namespace coral {

std::unordered_set<std::string> GetInputTensorNames(
    const tflite::Interpreter& interpreter) {
  std::unordered_set<std::string> names(interpreter.inputs().size());
  for (int i = 0; i < interpreter.inputs().size(); ++i) {
    names.insert(interpreter.input_tensor(i)->name);
  }
  return names;
}

void FreeTensors(const std::vector<PipelineTensor>& tensors,
                 Allocator* allocator) {
  for (const auto& tensor : tensors) {
    VLOG(1) << "Releasing tensor "
            << " at addr: " << static_cast<void*>(tensor.data.data);
    allocator->free(tensor.data.data, tensor.bytes);
  }
}

const TfLiteTensor* GetInputTensor(const tflite::Interpreter& interpreter,
                                   const char* name) {
  for (const int input_index : interpreter.inputs()) {
    const auto* input_tensor = interpreter.tensor(input_index);
    if (std::strcmp(input_tensor->name, name) == 0) {
      return input_tensor;
    }
  }
  return nullptr;
}
}  // namespace coral
