#include "src/cpp/pipeline/test_utils.h"

#include <random>

#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "tensorflow/lite/kernels/register.h"

ABSL_FLAG(std::string, test_data_dir, "test_data", "Test data directory");

namespace coral {
using tflite::ops::builtin::BuiltinOpResolver;

std::string TestDataPath(const std::string& name) {
  return absl::StrCat(absl::GetFlag(FLAGS_test_data_dir), "/", name);
}

void FillRandom(uint8_t* buffer, size_t size) {
  CHECK(buffer);
  std::default_random_engine generator(12345);
  std::uniform_int_distribution<> distribution(0, UINT8_MAX);
  std::generate(buffer, buffer + size,
                [&]() { return distribution(generator); });
}

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* context) {
  CHECK(context);
  BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(model.GetModel(), resolver);
  CHECK_EQ(interpreter_builder(&interpreter), kTfLiteOk)
      << "Error in interpreter initialization.";

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
  interpreter->SetNumThreads(1);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk)
      << "Failed to allocate tensors.";

  return interpreter;
}

std::vector<std::string> SegmentsNames(const std::string& model_base_name,
                                       int num_segments) {
  std::vector<std::string> result(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    result[i] = absl::StrCat(model_base_name, "_segment_", i, "_of_",
                             num_segments, "_edgetpu.tflite");
  }
  return result;
}

std::vector<PipelineTensor> CreateRandomInputTensors(
    const tflite::Interpreter* interpreter, Allocator* allocator) {
  std::vector<PipelineTensor> input_tensors;
  for (int input_index : interpreter->inputs()) {
    const auto* input_tensor = interpreter->tensor(input_index);
    PipelineTensor input_buffer;
    if (allocator) {
      input_buffer.data.data = allocator->alloc(input_tensor->bytes);
    } else {
      input_buffer.data.data = std::malloc(input_tensor->bytes);
    }
    input_buffer.bytes = input_tensor->bytes;
    input_buffer.type = input_tensor->type;
    FillRandom(reinterpret_cast<uint8_t*>(input_buffer.data.data),
               input_buffer.bytes);
    input_tensors.push_back(input_buffer);
  }
  return input_tensors;
}

}  // namespace coral
