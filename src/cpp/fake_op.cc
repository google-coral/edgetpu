#include "src/cpp/fake_op.h"

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace coral {
namespace fake_op_double {
using tflite::GetInput;
using tflite::GetOutput;
using tflite::NumDimensions;
using tflite::NumInputs;
using tflite::NumOutputs;

struct OpData {
  bool throw_error;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  op_data->throw_error = m["throw_error"].AsBool();
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int num_dims = NumDimensions(input);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    output_size->data[i] = input->dims->data[i];
  }

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  uint8_t* input_data = input->data.uint8;
  float* output_data = output->data.f;

  size_t count = 1;
  int num_dims = NumDimensions(input);
  for (int i = 0; i < num_dims; ++i) {
    count *= input->dims->data[i];
  }

  for (size_t i = 0; i < count; ++i) {
    output_data[i] = static_cast<float>(input_data[i]) * 2;
  }
  if (op_data->throw_error) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fake_op_double

TfLiteRegistration* RegisterFakeOpDouble() {
  static TfLiteRegistration r = {fake_op_double::Init, fake_op_double::Free,
                                 fake_op_double::Prepare, fake_op_double::Eval};
  return &r;
}

}  // namespace coral
