#include "src/cpp/learn/utils.h"

#include <algorithm>
#include <cstdio>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "glog/logging.h"
#include "src/cpp/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace learn {

namespace {

// Gets total number of elements represented by `shape`.
int GetTotalElements(const std::vector<int>& shape) {
  int result = 1;
  for (const auto& s : shape) {
    result *= s;
  }
  return result;
}

// Finds the index of `target_op`'s opcode, if it does not exist, add
// corresponding opcode to `model_t`. Returns index of `target_op`'s opcode.
int FindOrCreateOpcode(const tflite::BuiltinOperator target_op,
                       const std::string& custom_code,
                       tflite::ModelT* model_t) {
  int target_opcode_index =
      FindOpcodeIndex(model_t->operator_codes, target_op, custom_code);
  VLOG(1) << "Target " << EnumNameBuiltinOperator(target_op)
          << "'s opcode index: " << target_opcode_index;

  if (target_opcode_index == -1) {
    // Add opcode.
    auto target_opcode = absl::make_unique<tflite::OperatorCodeT>();
    target_opcode->builtin_code = target_op;
    if (target_op == tflite::BuiltinOperator_CUSTOM) {
      target_opcode->custom_code = custom_code;
    }
    model_t->operator_codes.push_back(std::move(target_opcode));
    target_opcode_index = model_t->operator_codes.size() - 1;
    VLOG(1) << "Opcode is added with index: " << target_opcode_index;
  }
  return target_opcode_index;
}

// Gets operator's builtin options. NOTE that These options are tuned only for
// last layer backprop method. Please modify them if use for other purposes.
tflite::BuiltinOptionsUnion GetOpBuiltinOptions(
    tflite::BuiltinOperator op_type,
    const std::vector<internal::TensorConfig>& tensor_configs) {
  tflite::BuiltinOptionsUnion result;
  switch (op_type) {
    case tflite::BuiltinOperator_L2_NORMALIZATION: {
      auto l2_norm_builtin_options =
          absl::make_unique<tflite::L2NormOptionsT>();
      result.type = tflite::BuiltinOptions_L2NormOptions;
      result.value = l2_norm_builtin_options.release();
      break;
    }
    case tflite::BuiltinOperator_CONV_2D: {
      auto conv2d_builtin_options = absl::make_unique<tflite::Conv2DOptionsT>();
      conv2d_builtin_options->padding = tflite::Padding_SAME;
      conv2d_builtin_options->stride_h = 1;
      conv2d_builtin_options->stride_w = 1;
      result.type = tflite::BuiltinOptions_Conv2DOptions;
      result.value = conv2d_builtin_options.release();
      break;
    }
    case tflite::BuiltinOperator_FULLY_CONNECTED: {
      auto fullyconnected_builtin_options =
          absl::make_unique<tflite::FullyConnectedOptionsT>();
      result.type = tflite::BuiltinOptions_FullyConnectedOptions;
      result.value = fullyconnected_builtin_options.release();
      break;
    }
    case tflite::BuiltinOperator_RESHAPE: {
      auto reshape_builtin_options =
          absl::make_unique<tflite::ReshapeOptionsT>();
      reshape_builtin_options->new_shape = tensor_configs[0].shape;
      result.type = tflite::BuiltinOptions_ReshapeOptions;
      result.value = reshape_builtin_options.release();
      break;
    }
    case tflite::BuiltinOperator_SOFTMAX: {
      auto softmax_builtin_options =
          absl::make_unique<tflite::SoftmaxOptionsT>();
      // Can NOT leave `beta` as default 0. Otherwise, it will trigger check
      // failure in `QuantizeMultiplierGreaterThanOne`.
      // //depot/google3/third_party/tensorflow/lite/kernels/internal/quantization_util.cc
      softmax_builtin_options->beta = 1.0f;
      result.type = tflite::BuiltinOptions_SoftmaxOptions;
      result.value = softmax_builtin_options.release();
      break;
    }
    default:
      LOG(FATAL) << "Unsupported operator type: " << op_type;
      break;
  }
  return result;
}

// Returns size in bytes for tensor buffer, based on location, type and shape.
//
// NOTE: intermediate tensors (i.e., non-parameter tensors) must be refer to
// empty buffer; otherwise, the tensors' buffer will be treated as read-only,
// which at least causes Conv2d to fail because it always resizes the output
// tensor.
// //depot/google3/third_party/tensorflow/lite/kernels/conv.cc?l=319
int GetBufferSizeBytes(const std::vector<int>& shape,
                       internal::TensorLocation location,
                       tflite::TensorType type) {
  int result = 0;
  if (location == internal::TensorLocation::kParameter) {
    if (type == tflite::TensorType_UINT8) {
      result = GetTotalElements(shape);
    } else if (type == tflite::TensorType_INT32) {
      result = GetTotalElements(shape) * sizeof(int32_t);
    } else if (type == tflite::TensorType_FLOAT32) {
      result = GetTotalElements(shape) * sizeof(float);
    } else {
      LOG(FATAL) << "Unsupported tensor type: " << type;
    }
  }
  VLOG(1) << "Buffer size in bytes: " << result;
  return result;
}

// Sanity checks.
EdgeTpuApiStatus ValidateClassificationModel(const tflite::ModelT* model_t,
                                             EdgeTpuErrorReporter* reporter) {
  CHECK(model_t);
  CHECK(reporter);
  EDGETPU_API_REPORT_ERROR(
      reporter, model_t->subgraphs.size() != 1,
      absl::Substitute("Model must have one and only one subgraph. Actual: $0.",
                       model_t->subgraphs.size()));
  EDGETPU_API_REPORT_ERROR(
      reporter, model_t->subgraphs[0]->outputs.size() != 1,
      absl::Substitute("Model must have one and only one output. Actual: $0.",
                       model_t->subgraphs[0]->outputs.size()));
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ValidateOperatorInputs(
    const std::vector<internal::TensorConfig>& tensor_configs,
    tflite::BuiltinOperator op_type, const tflite::ModelT* model_t,
    EdgeTpuErrorReporter* reporter) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);

  EDGETPU_API_REPORT_ERROR(
      reporter, model_t->subgraphs[0]->outputs.size() != 1,
      absl::Substitute("Model must have one and only one output. Actual: $0.",
                       model_t->subgraphs[0]->outputs.size()));

  switch (op_type) {
    case tflite::BuiltinOperator_CONV_2D:
    case tflite::BuiltinOperator_FULLY_CONNECTED:
      EDGETPU_API_REPORT_ERROR(
          reporter, tensor_configs.size() != 3,
          absl::Substitute("Conv, FullyConnected operator must have three "
                           "input tensors. Actual: $0",
                           tensor_configs.size()));
      break;
    case tflite::BuiltinOperator_L2_NORMALIZATION:
    case tflite::BuiltinOperator_RESHAPE:
    case tflite::BuiltinOperator_SOFTMAX:
      EDGETPU_API_REPORT_ERROR(
          reporter, tensor_configs.size() != 1,
          absl::Substitute("L2-Norm, Reshape, and Softmax operators must have "
                           "one input tensor. Actual: $0",
                           tensor_configs.size()));
      break;
    case tflite::BuiltinOperator_CUSTOM:
      break;
    default:
      reporter->Report(
          absl::Substitute("Unsupported operator type: $0", op_type));
      return kEdgeTpuApiError;
  }
  return kEdgeTpuApiOk;
}

// Calculates quantization parameters for Conv2d / FullyConnected operator.
// Returns quantization parameters for kernel, bias, and output tensor.
std::vector<std::unique_ptr<tflite::QuantizationParametersT>>
CalculateQuantParamsLinearLayer(
    const float* weights, int weights_size, const float* biases,
    int biases_size, const tflite::QuantizationParametersT& input_tensor_quant,
    float out_tensor_min, float out_tensor_max) {
  std::vector<std::unique_ptr<tflite::QuantizationParametersT>> quant_params(3);
  // For weights.
  float min_weights_value =
      std::min<float>(0.0, *std::min_element(weights, weights + weights_size));
  float max_weights_value =
      std::max<float>(0.0, *std::max_element(weights, weights + weights_size));
  VLOG(1) << absl::Substitute("Weights range: ($0, $1).", min_weights_value,
                              max_weights_value);
  float weights_scale;
  int32_t weights_zero_point;
  std::tie(weights_scale, weights_zero_point) =
      QuantizationParams<uint8_t>(min_weights_value, max_weights_value);
  VLOG(1) << absl::Substitute("Weights (scale, zero_point): ($0, $1) ",
                              weights_scale, weights_zero_point);
  quant_params[0] =
      CreateQuantParam(/*min=*/{min_weights_value}, /*max=*/{max_weights_value},
                       /*scale=*/{weights_scale},
                       /*zero_point=*/{weights_zero_point});

  // For bias.
  auto min_biases_value =
      std::min<float>(0.0, *std::min_element(biases, biases + biases_size));
  auto max_biases_value =
      std::max<float>(0.0, *std::max_element(biases, biases + biases_size));
  VLOG(1) << absl::Substitute("Biases range: ($0, $1).", min_biases_value,
                              max_biases_value);
  // TFLite's conv2d implementation is very picky about quantization parameter
  // of bias. See `scale` computation in `GetQuantizedConvolutionMultipler` of
  // tensorflow/lite/kernels/kernel_util.cc
  //
  // Basically, it asks for biases_scale = input_tensor_scale * weights_scale
  float biases_scale = input_tensor_quant.scale[0] * weights_scale;
  // TFLite requires biases's zero point be 0.
  int32_t biases_zero_point = 0;
  VLOG(1) << absl::Substitute("Biases (scale, zero_point): ($0, $1) ",
                              biases_scale, biases_zero_point);
  quant_params[1] =
      CreateQuantParam(/*min=*/{min_biases_value}, /*max=*/{max_biases_value},
                       /*scale=*/{biases_scale},
                       /*zero_point=*/{biases_zero_point});

  // For output tensor.
  out_tensor_min = std::min<float>(0.0, out_tensor_min);
  out_tensor_max = std::max<float>(0.0, out_tensor_max);
  VLOG(1) << absl::Substitute("Output range: ($0, $1).", out_tensor_min,
                              out_tensor_max);
  float output_scale;
  int32_t output_zero_point;
  std::tie(output_scale, output_zero_point) =
      QuantizationParams<uint8_t>(out_tensor_min, out_tensor_max);
  VLOG(1) << absl::Substitute("Output (scale, zero_point): ($0, $1) ",
                              output_scale, output_zero_point);
  quant_params[2] = CreateQuantParam(/*min=*/{out_tensor_min},
                                     /*max=*/{out_tensor_max},
                                     /*scale=*/{output_scale},
                                     /*zero_point=*/{output_zero_point});
  return quant_params;
}

}  // namespace

namespace internal {

// Returns index of a tensor specified by name. If non-found, return -1;
int FindTensor(const std::string& name, const tflite::SubGraphT& subgraph_t) {
  for (int i = 0; i < subgraph_t.tensors.size(); ++i) {
    if (subgraph_t.tensors[i]->name == name) {
      return i;
    }
  }
  return -1;
}

EdgeTpuApiStatus AppendL2Norm(tflite::ModelT* model_t, int* new_op_index,
                              EdgeTpuErrorReporter* reporter) {
  EDGETPU_API_ENSURE_STATUS(ValidateClassificationModel(model_t, reporter));

  // Setup Quantization Parameters for L2Norm's output tensor.
  auto l2_norm_output_quant =
      CreateQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f}, /*scale=*/{1.0f / 128},
                       /*zero_point=*/{128});

  std::vector<TensorConfig> tensor_configs;
  const auto& output_tensors = GetGraphOutputTensors(model_t);
  const tflite::TensorT* current_output_tensor = output_tensors[0];
  tensor_configs.push_back({"Imprinting/L2Norm/Output",
                            tflite::TensorType_UINT8, TensorLocation::kOutput,
                            current_output_tensor->shape,
                            l2_norm_output_quant.release()});

  return AppendOperator(tensor_configs,
                        tflite::BuiltinOperator_L2_NORMALIZATION, model_t,
                        new_op_index, reporter);
}

EdgeTpuApiStatus AppendLinearLayer(
    const std::vector<int>& kernel_shape,
    std::vector<std::unique_ptr<tflite::QuantizationParametersT>> quant_params,
    tflite::ModelT* model_t, int* new_op_index,
    EdgeTpuErrorReporter* reporter) {
  EDGETPU_API_ENSURE_STATUS(ValidateClassificationModel(model_t, reporter));

  CHECK_EQ(quant_params.size(), 3);
  auto kernel_quant = std::move(quant_params[0]);
  auto bias_quant = std::move(quant_params[1]);
  auto output_quant = std::move(quant_params[2]);

  std::vector<TensorConfig> tensor_configs;
  std::vector<int> bias_shape = {kernel_shape[0]};
  const auto& output_tensors = GetGraphOutputTensors(model_t);
  const tflite::TensorT* current_output_tensor = output_tensors[0];
  std::vector<int> output_shape = CalculateLinearLayerOutputShape(
      current_output_tensor->shape, kernel_shape);
  tensor_configs.push_back({"Appended/FC/Weights", tflite::TensorType_UINT8,
                            TensorLocation::kParameter, kernel_shape,
                            kernel_quant.release()});

  tensor_configs.push_back({"Appended/FC/Bias", tflite::TensorType_INT32,
                            TensorLocation::kParameter, bias_shape,
                            bias_quant.release()});

  tensor_configs.push_back({"Appended/FC/Output", tflite::TensorType_UINT8,
                            TensorLocation::kOutput, output_shape,
                            output_quant.release()});
  if (kernel_shape.size() == 2) {
    return AppendOperator(tensor_configs,
                          tflite::BuiltinOperator_FULLY_CONNECTED, model_t,
                          new_op_index, reporter);
  } else {
    return AppendOperator(tensor_configs, tflite::BuiltinOperator_CONV_2D,
                          model_t, new_op_index, reporter);
  }
}

EdgeTpuApiStatus AppendReshape(tflite::ModelT* model_t, int* new_op_index,
                               EdgeTpuErrorReporter* reporter) {
  EDGETPU_API_ENSURE_STATUS(ValidateClassificationModel(model_t, reporter));

  // Output tensor of reshape should have the same quantization parameters as
  // its input, which is current graph's output tensor.
  const auto& output_tensors = GetGraphOutputTensors(model_t);
  const tflite::TensorT* current_output_tensor = output_tensors[0];
  auto reshape_output_quant =
      CreateQuantParam(current_output_tensor->quantization->min,
                       current_output_tensor->quantization->max,
                       current_output_tensor->quantization->scale,
                       current_output_tensor->quantization->zero_point);

  std::vector<int> reshape_output_shape = {current_output_tensor->shape.front(),
                                           current_output_tensor->shape.back()};
  std::vector<TensorConfig> tensor_configs;
  tensor_configs.push_back({"Imprinting/Reshape/Output",
                            tflite::TensorType_UINT8, TensorLocation::kOutput,
                            reshape_output_shape,
                            reshape_output_quant.release()});

  return AppendOperator(tensor_configs, tflite::BuiltinOperator_RESHAPE,
                        model_t, new_op_index, reporter);
}

EdgeTpuApiStatus AppendSoftmax(tflite::ModelT* model_t, int* new_op_index,
                               EdgeTpuErrorReporter* reporter) {
  EDGETPU_API_ENSURE_STATUS(ValidateClassificationModel(model_t, reporter));

  // Softmax's output is always within [0,1]
  auto softmax_output_quant =
      CreateQuantParam(/*min=*/{0.0f}, /*max=*/{1.0f}, /*scale=*/{1.0f / 256},
                       /*zero_point=*/{0});

  std::vector<TensorConfig> tensor_configs;
  const auto& output_tensors = GetGraphOutputTensors(model_t);
  const tflite::TensorT* current_output_tensor = output_tensors[0];
  tensor_configs.push_back({"Imprinting/Softmax/Output",
                            tflite::TensorType_UINT8, TensorLocation::kOutput,
                            current_output_tensor->shape,
                            softmax_output_quant.release()});

  return AppendOperator(tensor_configs, tflite::BuiltinOperator_SOFTMAX,
                        model_t, new_op_index, reporter);
}

int AppendBuffer(int buffer_size_bytes, tflite::ModelT* model_t) {
  CHECK(model_t);
  auto buffer = absl::make_unique<tflite::BufferT>();
  if (buffer_size_bytes > 0) {
    buffer->data.resize(buffer_size_bytes);
  }
  model_t->buffers.push_back(std::move(buffer));

  const auto buffer_index = model_t->buffers.size() - 1;
  VLOG(1) << "New buffer index is: " << buffer_index;
  return buffer_index;
}

int AppendTensor(const std::vector<int>& shape, const std::string& name,
                 int buffer_index, tflite::TensorType type,
                 std::unique_ptr<tflite::QuantizationParametersT> q_param,
                 tflite::SubGraphT* subgraph) {
  auto tensor = absl::make_unique<tflite::TensorT>();
  tensor->type = type;
  tensor->shape = shape;
  tensor->buffer = buffer_index;
  tensor->name = name;
  if (q_param) tensor->quantization = std::move(q_param);
  subgraph->tensors.push_back(std::move(tensor));

  const auto tensor_index = subgraph->tensors.size() - 1;
  VLOG(1) << "New tensor index: " << tensor_index;
  return tensor_index;
}

EdgeTpuApiStatus AppendOperator(const std::vector<TensorConfig>& tensor_configs,
                                tflite::BuiltinOperator op_type,
                                tflite::ModelT* model_t, int* new_op_index,
                                EdgeTpuErrorReporter* reporter) {
  CHECK(new_op_index);

  EDGETPU_API_ENSURE_STATUS(
      ValidateOperatorInputs(tensor_configs, op_type, model_t, reporter));
  const int opcode_index =
      FindOrCreateOpcode(op_type, /*custom_code=*/"", model_t);

  auto& subgraph = model_t->subgraphs[0];
  auto op = absl::make_unique<tflite::OperatorT>();

  op->opcode_index = opcode_index;
  // Current graph's output will become input tensor.
  op->inputs.push_back(subgraph->outputs[0]);
  // Be careful about the ownership transfer here. Check BuiltinOptionUnion
  // class's API to understand better.
  op->builtin_options = GetOpBuiltinOptions(op_type, tensor_configs);

  // Add tensor to subgraph.
  for (const auto& config : tensor_configs) {
    VLOG(1) << "-----";
    VLOG(1) << "Tensor name: " << config.name;
    const int buffer_size_bytes =
        GetBufferSizeBytes(config.shape, config.location, config.type);
    const int buffer_index = AppendBuffer(buffer_size_bytes, model_t);
    const int tensor_index =
        AppendTensor(config.shape, config.name, buffer_index, config.type,
                     absl::WrapUnique(config.quant), subgraph.get());
    if (config.location == TensorLocation::kOutput) {
      op->outputs.push_back(tensor_index);
    } else {
      op->inputs.push_back(tensor_index);
    }
  }

  // Update subgraph output.
  subgraph->outputs[0] = op->outputs[0];

  // Add operator to subgraph.
  subgraph->operators.push_back(std::move(op));

  *new_op_index = subgraph->operators.size() - 1;

  return kEdgeTpuApiOk;
}

void SetLinearParams(const std::vector<uint8_t>& kernel,
                     const std::vector<int32_t>& bias, int op_index,
                     tflite::ModelT* model_t) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);
  auto& subgraph = model_t->subgraphs[0];
  CHECK_GT(subgraph->operators.size(), op_index);
  auto& conv2d_op = subgraph->operators[op_index];

  // Conv2d in TFLite has 3 inputs and uses the following convention:
  //  - input 1 is input tensor;
  //  - input 2 is kernel tensor;
  //  - input 3 is bias tensor;
  const int kernel_tensor_index = conv2d_op->inputs[1];
  const int bias_tensor_index = conv2d_op->inputs[2];

  auto& kernel_tensor = subgraph->tensors[kernel_tensor_index];
  auto& kernel_buffer = model_t->buffers[kernel_tensor->buffer];
  // Resize buffer if necessary.
  if (kernel_buffer->data.size() < kernel.size()) {
    kernel_buffer->data.resize(kernel.size());
  }
  std::memcpy(kernel_buffer->data.data(), kernel.data(), kernel.size());

  auto& bias_tensor = subgraph->tensors[bias_tensor_index];
  auto& bias_buffer = model_t->buffers[bias_tensor->buffer];
  if (!bias.empty()) {
    CHECK_EQ(bias.size() * sizeof(bias.data()[0]), bias_buffer->data.size());
    std::memcpy(bias_buffer->data.data(), bias.data(), bias.size());
  } else {
    std::memset(bias_buffer->data.data(), 0, bias_buffer->data.size());
  }
}

std::vector<int> CalculateLinearLayerOutputShape(
    const std::vector<int>& input_shape, const std::vector<int>& kernel_shape) {
  std::vector<int> output_shape;
  for (auto it = input_shape.begin(); it != input_shape.end() - 1; ++it) {
    output_shape.push_back(*it);
  }
  output_shape.push_back(kernel_shape.front());
  return output_shape;
}

}  // namespace internal

int FindOpcodeIndex(
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& opcodes,
    const tflite::BuiltinOperator target_op, const std::string& custom_code) {
  for (int i = 0; i < opcodes.size(); ++i) {
    if (opcodes[i]->builtin_code == target_op &&
        (opcodes[i]->builtin_code != tflite::BuiltinOperator_CUSTOM ||
         opcodes[i]->custom_code == custom_code)) {
      return i;
    }
  }
  return -1;
}

std::vector<int> FindOperatorsWithInput(const tflite::BuiltinOperator target_op,
                                        const int input_tensor_index,
                                        const tflite::ModelT* model_t,
                                        const int base_op_index) {
  CHECK(model_t);
  CHECK_GE(base_op_index, 0);
  const auto& ops = model_t->subgraphs[0]->operators;
  const auto& opcodes = model_t->operator_codes;
  std::vector<int> operator_indices;
  int op_index = base_op_index;
  while (op_index < ops.size()) {
    const auto& op = ops[op_index];
    if (op->inputs[0] == input_tensor_index &&
        opcodes[op->opcode_index]->builtin_code == target_op) {
      operator_indices.push_back(op_index);
    }
    ++op_index;
  }
  return operator_indices;
}

int FindSingleOperatorWithInput(const tflite::BuiltinOperator target_op,
                                const int input_tensor_index,
                                const tflite::ModelT* model_t,
                                const int base_op_index) {
  const auto& op_indices = FindOperatorsWithInput(target_op, input_tensor_index,
                                                  model_t, base_op_index);
  if (op_indices.empty() || op_indices.size() > 1) return -1;
  return op_indices[0];
}

std::vector<int> FindOperators(const tflite::BuiltinOperator target_op,
                               const tflite::ModelT* model_t) {
  CHECK(model_t);
  const auto& ops = model_t->subgraphs[0]->operators;
  const auto& opcodes = model_t->operator_codes;
  std::vector<int> operator_indices;
  for (int i = 0; i < ops.size(); ++i) {
    if (opcodes[ops[i]->opcode_index]->builtin_code == target_op) {
      operator_indices.push_back(i);
    }
  }
  return operator_indices;
}

int FindSingleOperator(const tflite::BuiltinOperator target_op,
                       const tflite::ModelT* model_t) {
  const auto& op_indices = FindOperators(target_op, model_t);
  if (op_indices.empty() || op_indices.size() > 1) return -1;
  return op_indices[0];
}

std::unique_ptr<tflite::QuantizationParametersT> CreateQuantParam(
    const std::vector<float>& min, const std::vector<float>& max,
    const std::vector<float>& scale, const std::vector<int64_t>& zero_point) {
  auto result = absl::make_unique<tflite::QuantizationParametersT>();
  for (const auto& v : min) {
    result->min.push_back(v);
  }
  for (const auto& v : max) {
    result->max.push_back(v);
  }
  for (const auto& v : scale) {
    result->scale.push_back(v);
  }
  for (const auto& v : zero_point) {
    result->zero_point.push_back(v);
  }
  return result;
}

std::vector<tflite::TensorT*> GetGraphOutputTensors(
    const tflite::ModelT* model_t) {
  CHECK(model_t);
  CHECK_EQ(model_t->subgraphs.size(), 1);
  std::vector<tflite::TensorT*> output_tensors;
  const auto& subgraph = model_t->subgraphs[0];
  // Get current graph output.
  const auto& graph_outputs = subgraph->outputs;
  CHECK_GT(graph_outputs.size(), 0);
  for (auto graph_output_tensor_index : graph_outputs) {
    const auto& graph_output_tensor =
        subgraph->tensors[graph_output_tensor_index];
    output_tensors.push_back(graph_output_tensor.get());
  }
  return output_tensors;
}

std::unique_ptr<flatbuffers::FlatBufferBuilder> GetFlatBufferBuilder(
    const tflite::ModelT* model_t) {
  auto fbb = absl::make_unique<flatbuffers::FlatBufferBuilder>();
  auto output_model_location = tflite::Model::Pack(*fbb, model_t);
  tflite::FinishModelBuffer(*fbb, output_model_location);
  return fbb;
}

EdgeTpuApiStatus AppendFullyConnectedAndSoftmaxLayerToModel(
    const std::string& in_model_path, const std::string& out_model_path,
    const float* weights, size_t weights_size, const float* biases,
    size_t biases_size, float out_tensor_min, float out_tensor_max,
    EdgeTpuErrorReporter* reporter) {
  // Read input model.
  std::string input_model_content;
  EDGETPU_API_ENSURE_STATUS(
      ReadFile(in_model_path, &input_model_content, reporter));
  VLOG(1) << "Input model size in bytes: " << input_model_content.size();
  const tflite::Model* model = tflite::GetModel(input_model_content.data());
  LOG_IF(FATAL, model == nullptr) << "Failed to read model.";
  auto model_t = absl::WrapUnique<tflite::ModelT>(model->UnPack());

  // Get last tensor of input model.
  std::vector<tflite::TensorT*> graph_output_tensors =
      GetGraphOutputTensors(model_t.get());
  auto* embedding_output_tensor = graph_output_tensors[0];
  auto output_tensor_shape = embedding_output_tensor->shape;

  auto embedding_vector_dim = *(output_tensor_shape.end() - 1);

  // Get quantization parameter for weights, biases and output tensor of FC.
  auto quant_params = CalculateQuantParamsLinearLayer(
      weights, weights_size, biases, biases_size,
      *(embedding_output_tensor->quantization), out_tensor_min, out_tensor_max);

  // Quantize weights and biases.
  auto weights_quant =
      Quantize<uint8_t>(weights, weights_size, quant_params[0]->scale[0],
                        quant_params[0]->zero_point[0]);
  auto biases_quant =
      Quantize<int32_t>(biases, biases_size, quant_params[1]->scale[0],
                        quant_params[1]->zero_point[0]);

  // Append operators.
  int fc_op_index, reshape_op_index, softmax_op_index;

  EDGETPU_API_ENSURE_STATUS(internal::AppendLinearLayer(
      /*kernel_shape=*/{static_cast<int>(weights_size) / embedding_vector_dim,
                        embedding_vector_dim},
      std::move(quant_params), model_t.get(), &fc_op_index, reporter));

  if (output_tensor_shape.size() == 4) {
    EDGETPU_API_ENSURE_STATUS(
        internal::AppendReshape(model_t.get(), &reshape_op_index, reporter));
  }

  EDGETPU_API_ENSURE_STATUS(
      internal::AppendSoftmax(model_t.get(), &softmax_op_index, reporter));

  // Fill weights.
  internal::SetLinearParams(weights_quant, biases_quant, fc_op_index,
                            model_t.get());

  // Convert from tflite::ModelT format to FlatBufferBuilder.
  auto fbb = GetFlatBufferBuilder(model_t.get());
  VLOG(1) << "Output model size in bytes: " << fbb->GetSize();
  return WriteFile(
      std::string(reinterpret_cast<const char*>(fbb->GetBufferPointer()),
                  fbb->GetSize()),
      out_model_path, reporter);
}

}  // namespace learn
}  // namespace coral
