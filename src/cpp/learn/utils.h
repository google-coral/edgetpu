#ifndef EDGETPU_CPP_LEARN_UTILS_H_
#define EDGETPU_CPP_LEARN_UTILS_H_

#include <cmath>
#include <numeric>
#include <vector>

#include "absl/memory/memory.h"
#include "glog/logging.h"
#include "src/cpp/error_reporter.h"
#include "src/cpp/utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace learn {

namespace internal {

// Computes the sqaure sum of a vector.
inline float SquaredSum(const std::vector<float>& embedding) {
  return std::inner_product(embedding.begin(), embedding.end(),
                            embedding.begin(), 0.0f);
}

// Computes the l2-normalized value of a vector.
inline float L2Norm(const std::vector<float>& embedding) {
  return std::sqrt(SquaredSum(embedding));
}

// NOTE: all of the following AppendXXX functions are tuned for last layer
// backprop method, especially quantization parameters. You should adapt the
// implementation accordingly if used in other cases.

// Appends L2Normalization. Returns index of the L2Norm operator in subgraph.
EdgeTpuApiStatus AppendL2Norm(tflite::ModelT* model_t, int* new_op_index,
                              EdgeTpuErrorReporter* reporter);

// Appends Conv2d / FC layer. Returns index of the Conv2d / FC operator in
// subgraph. It distinguishes the two operatos using the size of kernel_shape,
// where size of 4 indicates a Conv while size of 2 indicates a FC.
//
// |quant_params| contains the quantization parameters for kernel weights,
// biases and output tensor.
EdgeTpuApiStatus AppendLinearLayer(
    const std::vector<int>& kernel_shape,
    std::vector<std::unique_ptr<tflite::QuantizationParametersT>> quant_params,
    tflite::ModelT* model_t, int* new_op_index, EdgeTpuErrorReporter* reporter);

// Appends Reshape. Returns index of the Reshape operator in subgraph.
EdgeTpuApiStatus AppendReshape(tflite::ModelT* model_t, int* new_op_index,
                               EdgeTpuErrorReporter* reporter);

// Appends Softmax. Returns index of the Softmax operator in subgraph.
EdgeTpuApiStatus AppendSoftmax(tflite::ModelT* model_t, int* new_op_index,
                               EdgeTpuErrorReporter* reporter);

// Returns index of a tensor specified by name. If non-found, return -1.
int FindTensor(const std::string& name, const tflite::SubGraphT& subgraph_t);

// Creates and appends buffer to model. Returns new buffer index.
int AppendBuffer(int buffer_size_bytes, tflite::ModelT* model_t);

// Appends tensor to subgraph and returns new tensor's index.
int AppendTensor(const std::vector<int>& shape, const std::string& name,
                 int buffer_index, tflite::TensorType type,
                 std::unique_ptr<tflite::QuantizationParametersT> q_param,
                 tflite::SubGraphT* subgraph);

enum class TensorLocation {
  // Refers intermediate tensor, input of an operator.
  kInput,
  // Refers intermediate tensor, output of an operator.
  kOutput,
  // Refers parameter tensor, e.g., kernel of convolution.
  kParameter,
};

struct TensorConfig {
  std::string name;
  tflite::TensorType type;
  TensorLocation location;
  std::vector<int> shape;
  tflite::QuantizationParametersT* quant;

  TensorConfig(const std::string& name, tflite::TensorType type,
               TensorLocation location, const std::vector<int>& shape,
               tflite::QuantizationParametersT* quant)
      : name(name),
        type(type),
        location(location),
        shape(shape),
        quant(quant) {}
};

// Appends an operator to model. Returns index of the new operator in subgraph.
// |tensor_configs| should only contains parameter tensors and output tensors
// for the new operator. It assumes the input of the new operator is the first
// output of the graph, and the output of the new operator is the new first
// output tensor of the graph. Does not support custom operator.
EdgeTpuApiStatus AppendOperator(const std::vector<TensorConfig>& tensor_configs,
                                tflite::BuiltinOperator op_type,
                                tflite::ModelT* model_t, int* new_op_index,
                                EdgeTpuErrorReporter* reporter);

// Sets Conv2d / FC parameters, i.e., kernel and bias.
// Bias will be set to zeros if `bias` is set to empty.
//
// Note on weights data ordering.
// "Typical TF Lite weights are [filter_count, filter_height, filter_width,
// input_depth]". See comments inside `AllocateTemporaryTensorsIfRequired` in
// //depot/google3/third_party/tensorflow/lite/kernels/conv.cc
void SetLinearParams(const std::vector<uint8_t>& kernel,
                     const std::vector<int32_t>& bias, int op_index,
                     tflite::ModelT* model_t);

// Calculates the shape of conv/fc's output, given shape of input tensor and
// kernel tensor.
std::vector<int> CalculateLinearLayerOutputShape(
    const std::vector<int>& input_shape, const std::vector<int>& kernel_shape);

}  // namespace internal

// Quantize(), Dequantize() and QuantizationParams() come from
// tensorflow/lite/kernels/test_util.h
//
// Quantizes float type data to target type, e.g., uint8_t, or int32_t.
template <typename T>
inline std::vector<T> Quantize(const float* data, int data_size, float scale,
                               int32_t zero_point) {
  std::vector<T> q;
  q.reserve(data_size);
  for (int i = 0; i < data_size; ++i) {
    q.push_back(static_cast<T>(std::max<float>(
        std::numeric_limits<T>::min(),
        std::min<float>(std::numeric_limits<T>::max(),
                        std::round(zero_point + (data[i] / scale))))));
  }
  return q;
}

// Similar to above function, but takes a vector of float data.
template <typename T>
inline std::vector<T> Quantize(const std::vector<float>& data, float scale,
                               int32_t zero_point) {
  return Quantize<T>(data.data(), data.size(), scale, zero_point);
}

template <typename T>
inline std::vector<float> Dequantize(const std::vector<T>& data, float scale,
                                     int32_t zero_point) {
  std::vector<float> f;
  f.reserve(data.size());
  for (T q : data) {
    f.push_back(scale * (q - zero_point));
  }
  return f;
}

// Calculates scale and zero point, given min, max range and target data type T.
template <typename T>
std::pair<float, int32_t> QuantizationParams(float f_min, float f_max) {
  int32_t zero_point = 0;
  float scale = 0;
  const T qmin = std::numeric_limits<T>::min();
  const T qmax = std::numeric_limits<T>::max();
  const float qmin_double = qmin;
  const float qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  CHECK_LE(f_min, 0);
  CHECK_GE(f_max, 0);
  if (f_min == f_max) {
    // Special case where the min,max range is a point. Should be {0}.
    CHECK_EQ(f_min, 0);
    CHECK_EQ(f_max, 0);
    return {scale, zero_point};
  }

  // General case.
  //
  // First determine the scale.
  scale = (f_max - f_min) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const float zero_point_from_min = qmin_double - f_min / scale;
  const float zero_point_from_max = qmax_double - f_max / scale;

  const float zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(f_min / scale);

  const float zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(f_max / scale);

  const float zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  //  padding).

  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(std::round(zero_point_double));
  }

  // The zero point should always be in the range of quantized value,
  // // [qmin, qmax].
  CHECK_GE(nudged_zero_point, qmin);
  CHECK_LE(nudged_zero_point, qmax);

  zero_point = nudged_zero_point;
  // finally, return the values
  return {scale, zero_point};
}

// L2-normalizes a vector.
inline std::vector<float> L2Normalize(const std::vector<float>& embedding) {
  std::vector<float> normalized_embedding(embedding.size(), 0.f);
  const float norm = internal::L2Norm(embedding);
  if (std::fabs(norm) > 1e-5) {
    std::transform(embedding.begin(), embedding.end(),
                   normalized_embedding.begin(),
                   [norm](const float value) { return value / norm; });
  }
  return normalized_embedding;
}

// Finds the opcode index of target operator. Returns -1 if `target_op` does not
// exist in `opcodes`. For custom operator, custom code must match as well.
int FindOpcodeIndex(
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& opcodes,
    const tflite::BuiltinOperator target_op, const std::string& custom_code);

// Returns the indices of operators specified by operator code with given tensor
// as their inputs[0]. It is counted from base_op_index.
std::vector<int> FindOperatorsWithInput(const tflite::BuiltinOperator target_op,
                                        const int input_tensor_index,
                                        const tflite::ModelT* model_t,
                                        const int base_op_index = 0);

// Returns the index of the single operator specified by operator code with
// given tensor as their inputs[0]. It is counted from base_op_index.
// Returns -1 if the operator can not be found or there are multiple matches.
int FindSingleOperatorWithInput(const tflite::BuiltinOperator target_op,
                                const int input_tensor_index,
                                const tflite::ModelT* model_t,
                                const int base_op_index);

// Returns indices of operators specified by operator code.
std::vector<int> FindOperators(const tflite::BuiltinOperator target_op,
                               const tflite::ModelT* model_t);

// Returns the index of the single operator specified by operator code.
// Returns -1 if the operator can not be found or there are multiple matches.
int FindSingleOperator(const tflite::BuiltinOperator target_op,
                       const tflite::ModelT* model_t);

// Creates quantization parameters.
std::unique_ptr<tflite::QuantizationParametersT> CreateQuantParam(
    const std::vector<float>& min, const std::vector<float>& max,
    const std::vector<float>& scale, const std::vector<int64_t>& zero_point);

// Returns vector of pointers to graph output tensors.
std::vector<tflite::TensorT*> GetGraphOutputTensors(
    const tflite::ModelT* model_t);

// Gets FlatBufferBuilder based on ModelT type.
std::unique_ptr<flatbuffers::FlatBufferBuilder> GetFlatBufferBuilder(
    const tflite::ModelT* model_t);

// Appends Fully-Connected (FC) layer and softmax layer to tflite model.
//
// This function does the following:
//   1) Read tflite model from |in_model_path| as input;
//        input model is assumed to be an embedding extractor, e.g., a
//        classification model without the last FC+Softmax layer.
//   2) Append (learned) weights and biases as a FC/Conv layer to input modell
//        the appended operator is decided by the output shape of tflite model.
//        If the output is a 2D tensor, FC would be appended, otherwise Conv
//        would be appended.
//   3) Append softmax layer after the FC/Conv layer;
//   4) Save tflite model to |out_model_path|;
EdgeTpuApiStatus AppendFullyConnectedAndSoftmaxLayerToModel(
    const std::string& in_model_path, const std::string& out_model_path,
    const float* weights, size_t weights_size, const float* biases,
    size_t biases_size, float out_tensor_min, float out_tensor_max,
    EdgeTpuErrorReporter* reporter);

}  // namespace learn
}  // namespace coral

#endif  // EDGETPU_CPP_LEARN_UTILS_H_
