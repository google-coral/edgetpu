#include "src/cpp/basic/basic_engine_native.h"

#include <sys/stat.h>

#include <chrono>  // NOLINT(build/c++11)
#include <numeric>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "edgetpu.h"
#include "src/cpp/error_reporter.h"
#include "src/cpp/posenet/posenet_decoder_op.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

namespace coral {
namespace {
using tflite::ops::builtin::BuiltinOpResolver;

#define BASIC_ENGINE_NATIVE_ENSURE(condition, msg) \
  EDGETPU_API_REPORT_ERROR(error_reporter_, !(condition), msg)

#define BASIC_ENGINE_NATIVE_ENSURE_WITH_ARGS(condition, msg, ...)        \
  EDGETPU_API_REPORT_ERROR_WITH_ARGS(error_reporter_, !(condition), msg, \
                                     __VA_ARGS__)

#define BASIC_ENGINE_INIT_CHECK()                                             \
  BASIC_ENGINE_NATIVE_ENSURE(                                                 \
      is_initialized_,                                                        \
      "BasicEngineNative must be initialized! Please ensure the instance is " \
      "created by BasicEngineNativeBuilder!")

TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
  if (!src) return nullptr;
  TfLiteFloatArray* ret = static_cast<TfLiteFloatArray*>(
      malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
  if (!ret) return nullptr;
  ret->size = src->size;
  std::memcpy(ret->data, src->data, src->size * sizeof(float));
  return ret;
}
}  // namespace

BasicEngineNative::BasicEngineNative() {
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
  is_initialized_ = false;
}

BasicEngineNative::~BasicEngineNative() {
  // EdgeTpuResource must be destructed after interpreter_. Because the edgetpu
  // context will be used by destructor of Custom Op.
  interpreter_.reset();
  edgetpu_resource_.reset();
}

EdgeTpuApiStatus BasicEngineNative::BuildModelFromFile(
    const std::string& model_path) {
  model_path_ = model_path;
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str(),
                                                  error_reporter_.get());
  EDGETPU_API_ENSURE(model_);
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::InitializeEdgeTpuResource(
    const std::string& device_path) {
  EdgeTpuApiStatus status =
      device_path.empty()
          ? EdgeTpuResourceManager::GetSingleton()->GetEdgeTpuResource(
                &edgetpu_resource_)
          : EdgeTpuResourceManager::GetSingleton()->GetEdgeTpuResource(
                device_path, &edgetpu_resource_);
  if (status == kEdgeTpuApiError) {
    error_reporter_->Report(
        EdgeTpuResourceManager::GetSingleton()->get_error_message());
  }
  return status;
}

EdgeTpuApiStatus BasicEngineNative::CreateInterpreterWithResolver(
    BuiltinOpResolver* resolver) {
  BuiltinOpResolver new_resolver;

  BuiltinOpResolver* effective_resolver =
      (resolver == nullptr ? &new_resolver : resolver);
  effective_resolver->AddCustom(edgetpu::kCustomOp,
                                edgetpu::RegisterCustomOp());
  effective_resolver->AddCustom(kPosenetDecoderOp, RegisterPosenetDecoderOp());

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(
      model_->GetModel(), *effective_resolver, error_reporter_.get());
  if (interpreter_builder(&interpreter_) != kTfLiteOk) {
    error_reporter_->Report("Error in interpreter initialization.");
    return kEdgeTpuApiError;
  }
  // Bind given context with interpreter.
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext,
                                   edgetpu_resource_->context());
  interpreter_->SetNumThreads(1);
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    error_reporter_->Report("Failed to allocate tensors.");
    return kEdgeTpuApiError;
  }

  EDGETPU_API_ENSURE(interpreter_);
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::InitializeInputAndOutput() {
  BASIC_ENGINE_NATIVE_ENSURE(!interpreter_->inputs().empty(),
                             "Invalid model, no input tensor!");
  BASIC_ENGINE_NATIVE_ENSURE(interpreter_->inputs().size() == 1,
                             "We don't support multiple input tensors yet!");
  // Calculate input size of the model.
  const auto& dimensions =
      interpreter_->tensor(interpreter_->inputs()[0])->dims;
  input_array_size_ = 1;
  BASIC_ENGINE_NATIVE_ENSURE(dimensions->size > 0,
                             "Number of input tensor's dimensions must > 0!");
  // Allocate memory for input tensor shape.
  input_tensor_shape_.resize(dimensions->size);
  for (int i = 0; i < dimensions->size; ++i) {
    input_tensor_shape_[i] = dimensions->data[i];
    input_array_size_ *= input_tensor_shape_[i];
  }
  BASIC_ENGINE_NATIVE_ENSURE(input_array_size_ > 0,
                             "Size of input array(Model's input) must > 0!");

  // Allocate memory for output tensors.
  const std::vector<int>& indices = interpreter_->outputs();
  output_tensor_sizes_.resize(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    const auto* output_tensor = interpreter_->tensor(indices[i]);
    if (output_tensor->type == kTfLiteUInt8) {
      output_tensor_sizes_[i] = output_tensor->bytes;
    } else if (output_tensor->type == kTfLiteFloat32) {
      output_tensor_sizes_[i] = output_tensor->bytes / sizeof(float);
    } else if (output_tensor->type == kTfLiteInt64) {
      output_tensor_sizes_[i] = output_tensor->bytes / sizeof(int64_t);
    } else {
      BASIC_ENGINE_NATIVE_ENSURE(
          false,
          absl::Substitute(
              "Output tensor type not supported! output_tensor->type = ($0)!",
              output_tensor->type));
    }
  }
  output_array_size_ = std::accumulate(output_tensor_sizes_.begin(),
                                       output_tensor_sizes_.end(), 0);
  inference_result_.resize(output_array_size_);
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::Init(
    std::unique_ptr<tflite::FlatBufferModel> model,
    BuiltinOpResolver* resolver) {
  model_ = std::move(model);
  EDGETPU_API_ENSURE_STATUS(InitializeEdgeTpuResource(""));
  EDGETPU_API_ENSURE_STATUS(CreateInterpreterWithResolver(resolver));
  EDGETPU_API_ENSURE_STATUS(InitializeInputAndOutput());
  is_initialized_ = true;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::Init(const std::string& model_path,
                                         const std::string& device_path) {
  EDGETPU_API_ENSURE_STATUS(BuildModelFromFile(model_path));
  EDGETPU_API_ENSURE_STATUS(InitializeEdgeTpuResource(device_path));
  EDGETPU_API_ENSURE_STATUS(CreateInterpreterWithResolver(nullptr));
  EDGETPU_API_ENSURE_STATUS(InitializeInputAndOutput());
  is_initialized_ = true;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::RunInference(const uint8_t* const input,
                                                 const size_t in_size,
                                                 float const** const output,
                                                 size_t* const out_size) {
  BASIC_ENGINE_INIT_CHECK();
  BASIC_ENGINE_NATIVE_ENSURE(
      in_size >= input_array_size_,
      absl::Substitute(
          "Input buffer size $0 smaller than model input tensor size $1.",
          in_size, input_array_size_));

  const auto& start_time = std::chrono::steady_clock::now();

  // Set input tensor to use input buffer and invoke.
  const int input_tensor_index = interpreter_->inputs()[0];
  const TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_index);
  const TfLiteType input_type = input_tensor->type;
  const char* input_name = input_tensor->name;
  std::vector<int> input_dims(
      input_tensor->dims->data,
      input_tensor->dims->data + input_tensor->dims->size);
  if (input_tensor->quantization.type == kTfLiteNoQuantization) {
    // Deal with legacy model with old quantization parameters.
    EDGETPU_API_ENSURE(interpreter_->SetTensorParametersReadOnly(
                           input_tensor_index, input_type, input_name,
                           input_dims, input_tensor->params,
                           reinterpret_cast<const char*>(input),
                           std::min(in_size, input_array_size_)) == kTfLiteOk);
  } else {
    // For models with new quantization parameters, deep copy the parameters.
    EDGETPU_API_ENSURE(input_tensor->quantization.type ==
                       kTfLiteAffineQuantization);
    EDGETPU_API_ENSURE(input_tensor->quantization.params);
    TfLiteQuantization input_quant_clone = input_tensor->quantization;
    const TfLiteAffineQuantization* input_quant_params =
        reinterpret_cast<TfLiteAffineQuantization*>(
            input_tensor->quantization.params);
    // |input_quant_params_clone| will be owned by |input_quant_clone|, and will
    // be deallocated by free(). Therefore malloc is used to allocate its
    // memory here.
    TfLiteAffineQuantization* input_quant_params_clone =
        reinterpret_cast<TfLiteAffineQuantization*>(
            malloc(sizeof(TfLiteAffineQuantization)));
    input_quant_params_clone->scale =
        TfLiteFloatArrayCopy(input_quant_params->scale);
    EDGETPU_API_ENSURE(input_quant_params_clone->scale);
    input_quant_params_clone->zero_point =
        TfLiteIntArrayCopy(input_quant_params->zero_point);
    EDGETPU_API_ENSURE(input_quant_params_clone->zero_point);
    input_quant_params_clone->quantized_dimension =
        input_quant_params->quantized_dimension;
    input_quant_clone.params = input_quant_params_clone;
    EDGETPU_API_ENSURE(interpreter_->SetTensorParametersReadOnly(
                           input_tensor_index, input_type, input_name,
                           input_dims, input_quant_clone,
                           reinterpret_cast<const char*>(input),
                           std::min(in_size, input_array_size_)) == kTfLiteOk);
  }
  uint8_t* input_tensor_ptr = interpreter_->typed_input_tensor<uint8_t>(0);
  BASIC_ENGINE_NATIVE_ENSURE(input_tensor_ptr == input,
                             "Input tensor does not reuse the given buffer!");

  EDGETPU_API_ENSURE(interpreter_->Invoke() == kTfLiteOk);

  EDGETPU_API_ENSURE_STATUS(ParseAndCopyInferenceResults(output, out_size));
  std::chrono::duration<double, std::milli> time_span =
      std::chrono::steady_clock::now() - start_time;
  inference_time_ = time_span.count();

  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::RunInference(const float* const input,
                                                 const size_t in_size,
                                                 float const** const output,
                                                 size_t* const out_size) {
  BASIC_ENGINE_INIT_CHECK();
  BASIC_ENGINE_NATIVE_ENSURE(
      in_size >= input_array_size_,
      absl::Substitute(
          "Input buffer size $0 smaller than model input tensor size $1.",
          in_size, input_array_size_));

  const auto& start_time = std::chrono::steady_clock::now();

  // Set input tensor to use input buffer and invoke.
  const int input_tensor_index = interpreter_->inputs()[0];
  const TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_index);
  const TfLiteType input_type = input_tensor->type;
  const char* input_name = input_tensor->name;
  std::vector<int> input_dims(
      input_tensor->dims->data,
      input_tensor->dims->data + input_tensor->dims->size);

  EDGETPU_API_ENSURE(interpreter_->SetTensorParametersReadOnly(
                         input_tensor_index, input_type, input_name, input_dims,
                         input_tensor->params,
                         reinterpret_cast<const char*>(input),
                         in_size * sizeof(float)) == kTfLiteOk);

  float* input_tensor_ptr = interpreter_->typed_input_tensor<float>(0);
  BASIC_ENGINE_NATIVE_ENSURE(input_tensor_ptr == input,
                             "Input tensor does not reuse the given buffer!");

  EDGETPU_API_ENSURE(interpreter_->Invoke() == kTfLiteOk);

  EDGETPU_API_ENSURE_STATUS(ParseAndCopyInferenceResults(output, out_size));
  std::chrono::duration<double, std::milli> time_span =
      std::chrono::steady_clock::now() - start_time;
  inference_time_ = time_span.count();

  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::ParseAndCopyInferenceResults(
    float const** const output, size_t* const out_size) {
  // Parse results.
  const auto& output_indices = interpreter_->outputs();
  const int num_outputs = output_indices.size();
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = interpreter_->tensor(output_indices[i]);
    BASIC_ENGINE_NATIVE_ENSURE_WITH_ARGS(out_tensor, "Tensor %d doesn't exist!",
                                         output_indices[i]);

    if (out_tensor->type == kTfLiteUInt8) {
      const int num_values = out_tensor->bytes;
      const uint8_t* output = interpreter_->typed_output_tensor<uint8_t>(i);
      BASIC_ENGINE_NATIVE_ENSURE_WITH_ARGS(output, "Tensor %s == nullptr",
                                           out_tensor->name);
      for (int j = 0; j < num_values; ++j) {
        inference_result_[out_idx++] =
            (output[j] - out_tensor->params.zero_point) *
            out_tensor->params.scale;
      }
    } else if (out_tensor->type == kTfLiteFloat32) {
      const int num_values = out_tensor->bytes / sizeof(float);
      const float* output = interpreter_->typed_output_tensor<float>(i);
      BASIC_ENGINE_NATIVE_ENSURE_WITH_ARGS(output, "Tensor %s == nullptr",
                                           out_tensor->name);
      for (int j = 0; j < num_values; ++j) {
        inference_result_[out_idx++] = output[j];
      }
    } else if (out_tensor->type == kTfLiteInt64) {
      const int num_values = out_tensor->bytes / sizeof(int64_t);
      const int64_t* output = interpreter_->typed_output_tensor<int64_t>(i);
      BASIC_ENGINE_NATIVE_ENSURE_WITH_ARGS(output, "Tensor %s == nullptr",
                                           out_tensor->name);
      for (int j = 0; j < num_values; ++j) {
        inference_result_[out_idx++] = output[j];
      }
    } else {
      error_reporter_->Report("Tensor %s has unsupported output type %d",
                              out_tensor->name, out_tensor->type);
      return kEdgeTpuApiError;
    }
  }
  BASIC_ENGINE_NATIVE_ENSURE(out_idx == output_array_size_,
                             "Abnormal output size!");
  *out_size = inference_result_.size();
  *output = inference_result_.data();
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::get_input_tensor_shape(
    int const** const dims, int* const dims_num) const {
  BASIC_ENGINE_INIT_CHECK();
  *dims_num = input_tensor_shape_.size();
  *dims = input_tensor_shape_.data();
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::get_input_array_size(
    size_t* array_size) const {
  BASIC_ENGINE_INIT_CHECK();
  *array_size = input_array_size_;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::get_all_output_tensors_sizes(
    size_t const** tensor_sizes, size_t* tensor_num) const {
  BASIC_ENGINE_INIT_CHECK();
  *tensor_num = output_tensor_sizes_.size();
  *tensor_sizes = output_tensor_sizes_.data();
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::get_num_of_output_tensors(
    size_t* output) const {
  BASIC_ENGINE_INIT_CHECK();
  *output = output_tensor_sizes_.size();
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::get_output_tensor_size(
    const int tensor_index, size_t* const output) const {
  BASIC_ENGINE_INIT_CHECK();
  BASIC_ENGINE_NATIVE_ENSURE(tensor_index >= 0, "tensor_index must >= 0!");
  BASIC_ENGINE_NATIVE_ENSURE(tensor_index < output_tensor_sizes_.size(),
                             "tensor_index doesn't exist!");
  *output = output_tensor_sizes_[tensor_index];
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::total_output_array_size(
    size_t* const output) const {
  BASIC_ENGINE_INIT_CHECK();
  *output = output_array_size_;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::get_raw_output(float const** output,
                                                   size_t* out_size) const {
  BASIC_ENGINE_INIT_CHECK();
  *out_size = inference_result_.size();
  *output = inference_result_.data();
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::model_path(std::string* path) const {
  BASIC_ENGINE_INIT_CHECK();
  BASIC_ENGINE_NATIVE_ENSURE(!model_path_.empty(), "No model path!");
  *path = model_path_;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::device_path(std::string* path) const {
  BASIC_ENGINE_INIT_CHECK();
  *path = edgetpu_resource_->path();
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus BasicEngineNative::get_inference_time(
    float* const time) const {
  BASIC_ENGINE_INIT_CHECK();
  *time = inference_time_;
  return kEdgeTpuApiOk;
}

std::string BasicEngineNative::get_error_message() {
  return error_reporter_->message();
}

BasicEngineNativeBuilder::BasicEngineNativeBuilder(
    const std::string& model_path)
    : model_path_(model_path), device_path_("") {
  read_from_file_ = true;
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
}

BasicEngineNativeBuilder::BasicEngineNativeBuilder(
    const std::string& model_path, const std::string& device_path)
    : model_path_(model_path), device_path_(device_path) {
  read_from_file_ = true;
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
}

BasicEngineNativeBuilder::BasicEngineNativeBuilder(
    std::unique_ptr<tflite::FlatBufferModel> model)
    : model_(std::move(model)), resolver_(nullptr) {
  read_from_file_ = false;
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
}

BasicEngineNativeBuilder::BasicEngineNativeBuilder(
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<BuiltinOpResolver> resolver)
    : model_(std::move(model)), resolver_(std::move(resolver)) {
  read_from_file_ = false;
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
}

EdgeTpuApiStatus BasicEngineNativeBuilder::operator()(
    std::unique_ptr<BasicEngineNative>* engine) {
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, !engine,
      "Null output pointer passed to BasicEngineNativeBuilder!");
  *engine = absl::make_unique<BasicEngineNative>();
  if (read_from_file_) {
    EDGETPU_API_REPORT_ERROR(
        error_reporter_,
        (*engine)->Init(model_path_, device_path_) != kEdgeTpuApiOk,
        (*engine)->get_error_message());
  } else {
    EDGETPU_API_REPORT_ERROR(error_reporter_, !model_, "model_ is nullptr!");
    EDGETPU_API_REPORT_ERROR(
        error_reporter_,
        (*engine)->Init(std::move(model_), resolver_.get()) != kEdgeTpuApiOk,
        (*engine)->get_error_message());
  }
  return kEdgeTpuApiOk;
}

std::string BasicEngineNativeBuilder::get_error_message() {
  return error_reporter_->message();
}

}  // namespace coral
