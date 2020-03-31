#include "src/cpp/learn/imprinting/engine_native.h"

#include <cmath>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"
#include "src/cpp/learn/utils.h"
#include "tensorflow/lite/model.h"

namespace coral {
namespace learn {
namespace imprinting {

namespace {
#define IMPRINTING_ENGINE_NATIVE_INIT_CHECK()                          \
  EDGETPU_API_REPORT_ERROR(                                            \
      error_reporter_, !is_initialized_,                               \
      "ImprintingEngineNative must be initialized! Please ensure the " \
      "instance is created by ImprintingEngineNativeBuilder!")
}  // namespace

void ImprintingEngineNative::ExtractModelTrainingMetadata() {
  metadata_ = {};
  if (model_t_->description.empty()) return;

  VLOG(1) << "Model description: " << model_t_->description;
  const std::vector<std::string> v =
      absl::StrSplit(model_t_->description, absl::ByAnyChar(" \n"));

  for (int i = 0; i < v.size() - 1; i += 2) {
    int label;
    float sqrt_sum;
    if (absl::SimpleAtoi(v[i], &label) &&
        absl::SimpleAtof(v[i + 1], &sqrt_sum)) {
      metadata_.insert({label, sqrt_sum});
    } else {
      metadata_ = {};
      return;
    }
  }
}

EdgeTpuApiStatus ImprintingEngineNative::UpdateModelTrainingMetaData() {
  EDGETPU_API_REPORT_ERROR(error_reporter_, !model_t_, "model_t == nullptr!");
  std::string description;
  for (const auto& entry : metadata_) {
    absl::StrAppend(&description,
                    absl::StrFormat("%d %f\n", entry.first, entry.second));
  }
  model_t_->description = description;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::PreprocessImprintingModel() {
  const int l2_norm_opcode_index = FindOpcodeIndex(
      model_t_->operator_codes, tflite::BuiltinOperator_L2_NORMALIZATION,
      /*custom_code=*/"");

  EDGETPU_API_REPORT_ERROR(
      error_reporter_, l2_norm_opcode_index == -1,
      "Unsupported model architecture. Input model must have an L2Norm layer.");

  // Validate the last 5 operators are L2Norm, Conv2d, Mul, Reshape,
  // Softmax.
  const auto& ops = model_t_->subgraphs[0]->operators;
  const auto& opcodes = model_t_->operator_codes;
  int index = ops.size() - 5;
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, index < 0,
      "Unsupported model architecture. Too few operators.");
  EDGETPU_API_REPORT_ERROR(
      error_reporter_,
      opcodes[ops[index]->opcode_index]->builtin_code !=
              tflite::BuiltinOperator_L2_NORMALIZATION ||
          opcodes[ops[index + 1]->opcode_index]->builtin_code !=
              tflite::BuiltinOperator_CONV_2D ||
          opcodes[ops[index + 2]->opcode_index]->builtin_code !=
              tflite::BuiltinOperator_MUL ||
          opcodes[ops[index + 3]->opcode_index]->builtin_code !=
              tflite::BuiltinOperator_RESHAPE ||
          opcodes[ops[index + 4]->opcode_index]->builtin_code !=
              tflite::BuiltinOperator_SOFTMAX,
      "The last 5 operators should be L2Norm, Conv2d, Mul, Reshape and "
      "Softmax");
  const auto& tensors = model_t_->subgraphs[0]->tensors;
  const auto& buffers = model_t_->buffers;
  if (keep_classes_) {
    // Keep a copy of weights.
    const auto& fc_op = ops[index + 1];
    const auto& kernel_buffer = buffers[tensors[fc_op->inputs[1]]->buffer];
    weights_.resize(kernel_buffer->data.size());
    std::memcpy(weights_.data(), kernel_buffer->data.data(),
                kernel_buffer->data.size());
    // Read metadata from model description.
    ExtractModelTrainingMetadata();
  } else {
    weights_.clear();
    metadata_.clear();
  }

  // Delete the classification output from model, insert the L2Norm output to
  // model outputs in order to extract embeddings.
  const auto& l2norm_op = ops[index];
  auto& outputs = model_t_->subgraphs[0]->outputs;
  classification_tensor_index_ = outputs[0];
  outputs.clear();
  outputs.push_back(l2norm_op->outputs[0]);
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::PostprocessImprintingModel() {
  EDGETPU_API_REPORT_ERROR(error_reporter_, !model_t_, "model_t == nullptr!");
  EDGETPU_API_REPORT_ERROR(error_reporter_, model_t_->subgraphs.size() != 1,
                           "subgraph number isn't 1!");

  auto& ops = model_t_->subgraphs[0]->operators;
  auto& tensors = model_t_->subgraphs[0]->tensors;
  auto& buffers = model_t_->buffers;

  int new_num_classes = weights_.size() / embedding_vector_dim_;
  // Modify FC shape and value.
  const int fc_op_index = ops.size() - 4;
  auto& fc_op = ops[fc_op_index];
  const auto& opcodes = model_t_->operator_codes;
  EDGETPU_API_REPORT_ERROR(error_reporter_,
                           opcodes[fc_op->opcode_index]->builtin_code !=
                               tflite::BuiltinOperator_CONV_2D,
                           "We only support Conv_2d as classification layer!");
  const int kernel_tensor_index = fc_op->inputs[1];
  const int bias_tensor_index = fc_op->inputs[2];
  const int output_tensor_index = fc_op->outputs[0];

  auto& kernel_tensor = tensors[kernel_tensor_index];
  auto& bias_tensor = tensors[bias_tensor_index];

  auto& kernel_buffer = buffers[kernel_tensor->buffer];
  kernel_buffer->data.resize(weights_.size());
  std::memcpy(kernel_buffer->data.data(), weights_.data(), weights_.size());
  kernel_tensor->shape[0] = new_num_classes;

  auto& bias_buffer = buffers[bias_tensor->buffer];
  bias_buffer->data.resize(new_num_classes * sizeof(int32_t));
  const auto bias_zero_point = bias_tensor->quantization->zero_point[0];
  EDGETPU_API_REPORT_ERROR(error_reporter_, bias_zero_point < 0,
                           "bias_zero_point < 0!");
  EDGETPU_API_REPORT_ERROR(error_reporter_, bias_zero_point > 255,
                           "bias_zero_point > 255!");
  std::memset(bias_buffer->data.data(), bias_zero_point,
              bias_buffer->data.size());
  bias_tensor->shape[0] = new_num_classes;

  auto& output_tensor = tensors[output_tensor_index];
  output_tensor->shape[3] = new_num_classes;
  // Modify quantization parameters to work for a new range, especailly for a
  // better one.
  output_tensor->quantization =
      CreateQuantParam(/*min=*/{-1.0f}, /*max=*/{1.0f},
                       /*scale=*/{1.0f / 128},
                       /*zero_point=*/{128});

  // Modify Mul shape.
  int mul_op_index = ops.size() - 3;
  auto& mul_op = ops[mul_op_index];
  const int mul_tensor_index = mul_op->outputs[0];
  auto& mul_tensor = tensors[mul_tensor_index];
  mul_tensor->shape[3] = new_num_classes;
  const int scale_factor_tensor_index = mul_op->inputs[1];
  auto& scale_factor_tensor = tensors[scale_factor_tensor_index];
  auto& scale_factor_buffer = buffers[scale_factor_tensor->buffer];
  float scale_factor = Dequantize(
      scale_factor_buffer->data,
      /*scale=*/scale_factor_tensor->quantization->scale[0],
      /*zero_point=*/scale_factor_tensor->quantization->zero_point[0])[0];
  mul_tensor->quantization =
      CreateQuantParam(/*min=*/{-scale_factor}, /*max=*/{scale_factor},
                       /*scale=*/{1.0f / 128 * scale_factor},
                       /*zero_point=*/{128});

  // Modify Reshape value.
  int reshape_op_index = ops.size() - 2;
  auto& reshape_op = ops[reshape_op_index];
  const int reshape_tensor_index = reshape_op->outputs[0];

  auto& reshape_tensor = tensors[reshape_tensor_index];
  reshape_tensor->shape.back() = new_num_classes;

  // There can be two types of reshape op for tflite.
  // - If there is a second input tensor, it indicates the reshape's shape.
  // - If there is only one input tensor, the operator will compare the input
  // tensor shape and the output tensor shape.
  if (reshape_op->inputs.size() == 2) {
    const int reshape_shape_tensor_index = reshape_op->inputs[1];
    auto& reshape_shape_tensor = tensors[reshape_shape_tensor_index];
    auto& reshape_shape_buffer = buffers[reshape_shape_tensor->buffer];
    reinterpret_cast<int32_t*>(reshape_shape_buffer->data.data())[1] =
        new_num_classes;
  }
  auto* reshape_option_t = reinterpret_cast<tflite::ReshapeOptionsT*>(
      reshape_op->builtin_options.value);
  auto& new_shape = reshape_option_t->new_shape;
  new_shape.back() = new_num_classes;

  reshape_tensor->quantization =
      CreateQuantParam(/*min=*/{-scale_factor}, /*max=*/{scale_factor},
                       /*scale=*/{1.0f / 128 * scale_factor},
                       /*zero_point=*/{128});

  // Modify Softmax shape.
  const int softmax_op_index = ops.size() - 1;
  auto& softmax_op = ops[softmax_op_index];
  const int softmax_tensor_index = softmax_op->outputs[0];
  auto& softmax_tensor = tensors[softmax_tensor_index];
  softmax_tensor->shape[1] = new_num_classes;

  // Update the number of classes in ImprintingEngine.
  num_classes_ = new_num_classes;

  // Remove the existing output, add the classification outputs back.
  auto& outputs = model_t_->subgraphs[0]->outputs;
  outputs.clear();
  outputs.push_back(classification_tensor_index_);
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::Init(const std::string& model_path,
                                              bool keep_classes) {
  keep_classes_ = keep_classes;
  std::string input_model_content;
  EDGETPU_API_ENSURE_STATUS(
      ReadFile(model_path, &input_model_content, error_reporter_.get()));
  VLOG(1) << "Input model size in bytes: " << input_model_content.size();
  const tflite::Model* model = tflite::GetModel(input_model_content.data());
  EDGETPU_API_REPORT_ERROR(error_reporter_, !model,
                           "Failed to parse input model.");
  model_t_ = absl::WrapUnique<tflite::ModelT>(model->UnPack());

  std::vector<tflite::TensorT*> graph_output_tensors =
      GetGraphOutputTensors(model_t_.get());

  EDGETPU_API_REPORT_ERROR(error_reporter_, graph_output_tensors.empty(),
                           "Number of graph output_tensor < 1!");
  auto* logit_output_tensor = graph_output_tensors[0];
  EDGETPU_API_REPORT_ERROR(error_reporter_,
                           logit_output_tensor->shape.size() != 2 ||
                               logit_output_tensor->shape[0] != 1,
                           "Logit output tensor should be [1, x]");
  num_classes_ = logit_output_tensor->shape[1];

  EDGETPU_API_ENSURE_STATUS(PreprocessImprintingModel());
  // Get embedding vector dimension.
  graph_output_tensors = GetGraphOutputTensors(model_t_.get());
  auto* embedding_output_tensor = graph_output_tensors[0];
  EDGETPU_API_REPORT_ERROR(
      error_reporter_,
      embedding_output_tensor->shape.size() != 4 ||
          embedding_output_tensor->shape[0] != 1 ||
          embedding_output_tensor->shape[1] != 1 ||
          embedding_output_tensor->shape[2] != 1,
      "Embedding extractor's output tensor should be [1, 1, 1, x]");
  embedding_vector_dim_ = embedding_output_tensor->shape[3];

  // Get fc kernel quantization parameter directly.
  auto& ops = model_t_->subgraphs[0]->operators;
  auto& tensors = model_t_->subgraphs[0]->tensors;
  const int fc_op_index = ops.size() - 4;
  EDGETPU_API_REPORT_ERROR(error_reporter_, fc_op_index < 0,
                           "Op index of FC layer < 0!");
  auto& fc_op = ops[fc_op_index];
  const int kernel_tensor_index = fc_op->inputs[1];
  auto& kernel_tensor = tensors[kernel_tensor_index];
  std::get<0>(fc_kernel_quant_param_) = kernel_tensor->quantization->scale[0];
  std::get<1>(fc_kernel_quant_param_) =
      kernel_tensor->quantization->zero_point[0];
  // Construct inference engine that can calculate embedding vectors.
  auto fbb = GetFlatBufferBuilder(model_t_.get());
  embedding_extractor_buffer_ = std::vector<char>(
      fbb->GetBufferPointer(), fbb->GetBufferPointer() + fbb->GetSize());

  BasicEngineNativeBuilder builder(tflite::FlatBufferModel::BuildFromBuffer(
      embedding_extractor_buffer_.data(), embedding_extractor_buffer_.size()));

  EDGETPU_API_REPORT_ERROR(error_reporter_,
                           builder(&embedding_extractor_) == kEdgeTpuApiError,
                           builder.get_error_message());
  size_t output_tensor_num;

  EDGETPU_API_REPORT_ERROR(error_reporter_,
                           embedding_extractor_->get_num_of_output_tensors(
                               &output_tensor_num) == kEdgeTpuApiError,
                           embedding_extractor_->get_error_message());
  EDGETPU_API_REPORT_ERROR(error_reporter_, output_tensor_num < 1,
                           "output tensor number of embedding extractor < 1!");
  size_t tensor_size;

  EDGETPU_API_REPORT_ERROR(error_reporter_,
                           embedding_extractor_->get_output_tensor_size(
                               0, &tensor_size) == kEdgeTpuApiError,
                           embedding_extractor_->get_error_message());
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, embedding_vector_dim_ != tensor_size,
      "embedding_vector_dim_ mismatches output size of embedding extractor!");
  is_initialized_ = true;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::Train(const uint8_t* input,
                                               size_t dim1, size_t dim2,
                                               const int class_id) {
  IMPRINTING_ENGINE_NATIVE_INIT_CHECK();
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, weights_.size() % embedding_vector_dim_ != 0,
      "Size of weights_ must be multiple of embedding_vector_dim_!");
  int num_classes_trained = weights_.size() / embedding_vector_dim_;
  // Get previous trained image number of class |class_id|.
  // Or add a new category if |class_id| is not trained.
  float sqrt_sum_old = 0.;
  auto iter = metadata_.find(class_id);
  if (iter != metadata_.end()) {
    sqrt_sum_old = iter->second;
    EDGETPU_API_REPORT_ERROR(
        error_reporter_, class_id >= num_classes_trained,
        "This class_id must be smaller than the number of trained classes if "
        "this class is trained before.");
  } else {
    // Ensure the |class_id| is exactly the new class id.
    EDGETPU_API_REPORT_ERROR(error_reporter_, class_id > num_classes_trained,
                             "The class index of a new category is too large!");
    EDGETPU_API_REPORT_ERROR(
        error_reporter_, class_id < num_classes_trained,
        "Cannot change the base model classes not trained with imprinting "
        "method!");
  }
  const int num_images = dim1;
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, num_images > 200,
      "Too many training images, max # images per category is 200!");
  EDGETPU_API_REPORT_ERROR(error_reporter_, num_images == 0,
                           "No images sent for training!");
  EDGETPU_API_REPORT_ERROR(error_reporter_, dim2 == 0, "Image size is zero!");

  std::vector<float> weights_sum(embedding_vector_dim_, 0.0f);

  if (sqrt_sum_old > 0.f) {
    weights_sum = Dequantize<uint8_t>(
        {weights_.begin() + class_id * embedding_vector_dim_,
         weights_.begin() + (class_id + 1) * embedding_vector_dim_},
        /*scale=*/std::get<0>(fc_kernel_quant_param_),
        /*zero_point=*/std::get<1>(fc_kernel_quant_param_));
    for (int j = 0; j < embedding_vector_dim_; ++j) {
      weights_sum[j] = sqrt_sum_old * weights_sum[j];
    }
  }

  for (int i = 0; i < num_images; ++i) {
    float const* result;
    size_t result_size;
    EDGETPU_API_REPORT_ERROR(
        error_reporter_,
        embedding_extractor_->RunInference(input + i * dim2, dim2, &result,
                                           &result_size) == kEdgeTpuApiError,
        embedding_extractor_->get_error_message());
    EDGETPU_API_REPORT_ERROR(
        error_reporter_, result_size != embedding_vector_dim_,
        "Unexpected inference result size of embedding extractor!");
    for (int j = 0; j < embedding_vector_dim_; ++j) {
      weights_sum[j] += result[j];
    }
  }

  // Average weights and then re-normalize, same as normalizing the sum.
  float sum = 0.0f;
  for (int i = 0; i < weights_sum.size(); ++i) {
    sum += weights_sum[i] * weights_sum[i];
  }
  float sqrt_sum = std::sqrt(sum);
  std::vector<float> normalized_weights(embedding_vector_dim_);

  for (int i = 0; i < weights_sum.size(); ++i) {
    normalized_weights[i] = weights_sum[i] / sqrt_sum;
  }
  auto quant_normalized_weights = Quantize<uint8_t>(
      normalized_weights, /*scale=*/std::get<0>(fc_kernel_quant_param_),
      /*zero_point=*/std::get<1>(fc_kernel_quant_param_));

  if (sqrt_sum_old > 0) {
    for (int j = 0; j < embedding_vector_dim_; ++j) {
      weights_[class_id * embedding_vector_dim_ + j] =
          quant_normalized_weights[j];
    }
  } else {
    weights_.insert(weights_.end(), quant_normalized_weights.begin(),
                    quant_normalized_weights.end());
  }
  // Insert or replace metadata_.
  metadata_[class_id] = sqrt_sum;
  needs_postprocess_ = true;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::SaveModel(
    const std::string& output_path) {
  IMPRINTING_ENGINE_NATIVE_INIT_CHECK();
  EDGETPU_API_REPORT_ERROR(error_reporter_, weights_.empty(),
                           "Model without training won't be saved!");
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, weights_.size() % embedding_vector_dim_ != 0,
      "Weights size mismatch! It must be multiple of embedding vector's "
      "dimension!");

  EDGETPU_API_ENSURE_STATUS(PostprocessImprintingModel());
  EDGETPU_API_ENSURE_STATUS(UpdateModelTrainingMetaData());

  // Convert from tflite::ModelT format to FlatBufferBuilder.
  auto fbb = GetFlatBufferBuilder(model_t_.get());
  VLOG(1) << "Output model size in bytes: " << fbb->GetSize();
  EDGETPU_API_ENSURE_STATUS(WriteFile(
      std::string(reinterpret_cast<const char*>(fbb->GetBufferPointer()),
                  fbb->GetSize()),
      output_path, error_reporter_.get()));
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::RunInference(
    const uint8_t* const input, size_t in_size, float const** const output,
    size_t* const out_size) {
  const auto& start_time = std::chrono::steady_clock::now();

  if (needs_postprocess_) {
    EDGETPU_API_REPORT_ERROR(error_reporter_, weights_.empty(),
                             "Model without training couldn't run inference!");
    EDGETPU_API_REPORT_ERROR(
        error_reporter_, weights_.size() % embedding_vector_dim_ != 0,
        "Weights size mismatch! It must be multiple of embedding vector's "
        "dimension!");

    EDGETPU_API_ENSURE_STATUS(PostprocessImprintingModel());

    auto fbb = GetFlatBufferBuilder(model_t_.get());
    classification_model_buffer_ = std::vector<char>(
        fbb->GetBufferPointer(), fbb->GetBufferPointer() + fbb->GetSize());

    BasicEngineNativeBuilder builder(tflite::FlatBufferModel::BuildFromBuffer(
        classification_model_buffer_.data(),
        classification_model_buffer_.size()));

    EDGETPU_API_REPORT_ERROR(
        error_reporter_, builder(&classification_model_) == kEdgeTpuApiError,
        builder.get_error_message());
    needs_postprocess_ = false;
  }

  float const* result;
  size_t result_size;
  EDGETPU_API_REPORT_ERROR(
      error_reporter_,
      classification_model_->RunInference(input, in_size, &result,
                                          &result_size) == kEdgeTpuApiError,
      classification_model_->get_error_message());
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, result_size != num_classes_,
      "Unexpected inference result size of classification model!");
  *out_size = num_classes_;
  inference_result_.resize(num_classes_);
  std::memcpy(&inference_result_[0], result, sizeof(float) * num_classes_);
  *output = inference_result_.data();

  std::chrono::duration<double, std::milli> time_span =
      std::chrono::steady_clock::now() - start_time;

  inference_time_ = time_span.count();
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::get_inference_time(
    float* const time) const {
  *time = inference_time_;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::get_metadata(
    std::map<int, float>* metadata) {
  IMPRINTING_ENGINE_NATIVE_INIT_CHECK();
  *metadata = metadata_;
  return kEdgeTpuApiOk;
}

EdgeTpuApiStatus ImprintingEngineNative::set_metadata(
    const std::map<int, float>& metadata) {
  IMPRINTING_ENGINE_NATIVE_INIT_CHECK();
  metadata_ = metadata;
  return kEdgeTpuApiOk;
}

// Creates BasicEngineNative with FlatBuffer file.
ImprintingEngineNativeBuilder::ImprintingEngineNativeBuilder(
    const std::string& model_path, bool keep_classes)
    : model_path_(model_path), keep_classes_(keep_classes) {
  error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
}

EdgeTpuApiStatus ImprintingEngineNativeBuilder::operator()(
    std::unique_ptr<ImprintingEngineNative>* engine) {
  EDGETPU_API_REPORT_ERROR(
      error_reporter_, !engine,
      "Null output pointer passed to ImprintingEngineNativeBuilder!");
  *engine = absl::make_unique<ImprintingEngineNative>();
  EDGETPU_API_REPORT_ERROR(
      error_reporter_,
      (*engine)->Init(model_path_, keep_classes_) == kEdgeTpuApiError,
      (*engine)->get_error_message());
  return kEdgeTpuApiOk;
}

}  // namespace imprinting
}  // namespace learn
}  // namespace coral
