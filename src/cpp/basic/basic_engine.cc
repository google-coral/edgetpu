#include "src/cpp/basic/basic_engine.h"

#include "glog/logging.h"

namespace coral {
BasicEngine::BasicEngine(const std::string& model_path) {
  BasicEngineNativeBuilder builder(model_path);
  LOG_IF(FATAL, builder(&engine_) == kEdgeTpuApiError)
      << builder.get_error_message();
}

BasicEngine::BasicEngine(const std::string& model_path,
                         const std::string& device_path) {
  BasicEngineNativeBuilder builder(model_path, device_path);
  LOG_IF(FATAL, builder(&engine_) == kEdgeTpuApiError)
      << builder.get_error_message();
}

BasicEngine::BasicEngine(std::unique_ptr<tflite::FlatBufferModel> model) {
  BasicEngineNativeBuilder builder(std::move(model));
  LOG_IF(FATAL, builder(&engine_) == kEdgeTpuApiError)
      << builder.get_error_message();
}

BasicEngine::BasicEngine(
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver) {
  BasicEngineNativeBuilder builder(std::move(model), std::move(resolver));
  LOG_IF(FATAL, builder(&engine_) == kEdgeTpuApiError)
      << builder.get_error_message();
}

std::vector<int> BasicEngine::get_input_tensor_shape() const {
  int const* dims;
  int dims_num;
  LOG_IF(FATAL,
         engine_->get_input_tensor_shape(&dims, &dims_num) == kEdgeTpuApiError)
      << engine_->get_error_message();
  return std::vector<int>(dims, dims + dims_num);
}

std::vector<size_t> BasicEngine::get_all_output_tensors_sizes() const {
  size_t const* tensor_sizes;
  size_t tensor_num;
  LOG_IF(FATAL, engine_->get_all_output_tensors_sizes(
                    &tensor_sizes, &tensor_num) == kEdgeTpuApiError)
      << engine_->get_error_message();
  return std::vector<size_t>(tensor_sizes, tensor_sizes + tensor_num);
}

// Gets the path of the model.
std::string BasicEngine::model_path() const {
  std::string path;
  LOG_IF(FATAL, engine_->model_path(&path) == kEdgeTpuApiError)
      << engine_->get_error_message();
  return path;
}

std::string BasicEngine::device_path() const {
  std::string path;
  LOG_IF(FATAL, engine_->device_path(&path) == kEdgeTpuApiError)
      << engine_->get_error_message();
  return path;
}

template <typename T>
std::vector<std::vector<float>> BasicEngine::RunInference(
    const std::vector<T>& input) {
  float const* tmp_result;
  size_t tmp_result_size;
  LOG_IF(FATAL, engine_->RunInference(input.data(), input.size(), &tmp_result,
                                      &tmp_result_size) == kEdgeTpuApiError)
      << engine_->get_error_message();

  // Parse 1d result vector into output tensors.
  std::vector<size_t> output_tensor_shape = get_all_output_tensors_sizes();
  std::vector<std::vector<float>> results(output_tensor_shape.size());
  int offset = 0;
  for (int i = 0; i < output_tensor_shape.size(); ++i) {
    const size_t size_of_output_tensor_i = output_tensor_shape[i];
    results[i].resize(size_of_output_tensor_i);
    std::memcpy(results[i].data(), tmp_result + offset,
                sizeof(float) * size_of_output_tensor_i);
    offset += size_of_output_tensor_i;
  }
  // Sanity check.
  CHECK(tmp_result_size == offset) << "Error in output tensor paring, mismatch "
                                      "between offset and output array size.";
  return results;
}

// Explicit instantiation.
template std::vector<std::vector<float>> BasicEngine::RunInference(
    const std::vector<uint8_t>&);
template std::vector<std::vector<float>> BasicEngine::RunInference(
    const std::vector<float>&);

// Gets time consumed for last inference (milliseconds).
float BasicEngine::get_inference_time() const {
  float time;
  LOG_IF(FATAL, engine_->get_inference_time(&time) == kEdgeTpuApiError)
      << engine_->get_error_message();
  return time;
}
}  // namespace coral
