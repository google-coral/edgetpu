#include "src/cpp/learn/imprinting/engine.h"

#include "glog/logging.h"

namespace coral {
namespace learn {
namespace imprinting {

ImprintingEngine::ImprintingEngine(const std::string& model_path,
                                   bool keep_classes) {
  ImprintingEngineNativeBuilder builder(model_path, keep_classes);
  CHECK_EQ(builder(&engine_), kEdgeTpuApiOk) << builder.get_error_message();
}

void ImprintingEngine::Train(const std::vector<std::vector<uint8_t>>& images,
                             const int class_id) {
  // Flatten input data.
  std::vector<uint8_t> tmp;
  int d1 = images.size();
  CHECK_GT(d1, 0) << "No images sent for training!";
  int d2 = images[0].size();
  CHECK_GT(d2, 0) << "Image size is zero!";
  tmp.resize(d1 * d2);
  int offset = 0;
  for (int i = 0; i < d1; ++i) {
    std::copy(images[i].begin(), images[i].end(), tmp.data() + offset);
    offset += d2;
  }
  CHECK_EQ(engine_->Train(tmp.data(), d1, d2, class_id), kEdgeTpuApiOk)
      << engine_->get_error_message();
}

std::vector<float> ImprintingEngine::RunInference(
    const std::vector<uint8_t>& input) {
  std::vector<float> results;
  float const* tmp_result;
  int tmp_result_size;
  LOG_IF(FATAL, engine_->RunInference(input.data(), input.size(), &tmp_result,
                                      &tmp_result_size) == kEdgeTpuApiError)
      << engine_->get_error_message();
  results.resize(tmp_result_size);
  std::memcpy(results.data(), tmp_result, sizeof(float) * tmp_result_size);
  return results;
}

float ImprintingEngine::get_inference_time() const {
  float time;
  LOG_IF(FATAL, engine_->get_inference_time(&time) == kEdgeTpuApiError)
      << engine_->get_error_message();
  return time;
}

void ImprintingEngine::SaveModel(const std::string& output_path) {
  CHECK_EQ(engine_->SaveModel(output_path), kEdgeTpuApiOk)
      << engine_->get_error_message();
}

}  // namespace imprinting
}  // namespace learn
}  // namespace coral
