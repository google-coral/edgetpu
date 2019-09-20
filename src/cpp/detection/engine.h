#ifndef EDGETPU_CPP_DETECTION_ENGINE_H_
#define EDGETPU_CPP_DETECTION_ENGINE_H_

#include <vector>

#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/bbox_utils.h"

namespace coral {

class DetectionEngine : public BasicEngine {
 public:
  // Loads detection model. Now we only support SSD model with postprocessing
  // operator.
  //  - 'model_path' : the file path of the model.
  explicit DetectionEngine(const std::string& model_path)
      : BasicEngine(model_path) {
    Validate();
  }

  // Loads detection model and specifies EdgeTpu to use.
  //  - 'model_path' : the file path of the model.
  //  - 'device_path' : the device path of EdgeTpu.
  explicit DetectionEngine(const std::string& model_path,
                           const std::string& device_path)
      : BasicEngine(model_path, device_path) {
    Validate();
  }

  // Detects objects with input tensor.
  //  - 'input' : vector of uint8, input to the model.
  //  - 'threshold' : float, minimum confidence threshold for returned
  //       predictions. For example, use 0.5 to receive only predictions
  //       with a confidence equal-to or higher-than 0.5.
  //  - 'top_k': int, the maximum number of predictions to return.
  //
  // The function will return a vector of predictions which is sorted by
  // <score, label_id> in descending order.
  std::vector<DetectionCandidate> DetectWithInputTensor(
      const std::vector<uint8_t>& input, float threshold = 0.0, int top_k = 3);

 private:
  // Checks the format of the model.
  void Validate();
};

}  // namespace coral

#endif  // EDGETPU_CPP_DETECTION_ENGINE_H_
