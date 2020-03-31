#ifndef EDGETPU_CPP_CLASSIFICATION_ENGINE_H_
#define EDGETPU_CPP_CLASSIFICATION_ENGINE_H_

#include <vector>

#include "src/cpp/basic/basic_engine.h"

namespace coral {

struct ClassificationCandidate {
  ClassificationCandidate(const int id_, const float score_)
      : id(id_), score(score_) {}
  int id;
  float score;
};

inline bool operator==(const ClassificationCandidate& x,
                       const ClassificationCandidate& y) {
  return x.score == y.score && x.id == y.id;
}

inline bool operator!=(const ClassificationCandidate& x,
                       const ClassificationCandidate& y) {
  return !(x == y);
}

class ClassificationEngine : public BasicEngine {
 public:
  // Loads classification model.
  //  - 'model_path' : the file path of the model.
  explicit ClassificationEngine(const std::string& model_path)
      : BasicEngine(model_path) {
    Validate();
  }

  // Loads classification model and specifies EdgeTpu to use.
  //  - 'model_path' : the file path of the model.
  //  - 'device_path' : the device path of EdgeTpu.
  explicit ClassificationEngine(const std::string& model_path,
                                const std::string& device_path)
      : BasicEngine(model_path, device_path) {
    Validate();
  }

  // Classifies with input tensor.
  //  - 'input' : vector of uint8, input to the model.
  //  - 'threshold' : float, minimum confidence threshold for returned
  //       classifications. For example, use 0.5 to receive only classifications
  //       with a confidence equal-to or higher-than 0.5.
  //  - 'top_k': int, the maximum number of classifications to return.
  //
  // The function will return a vector of predictions which is sorted by
  // <score, label_id> in descending order.
  std::vector<ClassificationCandidate> ClassifyWithInputTensor(
      const std::vector<uint8_t>& input, float threshold = 0.0, int top_k = 3);

 private:
  // Checks the format of the model.
  void Validate();
};

}  // namespace coral

#endif  // EDGETPU_CPP_CLASSIFICATION_ENGINE_H_
