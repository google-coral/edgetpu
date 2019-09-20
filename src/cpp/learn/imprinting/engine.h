#ifndef EDGETPU_CPP_LEARN_IMPRINTING_ENGINE_H_
#define EDGETPU_CPP_LEARN_IMPRINTING_ENGINE_H_

#include <string>
#include <vector>

#include "src/cpp/learn/imprinting/engine_native.h"

namespace coral {
namespace learn {
namespace imprinting {

// Class that implements `Imprinted Weights` transfer learning method proposed
// in paper https://arxiv.org/pdf/1712.07136.pdf.
class ImprintingEngine {
 public:
  // Validates input model and initializes `embedding_extractor`.
  //
  // The input model comes with L2Norm layer. It is supposed to be a full
  // classification model.
  //
  // Users can choose whether to keep previous classes by setting keep_classes
  // to true or false.
  explicit ImprintingEngine(const std::string& model_path,
                            bool keep_classes = false);

  // For input, we assume there is only one tensor with type uint8_t.
  // There is only one output tensor, the classification results after softmax.
  std::vector<float> RunInference(const std::vector<uint8_t>& input);

  // Gets time consumed for last inference (milliseconds).
  float get_inference_time() const;

  // Saves the re-trained model to specific path.
  //
  // Re-trained model will contain:
  //  [embedding_extractors] -> L2Norm -> Conv2d -> Mul -> Reshape -> Softmax
  void SaveModel(const std::string& output_path);

  // Online-trains the model with images of a certain category .
  //
  // Inputs: a list of input tensors(vector<uint8_t>) with each input tensor
  // converted from an input image, and class id.
  //
  // If training a new category, the class id should be exactly of the next
  // class.
  //
  // If training existing category with more images, the imprinting engine must
  // be under keep_classes mode.
  //
  // Call this function multiple times to train multiple different categories.
  void Train(const std::vector<std::vector<uint8_t>>& images,
             const int class_id);

  // Copying or assignment is disallowed
  ImprintingEngine(const ImprintingEngine&) = delete;
  ImprintingEngine& operator=(const ImprintingEngine&) = delete;

 private:
  std::unique_ptr<ImprintingEngineNative> engine_;
};

}  // namespace imprinting
}  // namespace learn
}  // namespace coral

#endif  // EDGETPU_CPP_LEARN_IMPRINTING_ENGINE_H_
