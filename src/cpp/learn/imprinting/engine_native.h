#ifndef EDGETPU_CPP_LEARN_IMPRINTING_ENGINE_NATIVE_H_
#define EDGETPU_CPP_LEARN_IMPRINTING_ENGINE_NATIVE_H_

#include <map>
#include <vector>

#include "absl/memory/memory.h"
#include "src/cpp/basic/basic_engine_native.h"
#include "src/cpp/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace coral {
namespace learn {
namespace imprinting {

// Class that implements `Imprinted Weights` transfer learning method proposed
// in paper https://arxiv.org/pdf/1712.07136.pdf.
class ImprintingEngineNative {
 public:
  ImprintingEngineNative() {
    error_reporter_ = absl::make_unique<EdgeTpuErrorReporter>();
    is_initialized_ = false;
  }

  ~ImprintingEngineNative() {
    // Free memory here explicitly to make sure `embedding_extractor_buffer_`
    // lives longer than `embedding_extractor_`.
    embedding_extractor_.reset();
    embedding_extractor_buffer_.clear();
  }

  // Copying or assignment is disallowed
  ImprintingEngineNative(const ImprintingEngineNative&) = delete;
  ImprintingEngineNative& operator=(const ImprintingEngineNative&) = delete;

  // Validates input model and initializes `embedding_extractor`.
  //
  // The input model comes with L2Norm layer. It is supposed to be a full
  // classification model.
  //
  // Users can choose whether to keep previous classes by setting keep_classes
  // to true or false.
  EdgeTpuApiStatus Init(const std::string& model_path, bool keep_classes);

  // Saves the re-trained model to specific path.
  //
  // Re-trained model will contain:
  //  [embedding_extractors] -> L2Norm -> Conv2d -> Mul -> Reshape -> Softmax
  EdgeTpuApiStatus SaveModel(const std::string& output_path);

  // For input, we assume there is only one tensor with type uint8_t.
  // For output, we assume there is only one tensor representing the
  // classification results.
  EdgeTpuApiStatus RunInference(const uint8_t* const input, const int in_size,
                                float const** const output,
                                int* const out_size);

  EdgeTpuApiStatus get_inference_time(float* const time) const;

  // Online-trains the model with images of a certain category .
  //
  // Inputs: flattened 2-D array in C style. dim1 is number of images and dim2
  // is size of each image (also flattened).
  //
  // If training a new category, the class id should be exactly of the next
  // class.
  //
  // If training existing category with more images, the imprinting engine must
  // be under keep_classes mode.
  //
  // Call this function multiple times to train multiple different categories.
  EdgeTpuApiStatus Train(const uint8_t* input, int dim1, int dim2,
                         const int class_id);

  // Getter/setter for metadata, used in tests only.
  EdgeTpuApiStatus get_metadata(std::map<int, float>* metadata);
  EdgeTpuApiStatus set_metadata(const std::map<int, float>& metadata);

  // Caller can use this function to retrieve error message when get
  // kEdgeTpuApiError.
  std::string get_error_message() { return error_reporter_->message(); }

 private:
  // Extracts training metadata from model description string. It assumes that
  // model description looks like:
  // 0 5.4
  // 1 6.5
  // 2 4.1
  // ...
  void ExtractModelTrainingMetadata();

  // Saves training metadata to model description.
  EdgeTpuApiStatus UpdateModelTrainingMetaData();

  // Preprocesses the model before training.
  //
  // - keeps a copy of previous weights if asked to keep previous classes, or
  // clears the weights if asked not.
  // - inserts the L2Norm output to model outputs for feature extracting.
  EdgeTpuApiStatus PreprocessImprintingModel();

  // Postprocesses the model after imprinting.
  //
  // - modifies the Conv2d->Reshape->Softmax sequence to work for new classes.
  // - removes the existing tensor from, adds classifcaiton output to the
  // model outputs.
  EdgeTpuApiStatus PostprocessImprintingModel();

  // Weights of Fully Connected layer.
  std::vector<uint8_t> weights_;

  // Model metadata, storing the numbers of training images for each class.
  // Training metadata is a map from class label index to the weight of seen
  // average embedding.
  //
  // To understand it better, e.g., we have two seen normalized embeddings `f_0`
  // and `f_1`. Their average embedding is `normalized_f_01`. We can know
  // sqrt(`f_0`+`f_1`) is `normalized_f_01` multiplied by some scalar weight
  // `sqrt_sum_01`. With this `normalized_f_01` and this weight, we can resume
  // (`f_0`+`f_1`), which is needed to calculate (`f_0`+`f_1`+`f_2)
  // when we want to train a new sample with embedding `f_2`.
  //
  // This map is used for online learning and is stored in model description
  // field with the following format:
  // 0 5.4
  // 1 6.5
  // 2 4.1
  std::map<int, float> metadata_;

  // Dimension of embedding vector.
  int embedding_vector_dim_;

  // Number of classes of the input model.
  int num_classes_ = 0;

  // Tensor index of classification results.
  int classification_tensor_index_;

  // Quantization parameters, i.e., scale and zero point for fc kernel.
  std::tuple<float, int64_t> fc_kernel_quant_param_;

  // `embedding_extractor_buffer_` is needed since
  // FlatBufferModel::BuildFromBuffer() (called by BasicEngine) requires caller
  // to maintain the ownership.
  std::vector<char> embedding_extractor_buffer_;
  // Run this engine to get embedding vectors.
  std::unique_ptr<BasicEngineNative> embedding_extractor_;

  std::vector<char> classification_model_buffer_;
  // Runs this engine to get classification results.
  std::unique_ptr<BasicEngineNative> classification_model_;
  // tflite::ModelT representation of the classification model.
  // Notice the model_t is initialized as an embedding extractor after
  // Preprocess step. With one Postprocess step, it will turn out to be a
  // classification model.
  std::unique_ptr<tflite::ModelT> model_t_;

  // Inference result.
  std::vector<float> inference_result_;
  // Time consumed on last inference.
  float inference_time_;
  // Whether the model needs postprocess. If the model just trained some images,
  // it needs postprocess.
  bool needs_postprocess_ = false;

  // Whether to keep previous clasess.
  bool keep_classes_;

  // Data structure to stores error messages.
  std::unique_ptr<EdgeTpuErrorReporter> error_reporter_;
  // Indicates whether the instance is initialized.
  bool is_initialized_;
};

// Builds an ImprintingEngineNative with given file path.
//
// model_path: The file path of FlatBuffer model file.
// keep_classes: Whether to keep classes existed in the model. By default it is
//   False.
//
// Returns kEdgeTpuApiOk when ImprintingEngineNative is successfully created.
// Otherwise you can call get_error_message() to retrieve error message.
//
// Example:
//   // With model path.
//   ImprintingEngineNativeBuilder builder('mobilenet_v1_l2_norm.tflite');
//   std::unique_ptr<ImprintingEngineNative> engine;
//   if (builder(&engine) != kEdgeTpuApiOk) {
//     // Handle error with 'builder.get_error_message()'
//   }
//
class ImprintingEngineNativeBuilder {
 public:
  // Creates ImprintingEngineNativeBuilder with FlatBuffer file.
  explicit ImprintingEngineNativeBuilder(const std::string& model_path,
                                         bool keep_classes = false);

  ImprintingEngineNativeBuilder(const ImprintingEngineNativeBuilder&) = delete;
  ImprintingEngineNativeBuilder& operator=(
      const ImprintingEngineNativeBuilder&) = delete;

  EdgeTpuApiStatus operator()(std::unique_ptr<ImprintingEngineNative>* engine);

  // Caller can use this function to retrieve error message when get
  // kEdgeTpuApiError.
  std::string get_error_message() { return error_reporter_->message(); }

 private:
  // Data structure to stores error messages.
  std::unique_ptr<EdgeTpuErrorReporter> error_reporter_;
  std::string model_path_;
  bool keep_classes_;
};

}  // namespace imprinting
}  // namespace learn
}  // namespace coral

#endif  // EDGETPU_CPP_LEARN_IMPRINTING_ENGINE_NATIVE_H_
