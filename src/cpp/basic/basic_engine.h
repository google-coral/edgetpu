#ifndef EDGETPU_CPP_BASIC_BASIC_ENGINE_H_
#define EDGETPU_CPP_BASIC_BASIC_ENGINE_H_

#include <vector>

#include "src/cpp/basic/basic_engine_native.h"

namespace coral {

// BasicEngine wraps given model, creates interpreter and initializes EdgetTpu.
class BasicEngine {
 public:
  // Loads TFlite model and initializes interpreter.
  //  - 'model_path' : the file path of the model.
  explicit BasicEngine(const std::string& model_path);
  // Similar to above, but uses Edge TPU specified at `device_path`.
  explicit BasicEngine(const std::string& model_path,
                       const std::string& device_path);
  // Initializes BasicEngine with FlatBufferModel object.
  explicit BasicEngine(std::unique_ptr<tflite::FlatBufferModel> model);
  // Initializes BasicEngine with FlatBufferModel object and customized
  // resolver.
  explicit BasicEngine(
      std::unique_ptr<tflite::FlatBufferModel> model,
      std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver);
  // For input, we assume there is only one tensor.
  // The input for RunInference is the flattened array of input tensor.
  template <typename T>
  std::vector<std::vector<float>> RunInference(const std::vector<T>& input);

  // Functions to get/check attributes.

  // Gets device path associated with Edge TPU.
  std::string device_path() const;
  // Gets the path of the model.
  std::string model_path() const;
  // Gets shape of input tensor.
  std::vector<int> get_input_tensor_shape() const;
  // Gets sizes of output tensors. We assume that all output tensors are
  // in 1 dimension so the output is an array of lengths for each output
  // tensor.
  std::vector<size_t> get_all_output_tensors_sizes() const;
  // Gets time consumed for last inference (milliseconds).
  float get_inference_time() const;

 private:
  std::unique_ptr<BasicEngineNative> engine_;
};
}  // namespace coral

#endif  // EDGETPU_CPP_BASIC_BASIC_ENGINE_H_
