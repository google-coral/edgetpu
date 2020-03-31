#ifndef EDGETPU_CPP_BASIC_BASIC_ENGINE_NATIVE_H_
#define EDGETPU_CPP_BASIC_BASIC_ENGINE_NATIVE_H_

#include "edgetpu.h"
#include "src/cpp/basic/edgetpu_resource_manager.h"
#include "src/cpp/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

namespace coral {

// BasicEngine wraps given model, creates interpreter and initializes EdgetTpu.
// The return type is EdgeTpuApiStatus for all functions except
// get_error_message(). When error occurred in the process, the function will
// return kEdgeTpuApiError and user can call get_error_message() to retrieve
// related info.
//
// For example:
//    int size;
//    if (engine->get_output_tensor_size(2, &size) == kEdgeTpuApiOk) {
//      // continue with output
//    } else {
//      // display error message.
//      std::cout<<engine->get_error_message()<<std::endl;
//    }
//
class BasicEngineNative {
 public:
  // Creates an BasicEngineNative and initializes ErrorReporter.
  BasicEngineNative();
  // Destruct members in order to avoid lifetime conflicts.
  ~BasicEngineNative();

  BasicEngineNative(const BasicEngineNative&) = delete;
  BasicEngineNative& operator=(const BasicEngineNative&) = delete;

  // For input, we assume there is only one tensor with type uint8_t.
  // Input buffer could have padded bytes at the end, in which case |in_size|
  // could be larger than the input tensor size, denoted by n, and only the
  // first n bytes of the input buffer will be used. |in_size| can not be
  // smaller than n.
  // For output, when there are multiple tensors we'll store them in one
  // continuous array in order. To parse output correctly, please use
  // get_num_of_output_tensors and get_output_tensor_size to get more info.
  EdgeTpuApiStatus RunInference(const uint8_t* const input,
                                const size_t in_size,
                                float const** const output,
                                size_t* const out_size);

  // Overloads RunInference to take float inputs.
  EdgeTpuApiStatus RunInference(const float* const input, const size_t in_size,
                                float const** const output,
                                size_t* const out_size);

  // Gets shape of input tensor.
  EdgeTpuApiStatus get_input_tensor_shape(int const** dims,
                                          int* dims_num) const;

  // Gets size of required input array.
  EdgeTpuApiStatus get_input_array_size(size_t* array_size) const;

  // Gets sizes of output tensors. We assume that all output tensors are
  // in 1 dimension so the output is an array of lengthes for each output
  // tensor.
  EdgeTpuApiStatus get_all_output_tensors_sizes(size_t const** tensor_sizes,
                                                size_t* tensor_num) const;

  // Gets number of output tensors.
  EdgeTpuApiStatus get_num_of_output_tensors(size_t* output) const;

  // Gets size of output tensor.
  EdgeTpuApiStatus get_output_tensor_size(const int tensor_index,
                                          size_t* const output) const;

  EdgeTpuApiStatus total_output_array_size(size_t* const output) const;

  // Gets raw output of last inference.
  EdgeTpuApiStatus get_raw_output(float const** output, size_t* out_size) const;

  EdgeTpuApiStatus model_path(std::string* path) const;

  EdgeTpuApiStatus device_path(std::string* path) const;

  EdgeTpuApiStatus get_inference_time(float* const time) const;

  // This function is offered to high level APIs to retrieve error message when
  // get kEdgeTpuApiError.
  std::string get_error_message();

  // Initializes with FlatBuffer model and customized resolver.
  // When resolver is nullptr, this function will create a new resolver with
  // edgetpu::kCustomOp added.
  EdgeTpuApiStatus Init(std::unique_ptr<tflite::FlatBufferModel> model,
                        tflite::ops::builtin::BuiltinOpResolver* resolver);

  // Initializes with FlatBuffer file path and Edge TPU path.
  EdgeTpuApiStatus Init(const std::string& model_path,
                        const std::string& device_path);

 private:
  // Parses FlatBuffer model from file.
  EdgeTpuApiStatus BuildModelFromFile(const std::string& model_path);
  // Initializes EdgeTpuResource.
  EdgeTpuApiStatus InitializeEdgeTpuResource(const std::string& device_path);
  // Initializes Interpreter.
  EdgeTpuApiStatus CreateInterpreterWithResolver(
      tflite::ops::builtin::BuiltinOpResolver* resolver);
  // Initializes input and output arrays.
  EdgeTpuApiStatus InitializeInputAndOutput();
  // Returns the output tensor sizes of the given model, assuming all tensors
  // have been allocated.
  EdgeTpuApiStatus GetOutputTensorSizes();
  // Helper for RunInference to prevent code repetitions.
  // This method performs a deep copy of the output tensors to output.
  EdgeTpuApiStatus ParseAndCopyInferenceResults(float const** const output,
                                                size_t* const out_size);

  // Indicates whether the instance is initialized.
  bool is_initialized_;
  // Path of the model.
  std::string model_path_;
  // EdgeTpuResource must be destructed after interpreter_. Because the Edge TPU
  // context will be used by destructor of Custom Op.
  std::unique_ptr<EdgeTpuResource> edgetpu_resource_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  // Shape of input tensor.
  std::vector<int> input_tensor_shape_;
  size_t input_array_size_;
  // Sizes of output tensors.
  std::vector<size_t> output_tensor_sizes_;
  size_t output_array_size_;
  // Inference result.
  std::vector<float> inference_result_;
  // Time consumed on last inference.
  float inference_time_;
  // Data structure to store error messages.
  std::unique_ptr<EdgeTpuErrorReporter> error_reporter_;
};

// Builds an BasicEngineNavtive with given model object or file path.
//
// model_path: The file path of FlatBuffer model file.
// model: the FlatBufferModel object passed by user.
// device_path: specific EdgeTpu device path.
// resolver: customized op_resolver for given model.
//
// Returns kEdgeTpuApiOk when BasicEngineNative is successfully created.
//
// Example:
//   // With model path.
//   BasicEngineNativeBuilder builder('test_data/mobilenet_v1.tflite');
//   std::unique_ptr<BasicEngineNative> engine;
//   builder(&engine);
//
//   // With FlatBufferModel object.
//   std::unique_ptr<tflite::FlatBufferModel> model = ...
//   BasicEngineNativeBuilder builder(model);
//   std::unique_ptr<BasicEngineNative> engine;
//   builder(&engine);
class BasicEngineNativeBuilder {
 public:
  // Creates BasicEngineNative with FlatBuffer file.
  explicit BasicEngineNativeBuilder(const std::string& model_path);
  // Creates BasicEngineNative with FlatBuffer file and specifies EdgeTpu.
  BasicEngineNativeBuilder(const std::string& model_path,
                           const std::string& device_path);
  // Creates BasicEngineNative with FlatBufferModel object.
  // In our design, each BasicEngineNative is binded with one model. Hence the
  // ownership of FlatBufferModel object will be transfer to created instance.
  // When the BasicEngineNativeBuilder is initialized with
  // std::unique_ptr<tflite::FlatBufferModel> model, it will be a one-time
  // builder due to the losing of ownership. In this case, the builder can only
  // create one BasicEngineNative.
  explicit BasicEngineNativeBuilder(
      std::unique_ptr<tflite::FlatBufferModel> model);
  // Creates BasicEngineNative with FlatBufferModel object and customized
  // resolver.
  BasicEngineNativeBuilder(
      std::unique_ptr<tflite::FlatBufferModel> model,
      std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver);

  BasicEngineNativeBuilder(const BasicEngineNativeBuilder&) = delete;
  BasicEngineNativeBuilder& operator=(const BasicEngineNativeBuilder&) = delete;
  EdgeTpuApiStatus operator()(std::unique_ptr<BasicEngineNative>* engine);

  // Caller can use this function to retrieve error message when get
  // kEdgeTpuApiError.
  std::string get_error_message();

 private:
  bool read_from_file_;
  std::string model_path_, device_path_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver_;
  // Data structure to stores error messages.
  std::unique_ptr<EdgeTpuErrorReporter> error_reporter_;
};

}  // namespace coral

#endif  // EDGETPU_CPP_BASIC_BASIC_ENGINE_NATIVE_H_
