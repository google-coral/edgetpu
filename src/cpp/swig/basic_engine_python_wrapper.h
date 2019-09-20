#ifndef EDGETPU_CPP_SWIG_BASIC_ENGINE_PYTHON_WRAPPER_H_
#define EDGETPU_CPP_SWIG_BASIC_ENGINE_PYTHON_WRAPPER_H_

#include <Python.h>

#include "src/cpp/basic/basic_engine_native.h"

namespace coral {

// Wrapper of BasicEngine for Python API.
class BasicEnginePythonWrapper {
 public:
  BasicEnginePythonWrapper();

  // Loads TFlite model and initializes interpreter.
  //  - 'model_path' : the file path of the model.
  // This function will return an empty string if initialization success,
  // otherwise it will return the error message.
  std::string Init(const std::string& model_path);
  // Loads TFlite model and initializes interpreter with specific EdgeTpu.
  //  - 'model_path' : the file path of the model.
  //  - 'deivce_path' : the path of specific EdgeTpu.
  // This function will return an empty string if initialization success,
  // otherwise it will return the error message.
  std::string Init(const std::string& model_path,
                   const std::string& device_path);

  PyObject* RunInference(const uint8_t* input, int in_size);

  // Gets shape of input tensor.
  // The value for each dimension is unsigned long, which may be converted to
  // npy_intp (signed long). It may cause narrowing conversion. However
  // considering the scale of input, it's safe.
  PyObject* get_input_tensor_shape() const;

  // Gets shapes of output tensors. We assume that all output tensors are
  // in 1 dimension so the output is an array of lengths for each output
  // tensor.
  PyObject* get_all_output_tensors_sizes() const;
  // Gets number of output tensors.
  PyObject* get_num_of_output_tensors() const;
  // Gets size of output tensor.
  PyObject* get_output_tensor_size(int tensor_index) const;
  // Size of input array (size of input tensor).
  PyObject* required_input_array_size() const;
  // Size of output array (The sum of size of all output tensors).
  PyObject* total_output_array_size() const;
  // Gets raw output of last inference.
  PyObject* get_raw_output() const;
  // Gets time consumed for last inference (milliseconds).
  PyObject* get_inference_time() const;

  // Gets the path of the model.
  PyObject* model_path() const;
  // Gets the EdgeTpu device path of the model.
  PyObject* device_path() const;

 private:
  std::unique_ptr<BasicEngineNative> engine_;
};
}  // namespace coral

#endif  // EDGETPU_CPP_SWIG_BASIC_ENGINE_PYTHON_WRAPPER_H_
