#ifndef EDGETPU_CPP_SWIG_IMPRINTING_ENGINE_PYTHON_WRAPPER_H_
#define EDGETPU_CPP_SWIG_IMPRINTING_ENGINE_PYTHON_WRAPPER_H_

#include <Python.h>

#include "src/cpp/learn/imprinting/engine_native.h"

namespace coral {
namespace learn {
namespace imprinting {
// Wrapper of ImprintingEngine for Python API.
class ImprintingEnginePythonWrapper {
 public:
  ImprintingEnginePythonWrapper();

  // Loads TFlite model and initializes training engine.
  //  - 'model_path' : the file path of the model.
  //  - 'keep_classes' :  whether to keep previous classes.
  // This function will return an empty string if initialization success,
  // otherwise it will return the error message.
  std::string Init(const std::string& model_path, bool keep_classes);

  PyObject* SaveModel(const std::string& output_path);

  PyObject* Train(const uint8_t* input, int dim1, int dim2, int class_id);

  PyObject* RunInference(const uint8_t* input, int in_size);

  // Gets time consumed for last inference (milliseconds).
  PyObject* get_inference_time() const;

 private:
  std::unique_ptr<ImprintingEngineNative> engine_;
};
}  // namespace imprinting
}  // namespace learn
}  // namespace coral

#endif  // EDGETPU_CPP_SWIG_IMPRINTING_ENGINE_PYTHON_WRAPPER_H_
