#include "src/cpp/swig/imprinting_engine_python_wrapper.h"

#include <Python.h>
#include <numpy/arrayobject.h>

#include "src/cpp/error_reporter.h"

namespace coral {
namespace learn {
namespace imprinting {
namespace {
#define ENSURE_ENGINE_STATUS(status)                         \
  do {                                                       \
    if ((status) == kEdgeTpuApiError) {                      \
      PyErr_SetString(PyExc_RuntimeError,                    \
                      engine_->get_error_message().c_str()); \
      return nullptr;                                        \
    }                                                        \
  } while (0)
#define ENSURE_ENGINE_INIT()                                           \
  do {                                                                 \
    if (!engine_) {                                                    \
      PyErr_SetString(PyExc_RuntimeError,                              \
                      "ImprintingEnginePythonWrapper uninitialized!"); \
      return nullptr;                                                  \
    }                                                                  \
  } while (0)
}  // namespace

ImprintingEnginePythonWrapper::ImprintingEnginePythonWrapper() {
  // This function must be called in the initialization section of a module that
  // will make use of the C-API (PyArray_SimpleNewFromData).
  // It imports the module where the function-pointer table is stored and points
  // the correct variable to it.
  // Different with import_array() import_array1() has return value.
  // https://docs.scipy.org/doc/numpy-1.14.2/reference/c-api.array.html
  import_array1();
}

std::string ImprintingEnginePythonWrapper::Init(const std::string& model_path,
                                                bool keep_classes) {
  ImprintingEngineNativeBuilder builder(model_path, keep_classes);
  builder(&engine_);
  return builder.get_error_message();
}

PyObject* ImprintingEnginePythonWrapper::SaveModel(
    const std::string& output_path) {
  ENSURE_ENGINE_INIT();
  ENSURE_ENGINE_STATUS(engine_->SaveModel(output_path));
  Py_RETURN_NONE;
}

PyObject* ImprintingEnginePythonWrapper::Train(const uint8_t* input, int dim1,
                                               int dim2, int class_id) {
  ENSURE_ENGINE_INIT();
  ENSURE_ENGINE_STATUS(engine_->Train(input, dim1, dim2, class_id));
  Py_RETURN_NONE;
}

PyObject* ImprintingEnginePythonWrapper::RunInference(const uint8_t* input,
                                                      int in_size) {
  ENSURE_ENGINE_INIT();
  float const* output;
  int out_size;
  ENSURE_ENGINE_STATUS(
      engine_->RunInference(input, in_size, &output, &out_size));
  // Parse results.
  npy_intp dims[1] = {out_size};
  return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, (void*)(output));
}

PyObject* ImprintingEnginePythonWrapper::get_inference_time() const {
  ENSURE_ENGINE_INIT();
  float time;
  ENSURE_ENGINE_STATUS(engine_->get_inference_time(&time));
  return PyFloat_FromDouble(time);
}

}  // namespace imprinting
}  // namespace learn
}  // namespace coral
