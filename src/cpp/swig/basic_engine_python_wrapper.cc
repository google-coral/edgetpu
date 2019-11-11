#include "src/cpp/swig/basic_engine_python_wrapper.h"

#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>

#include "src/cpp/error_reporter.h"

namespace coral {
namespace {
#define ENSURE_ENGINE_STATUS(status)                         \
  do {                                                       \
    if ((status) == kEdgeTpuApiError) {                      \
      PyErr_SetString(PyExc_RuntimeError,                    \
                      engine_->get_error_message().c_str()); \
      return nullptr;                                        \
    }                                                        \
  } while (0)

#define ENSURE_ENGINE_INIT()                                      \
  do {                                                            \
    if (!engine_) {                                               \
      PyErr_SetString(PyExc_RuntimeError,                         \
                      "BasicEnginePythonWrapper uninitialized!"); \
      return nullptr;                                             \
    }                                                             \
  } while (0)

}  // namespace

BasicEnginePythonWrapper::BasicEnginePythonWrapper() {
  // This function must be called in the initialization section of a module that
  // will make use of the C-API (PyArray_SimpleNewFromData).
  // It imports the module where the function-pointer table is stored and points
  // the correct variable to it.
  // Different with import_array() import_array1() has return value.
  // https://docs.scipy.org/doc/numpy-1.14.2/reference/c-api.array.html
  import_array1();
}

std::string BasicEnginePythonWrapper::Init(const std::string& model_path) {
  BasicEngineNativeBuilder builder(model_path);
  builder(&engine_);
  return builder.get_error_message();
}

std::string BasicEnginePythonWrapper::Init(const std::string& model_path,
                                           const std::string& device_path) {
  BasicEngineNativeBuilder builder(model_path, device_path);
  builder(&engine_);
  return builder.get_error_message();
}

PyObject* BasicEnginePythonWrapper::RunInference(const uint8_t* input,
                                                 int in_size) {
  ENSURE_ENGINE_INIT();
  float const* output;
  int out_size;
  EdgeTpuApiStatus status;
  // Let RunInference function play nicely with Python threading.
  Py_BEGIN_ALLOW_THREADS;
  status = engine_->RunInference(input, in_size, &output, &out_size);
  Py_END_ALLOW_THREADS;
  // Report errors after taking the GIL, or we crash.
  ENSURE_ENGINE_STATUS(status);
  // Parse results.
  npy_intp dims[1] = {out_size};
  return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, (void*)(output));
}

PyObject* BasicEnginePythonWrapper::get_input_tensor_shape() const {
  ENSURE_ENGINE_INIT();
  int* output = nullptr;
  int size = 0;
  ENSURE_ENGINE_STATUS(
      engine_->get_input_tensor_shape((int const**)&output, &size));
  npy_intp dims[1] = {size};
  return PyArray_SimpleNewFromData(1, dims, NPY_INT, (void*)output);
}

PyObject* BasicEnginePythonWrapper::get_all_output_tensors_sizes() const {
  ENSURE_ENGINE_INIT();
  int* output = nullptr;
  int size = 0;
  ENSURE_ENGINE_STATUS(
      engine_->get_all_output_tensors_sizes((int const**)&output, &size));
  npy_intp dims[1] = {size};
  return PyArray_SimpleNewFromData(1, dims, NPY_INT, (void*)output);
}

PyObject* BasicEnginePythonWrapper::get_num_of_output_tensors() const {
  ENSURE_ENGINE_INIT();
  int ret;
  ENSURE_ENGINE_STATUS(engine_->get_num_of_output_tensors(&ret));
  return PyLong_FromLong(ret);
}

PyObject* BasicEnginePythonWrapper::get_output_tensor_size(
    int tensor_index) const {
  ENSURE_ENGINE_INIT();
  int ret;
  ENSURE_ENGINE_STATUS(engine_->get_output_tensor_size(tensor_index, &ret));
  return PyLong_FromLong(ret);
}

PyObject* BasicEnginePythonWrapper::required_input_array_size() const {
  ENSURE_ENGINE_INIT();
  int size;
  ENSURE_ENGINE_STATUS(engine_->get_input_array_size(&size));
  return PyLong_FromLong(size);
}

PyObject* BasicEnginePythonWrapper::total_output_array_size() const {
  ENSURE_ENGINE_INIT();
  int size;
  ENSURE_ENGINE_STATUS(engine_->total_output_array_size(&size));
  return PyLong_FromLong(size);
}

PyObject* BasicEnginePythonWrapper::get_raw_output() const {
  ENSURE_ENGINE_INIT();
  float* output = nullptr;
  int size = 0;
  ENSURE_ENGINE_STATUS(engine_->get_raw_output((float const**)&output, &size));
  npy_intp dims[1] = {size};
  return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, (void*)(output));
}

PyObject* BasicEnginePythonWrapper::get_inference_time() const {
  ENSURE_ENGINE_INIT();
  float time;
  ENSURE_ENGINE_STATUS(engine_->get_inference_time(&time));
  return PyFloat_FromDouble(time);
}

}  // namespace coral
