%module edgetpu_cpp_wrapper

%include "stdint.i"
%include "std_string.i"

%{
#define SWIG_FILE_WITH_INIT
#include "src/cpp/error_reporter.h"
#include "src/cpp/learn/imprinting/engine.h"
#include "src/cpp/learn/utils.h"
#include "src/cpp/version.h"
#include "src/cpp/swig/basic_engine_python_wrapper.h"
#include "src/cpp/swig/imprinting_engine_python_wrapper.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%numpy_typemaps(uint8_t, NPY_UBYTE, int)
%apply (uint8_t* IN_ARRAY1, int DIM1) {
    (const uint8_t* input, int in_size)
}
%apply (uint8_t* IN_ARRAY2, int DIM1, int DIM2 ) {
    (const uint8_t* input, int dim1, int dim2)
}
%numpy_typemaps(float, NPY_FLOAT, int)
%apply (float* IN_ARRAY1, int DIM1) {
    (const float* weights, int weights_size),
    (const float* biases, int biases_size)
}

%feature("docstring") AppendFullyConnectedAndSoftmaxLayerToModel
"""Appends Fully-connected (FC) and softmax layer to input tflite model.

This function assumes the input tflite model is an embedding extractor, e.g., a
classification model without the last FC+Softmax layer. It does the following:
  *) Quantizes learned weights and biases from float32 to uint8;
  *) Appends quantized weights and biases as FC layer;
  *) Adds a Softmax layer;
  *) Stores the result in tflite file format specified by `out_model_path`;

Args:
  in_model_path: string, path to input tflite model;
  out_model_path: string, path to output tflite model;
  weights: 1 dimensional float32 np.ndarray, flattened learned weights. Learned
    weights is a num_classes x embedding_vector_dim matrix;
  biases: 1 dimensional float32 np.ndarray of length num_classes;
  out_tensor_min: float, expected min value of FC layer, for quantization parameter;
  out_tensor_max: float, expected max value of FC layer, for quantization parameter;

Raises:
  RuntimeError: with corresponding reason for failure.
"""

%feature("docstring") ListEdgeTpuPaths
"""Lists the paths for all available Edge TPU devices.

Args:
  state (int): The current state of devices you want to list. It can be one of
    :attr:`~edgetpu.swig.edgetpu_cpp_wrapper.edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_ASSIGNED`,
    :attr:`~edgetpu.swig.edgetpu_cpp_wrapper.edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED`,
    or :attr:`~edgetpu.swig.edgetpu_cpp_wrapper.edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_NONE`.

Returns:
  A tuple of strings, each representing a device path.
"""

%feature("docstring") coral::GetRuntimeVersion
"""Returns the Edge TPU runtime (libedgetpu.so) version.

Compare this version to the value of
:attr:`~edgetpu.swig.edgetpu_cpp_wrapper.edgetpu.basic.edgetpu_utils.SUPPORTED_RUNTIME_VERSION`,
which is the runtime version required by your Edge TPU library version.

This runtime version is dynamically retrieved from the shared object.

Returns:
  A string for the version name."""

%include "src/cpp/swig/basic_engine_python_wrapper.h"
%include "src/cpp/swig/imprinting_engine_python_wrapper.h"

namespace coral {
// Version of edgetpu_cpp_wrapper library.
// The version is a string with format 'API(version) + TF(version)'.
// API version corresponds to this C++ source code while TF version is the
// commit number of tensorflow submodule when compiling `edgetpu-native` repo.
extern const char kEdgeTpuCppWrapperVersion[];

// Supported runtime version. This is the runtime version required by this
// edgetpu_cpp_wrapper library.
extern const char kSupportedRuntimeVersion[];

// Returns version of the EdgeTpu runtime (libedgetpu.so).
// The version is dynamically retrieved from shared object.
std::string GetRuntimeVersion();

%extend BasicEnginePythonWrapper {
  // Version of the constructor that handles producing Python exceptions
  // that propagate strings.
  static PyObject* CreateFromFile(const std::string& model_path) {
    coral::BasicEnginePythonWrapper* wrapper =
        new coral::BasicEnginePythonWrapper();
    const std::string& error_message = wrapper->Init(model_path);
    if (error_message.empty()) {
      return SWIG_NewPointerObj(wrapper,
                                SWIGTYPE_p_coral__BasicEnginePythonWrapper,
                                /*flag=*/SWIG_POINTER_OWN);
    } else {
      PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
      return nullptr;
    }
  }

  static PyObject* CreateFromFile(const std::string& model_path,
                                  const std::string& device_path) {
    coral::BasicEnginePythonWrapper* wrapper =
        new coral::BasicEnginePythonWrapper();
    const std::string& error_message = wrapper->Init(model_path, device_path);
    if (error_message.empty()) {
      return SWIG_NewPointerObj(wrapper,
                                SWIGTYPE_p_coral__BasicEnginePythonWrapper,
                                /*flag=*/SWIG_POINTER_OWN);
    } else {
      PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
      return nullptr;
    }
  }

  // numpy typemap greedily applies to any method with params
  // const uint8_t* input, int in_size so we can't easily map
  // bytes too.
  PyObject* RunInferenceBytes(PyObject* input_obj) {
    return $self->RunInference(
        reinterpret_cast<uint8_t*>(PyBytes_AS_STRING(input_obj)),
        static_cast<int>(PyBytes_GET_SIZE(input_obj)));
  }

  // raw pointer stored as PyLong, size as PyLong.
  PyObject* BasicEnginePythonWrapper::RunInferenceRaw(PyObject* input_obj,
                                                      PyObject* size_obj) {
    return $self->RunInference(
        reinterpret_cast<uint8_t*>(PyLong_AsVoidPtr(input_obj)),
        static_cast<int>(PyLong_AsLong(size_obj)));
  }
}


namespace learn {
namespace imprinting {
%extend ImprintingEnginePythonWrapper {
  static PyObject* CreateFromFile(const std::string& model_path,
                                  bool keep_classes) {
    coral::learn::imprinting::ImprintingEnginePythonWrapper* wrapper =
        new coral::learn::imprinting::ImprintingEnginePythonWrapper();
    const std::string& error_message = wrapper->Init(model_path, keep_classes);
    if (error_message.empty()) {
      return SWIG_NewPointerObj(
          wrapper,
          SWIGTYPE_p_coral__learn__imprinting__ImprintingEnginePythonWrapper,
          /*flag=*/SWIG_POINTER_OWN);
    } else {
      PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
      return nullptr;
    }
  }
}
}  // namespace imprinting
}  // namespace learn
}  // namespace coral

%inline {
// TODO: Consider moving EdgeTpuState out of EdgeTpuResourceManager
// and use the same definition here.
enum class EdgeTpuState {
  kNone,
  kAssigned,
  kUnassigned,
};

PyObject* ListEdgeTpuPaths(const EdgeTpuState state) {
  std::vector<std::string> result;
  switch (state) {
    case EdgeTpuState::kNone:
      result = coral::EdgeTpuResourceManager::GetSingleton()->ListEdgeTpuPaths(
          coral::EdgeTpuResourceManager::EdgeTpuState::kNone);
      break;
    case EdgeTpuState::kAssigned:
      result = coral::EdgeTpuResourceManager::GetSingleton()->ListEdgeTpuPaths(
          coral::EdgeTpuResourceManager::EdgeTpuState::kAssigned);
      break;
    case EdgeTpuState::kUnassigned:
      result = coral::EdgeTpuResourceManager::GetSingleton()->ListEdgeTpuPaths(
          coral::EdgeTpuResourceManager::EdgeTpuState::kUnassigned);
      break;
  }
  PyObject* tuple = PyTuple_New(result.size());
  for (int i = 0; i < result.size(); ++i)
    PyTuple_SetItem(tuple, i, SWIG_From_std_string(result[i]));
  return tuple;
}

PyObject* AppendFullyConnectedAndSoftmaxLayerToModel(
    const std::string& in_model_path, const std::string& out_model_path,
    const float* weights, int weights_size, const float* biases,
    int biases_size, float out_tensor_min, float out_tensor_max) {
  coral::EdgeTpuErrorReporter reporter;
  const auto& status = coral::learn::AppendFullyConnectedAndSoftmaxLayerToModel(
               in_model_path, out_model_path, weights, weights_size, biases,
               biases_size, out_tensor_min, out_tensor_max, &reporter);
  if(status == coral::kEdgeTpuApiError) {
    PyErr_SetString(PyExc_RuntimeError, reporter.message().c_str());
    return nullptr;
  }
  Py_RETURN_NONE;
}

// SWIG_From_std_string is defined in SWIG generated cpp code. So we can't
// implement following functions in basic_engine_python_wrapper.cc except we
// replace it with other string --> PyObject convert method such as
// PyUnicode_FromString().
// Implementing them here can re-use SWIG_From_std_string function, which is
// reliable and robust.
PyObject* coral::BasicEnginePythonWrapper::model_path() const {
  if (!engine_) {
    PyErr_SetString(PyExc_RuntimeError,
                    "BasicEnginePythonWrapper uninitialized!");
    return nullptr;
  }
  std::string path;
  if (engine_->model_path(&path) == kEdgeTpuApiError) {
    PyErr_SetString(PyExc_RuntimeError, engine_->get_error_message().c_str());
    return nullptr;
  }
  return SWIG_From_std_string(path);
}

PyObject* coral::BasicEnginePythonWrapper::device_path() const {
  if (!engine_) {
    PyErr_SetString(PyExc_RuntimeError,
                    "BasicEnginePythonWrapper uninitialized!");
    return nullptr;
  }
  std::string path;
  if (engine_->device_path(&path) == kEdgeTpuApiError) {
    PyErr_SetString(PyExc_RuntimeError, engine_->get_error_message().c_str());
    return nullptr;
  }
  return SWIG_From_std_string(path);
}

}
