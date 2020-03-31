# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic inference engine."""

import contextlib
import ctypes
import numpy
from ctypes.util import find_library
from edgetpu.swig.edgetpu_cpp_wrapper import BasicEnginePythonWrapper
from edgetpu.utils.warning import deprecated

# ctypes definition of GstMapInfo. This is a stable API, guaranteed to be
# ABI compatible for any past and future GStreamer 1.0 releases.
# Used to get the underlying memory pointer without any copies, and without
# native library linking against libgstreamer.
class _GstMapInfo(ctypes.Structure):
  _fields_ = [('memory', ctypes.c_void_p),                # GstMemory *memory
              ('flags', ctypes.c_int),                    # GstMapFlags flags
              ('data', ctypes.c_void_p),                  # guint8 *data
              ('size', ctypes.c_size_t),                  # gsize size
              ('maxsize', ctypes.c_size_t),               # gsize maxsize
              ('user_data', ctypes.c_void_p * 4),         # gpointer user_data[4]
              ('_gst_reserved', ctypes.c_void_p * 4)]     # GST_PADDING

# Try to import GStreamer but don't fail if it's not available. If not available
# we're probably not getting GStreamer buffers as input anyway.
_libgst = None
try:
  import gi
  gi.require_version('Gst', '1.0')
  from gi.repository import Gst
  _libgst = ctypes.CDLL(find_library('gstreamer-1.0'))
  _libgst.gst_buffer_map.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GstMapInfo), ctypes.c_int]
  _libgst.gst_buffer_map.restype = ctypes.c_int
  _libgst.gst_buffer_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GstMapInfo)]
  _libgst.gst_buffer_unmap.restype = None
except (ImportError, ValueError, OSError):
  pass

def _is_valid_ctypes_input(input):
  if not isinstance(input, tuple):
    return False
  pointer, size = input
  if not isinstance(pointer, ctypes.c_void_p):
    return False
  return isinstance(size, int)

@contextlib.contextmanager
def _gst_buffer_map(buffer):
  mapping = _GstMapInfo()
  ptr = hash(buffer)
  success = _libgst.gst_buffer_map(ptr, mapping, Gst.MapFlags.READ)
  if not success:
    raise RuntimeError('gst_buffer_map failed')
  try:
    yield ctypes.c_void_p(mapping.data), mapping.size
  finally:
    _libgst.gst_buffer_unmap(ptr, mapping)

class BasicEngine(object):
  """Base inference engine to execute a TensorFlow Lite model on the Edge TPU."""

  def __init__(self, model_path, device_path=None):
    """
    Args:
      model_path (str): Path to a TensorFlow Lite (``.tflite``) file. This model
        must be `compiled for the Edge TPU
        <https://coral.ai/docs/edgetpu/compiler/>`_; otherwise, it simply
        executes on the host CPU.
      device_path (str): The device path for the Edge TPU this engine should
        use. This argument is needed only when you have multiple Edge TPUs and
        more inference engines than available Edge TPUs. For details, read `how
        to use multiple Edge TPUs
        <https://coral.ai/docs/edgetpu/multiple-edgetpu/>`_.
    """
    if device_path:
      self._engine = BasicEnginePythonWrapper.CreateFromFile(
          model_path, device_path)
    else:
      self._engine = BasicEnginePythonWrapper.CreateFromFile(model_path)

  def run_inference(self, input, size=None):
    """Performs inference with a raw input tensor.

    Args:
      input: (:obj:`numpy.ndarray`): A 1-D array as the input tensor. You can
        query the required size for this array with
        :func:`required_input_array_size`.
      size (int): input buffer size. When size is not None, it will throw exception if
        size does not match the expected input size, denoted by n. When size is None,
        it will throw exception when total input buffer size is smaller than n, and only use
        the first n bytes of the input buffer to set the input tensor, ignoring the remaining
        bytes if any in the buffer. (This behavior allows callers to use input buffers with
        padding bytes at the end, and have extra sanity check that the input matches the
        model's expectation.)

    Returns:
      A 2-tuple with the inference latency in milliseconds (float) and a 1-D array
      (:obj:`numpy.ndarray`) representing the output tensor. If there are multiple output tensors,
      they are compressed into a single 1-D array. For example, if the model outputs 2 tensors with
      values [1, 2, 3] and [0.1, 0.4, 0.9], the returned 1-D array is [1, 2, 3, 0.1, 0.4, 0.9]. You
      can calculate the size and offset for each tensor using :func:`get_all_output_tensors_sizes`,
      :func:`get_num_of_output_tensors`, and :func:`get_output_tensor_size`.
      Note that the inference result array is a reference, which needs to be deep copied if it
      needs to be preserved before next inference call.
    """
    expected_input_size = self.required_input_array_size()
    if size:
      assert size == expected_input_size, 'Wrong input size={}, expected={}.'.format(
          size, expected_input_size)
    if isinstance(input, bytes):
      assert len(input) >= expected_input_size
      result = self._engine.RunInferenceBytes(input)
    elif _is_valid_ctypes_input(input):
      pointer, actual_size = input
      assert actual_size >= expected_input_size
      result = self._engine.RunInferenceRaw(pointer.value, expected_input_size)
    elif _libgst and isinstance(input, Gst.Buffer):
      with _gst_buffer_map(input) as (pointer, actual_size):
        assert actual_size >= expected_input_size
        result = self._engine.RunInferenceRaw(pointer.value, expected_input_size)
    else:
      assert len(input) >= expected_input_size
      result = self._engine.RunInference(input)
    latency = self._engine.get_inference_time()
    return (latency, result)

  def get_input_tensor_shape(self):
    """Gets the shape required for the input tensor.

    For models trained for image classification / detection, the shape is always
    [1, height, width, channels]. To be used as input for :func:`run_inference`,
    this tensor shape must be flattened into a 1-D array with size ``height *
    width * channels``. To instead get that 1-D array size, use
    :func:`required_input_array_size`.

    Returns:
      A 1-D array (:obj:`numpy.ndarray`) representing the required input tensor
      shape.
    """
    return self._engine.get_input_tensor_shape()

  def get_all_output_tensors_sizes(self):
    """Gets the size of each output tensor.

    A model may output several tensors, but the return from :func:`run_inference`
    and :func:`get_raw_output` concatenates them together into a 1-D array.
    So this function provides the size for each original output tensor, allowing
    you to calculate the offset for each tensor within the concatenated array.

    Returns:
      An array (:obj:`numpy.ndarray`) with the length of each output tensor
      (this assumes that all output tensors are 1-D).
    """
    return self._engine.get_all_output_tensors_sizes()

  def get_num_of_output_tensors(self):
    """Gets the number of output tensors.

    Returns:
      An integer representing number of output tensors.
    """
    return self._engine.get_num_of_output_tensors()

  def get_output_tensor_size(self, index):
    """Gets the size of a specific output tensor.

    Args:
      index (int): The index position of the output tensor.

    Returns:
      An integer representing the size of the output tensor.
    """
    return self._engine.get_output_tensor_size(index)

  def total_output_array_size(self):
    """Gets the expected size of the 1-D output array returned by :func:`run_inference` and
    :func:`get_raw_output`.

    Returns:
      An integer representing the output tensor size.
    """
    return self._engine.total_output_array_size()

  def get_inference_time(self):
    """Gets the latency of the most recent inference.

    This can be used by higher level engines for debugging.

    Returns:
      A float representing the inference latency (in milliseconds).
    """
    return self._engine.get_inference_time()

  def model_path(self):
    """Gets the file path for model loaded by this inference engine.

    Returns:
      A string representing the model file's path.
    """
    return self._engine.model_path()

  def device_path(self):
    """Gets the path for the Edge TPU that's associated with this inference engine.

    See `how to run multiple models with multiple Edge TPUs
    <https://coral.ai/docs/edgetpu/multiple-edgetpu/>`_.

    Returns:
      A string representing this engine's Edge TPU device path.
    """
    return self._engine.device_path()

  def required_input_array_size(self):
    """Gets the required size for the ``input_tensor`` given to :func:`run_inference`.

    This is the total size of the 1-D array, once the tensor shape is flattened.

    Returns:
      An integer representing the required input tensor size.
    """
    return self._engine.required_input_array_size()

  def get_raw_output(self):
    """Gets the output of the most recent inference.

    This can be used by higher level engines for debugging.

    Returns:
      A 1-D array (:obj:`numpy.ndarray`) representing the output tensor. If
      there are multiple output tensors, they are compressed into a single 1-D
      array. (Same as what's returned by :func:`run_inference`.)
    """
    return self._engine.get_raw_output()

  @deprecated
  def RunInference(self, input):
    return self.run_inference(input)
