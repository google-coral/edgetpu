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

import ctypes
import math
import numpy
import unittest

from . import test_utils
from ctypes.util import find_library
from edgetpu.basic import edgetpu_utils
from edgetpu.basic.basic_engine import BasicEngine

# Detect whether GStreamer is available.
# This code session is copied from basic_engine.py.
class _GstMapInfo(ctypes.Structure):
  _fields_ = [('memory', ctypes.c_void_p),                # GstMemory *memory
              ('flags', ctypes.c_int),                    # GstMapFlags flags
              ('data', ctypes.c_void_p),                  # guint8 *data
              ('size', ctypes.c_size_t),                  # gsize size
              ('maxsize', ctypes.c_size_t),               # gsize maxsize
              ('user_data', ctypes.c_void_p * 4),         # gpointer user_data[4]
              ('_gst_reserved', ctypes.c_void_p * 4)]     # GST_PADDING
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
  Gst.init(None)
except (ImportError, ValueError, OSError):
  pass

class TestBasicEnginePythonAPI(unittest.TestCase):

  def _test_inference_with_different_input_types(
      self, engine, input_data, input_size=None):
    """Test inference with different input types. It doesn't check correctness
       of inference. Instead it checks inference repeatability with different
       input types.

    Args:
      input_data (list): A 1-D list as the input tensor.
      input_size (int): input buffer size.
    """
    expect_total_output_size = engine.total_output_array_size()
    # list
    latency, ret = engine.run_inference(input_data, input_size)
    self.assertEqual(ret.size, expect_total_output_size)
    ret0 = numpy.copy(ret)
    # numpy
    np_input = numpy.asarray(input_data, numpy.uint8)
    latency, ret = engine.run_inference(np_input, input_size)
    self.assertTrue(numpy.array_equal(ret0, ret))
    # bytes
    bytes_input = bytes(input_data)
    latency, ret = engine.run_inference(bytes_input, input_size)
    self.assertTrue(numpy.array_equal(ret0, ret))
    # ctypes
    latency, ret = engine.run_inference((
        np_input.ctypes.data_as(ctypes.c_void_p), np_input.size),
        input_size)
    self.assertTrue(numpy.array_equal(ret0, ret))
    # Gst buffer
    if _libgst:
      gst_input = Gst.Buffer.new_wrapped(bytes_input)
      latency, ret = engine.run_inference(gst_input, input_size)
      self.assertTrue(numpy.array_equal(ret0, ret))
    else:
      print('Can not import gi. Skip test on Gst.Buffer input type.');

  def test_debug_info(self):
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    # Check model's input format.
    input_tensor_shape = engine.get_input_tensor_shape()
    self.assertListEqual([1, 224, 224, 3], input_tensor_shape.tolist())
    self.assertEqual(224 * 224 * 3, engine.required_input_array_size())

    # Check model's output.
    output_tensors_sizes = engine.get_all_output_tensors_sizes()
    self.assertListEqual([1001], output_tensors_sizes.tolist())
    self.assertEqual(1, engine.get_num_of_output_tensors())
    self.assertEqual(1001, engine.get_output_tensor_size(0))
    self.assertEqual(1001, engine.total_output_array_size())

    # Check SSD model.
    ssd_engine = BasicEngine(
        test_utils.test_data_path(
            'ssd_mobilenet_v1_coco_quant_postprocess.tflite'))
    # Check model's input format.
    input_tensor_shape = ssd_engine.get_input_tensor_shape()
    self.assertListEqual([1, 300, 300, 3], input_tensor_shape.tolist())
    self.assertEqual(300 * 300 * 3, ssd_engine.required_input_array_size())

    # Check model's output.
    output_tensors_sizes = ssd_engine.get_all_output_tensors_sizes()
    self.assertListEqual([80, 20, 20, 1], output_tensors_sizes.tolist())
    self.assertEqual(4, ssd_engine.get_num_of_output_tensors())
    self.assertEqual(80, ssd_engine.get_output_tensor_size(0))
    self.assertEqual(20, ssd_engine.get_output_tensor_size(1))
    self.assertEqual(20, ssd_engine.get_output_tensor_size(2))
    self.assertEqual(1, ssd_engine.get_output_tensor_size(3))
    self.assertEqual(121, ssd_engine.total_output_array_size())

  def test_run_inference_implicit_size_different_types(self):
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    input_size = engine.required_input_array_size()
    input_data = test_utils.generate_random_input(1, input_size)
    self._test_inference_with_different_input_types(engine, input_data)
    input_data = test_utils.generate_random_input(1, input_size + 1)
    self._test_inference_with_different_input_types(engine, input_data)
    input_data = test_utils.generate_random_input(1, input_size + 64)
    self._test_inference_with_different_input_types(engine, input_data)

  def test_run_inference_explicit_size_different_types(self):
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    input_size = engine.required_input_array_size()
    input_data = test_utils.generate_random_input(1, input_size)
    self._test_inference_with_different_input_types(
        engine, input_data, input_size)
    input_data = test_utils.generate_random_input(1, input_size + 1)
    self._test_inference_with_different_input_types(
        engine, input_data, input_size)
    input_data = test_utils.generate_random_input(1, input_size + 64)
    self._test_inference_with_different_input_types(
        engine, input_data, input_size)

  def test_device_path(self):
    all_edgetpu_paths = edgetpu_utils.ListEdgeTpuPaths(
        edgetpu_utils.EDGE_TPU_STATE_NONE)
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'),
        all_edgetpu_paths[0])
    self.assertEqual(engine.device_path(), all_edgetpu_paths[0])

if __name__ == '__main__':
  unittest.main()
