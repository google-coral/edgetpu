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

import math
import unittest

from . import test_utils
from edgetpu.basic import edgetpu_utils
from edgetpu.basic.basic_engine import BasicEngine


class TestBasicEnginePythonAPI(unittest.TestCase):

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
            'mobilenet_ssd_v1_coco_quant_postprocess.tflite'))
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

  def test_run_inference(self):
    for model in test_utils.get_model_list():
      print('Testing model :', model)
      engine = BasicEngine(test_utils.test_data_path(model))
      input_data = test_utils.generate_random_input(
          1, engine.required_input_array_size())
      latency, ret = engine.run_inference(input_data)
      self.assertEqual(ret.size, engine.total_output_array_size())
      # Check debugging functions.
      self.assertLess(math.fabs(engine.get_inference_time() - latency), 0.001)
      raw_output = engine.get_raw_output()
      self.assertEqual(ret.size, raw_output.size)
      for i in range(ret.size):
        if math.isnan(ret[i]) and math.isnan(raw_output[i]):
          continue
        self.assertLess(math.fabs(ret[i] - raw_output[i]), 0.001)

  def test_device_path(self):
    all_edgetpu_paths = edgetpu_utils.ListEdgeTpuPaths(
        edgetpu_utils.EDGE_TPU_STATE_NONE)
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'),
        all_edgetpu_paths[0])
    self.assertEqual(engine.device_path(), all_edgetpu_paths[0])

if __name__ == '__main__':
  unittest.main()
