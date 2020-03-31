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

import tempfile
import unittest

from . import test_utils
from edgetpu.basic import edgetpu_utils
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine


class TestExceptions(unittest.TestCase):

  def test_invalid_model_path(self):
    error_message = None
    try:
      _ = BasicEngine('invalid_model_path.tflite')
    except RuntimeError as e:
      error_message = str(e)
    self.assertEqual('Could not open \'invalid_model_path.tflite\'.',
                     error_message)

  def test_negative_tensor_index(self):
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    error_message = None
    try:
      engine.get_output_tensor_size(-1)
    except RuntimeError as e:
      error_message = str(e)
    self.assertEqual('tensor_index must >= 0!', error_message)

  def test_tensor_index_exceed(self):
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    error_message = None
    try:
      engine.get_output_tensor_size(100)
    except RuntimeError as e:
      error_message = str(e)
    self.assertEqual('tensor_index doesn\'t exist!', error_message)

  def test_inference_with_bad_input_size(self):
    engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    expected_size = engine.required_input_array_size()
    input_data = test_utils.generate_random_input(1, expected_size - 1)
    error_message = None
    try:
      engine.run_inference(input_data, expected_size - 1)
    except AssertionError as e:
      error_message = str(e)
    self.assertEqual('Wrong input size={}, expected={}.'.format(
        expected_size - 1, expected_size), error_message)

  def test_imprinting_engine_saving_without_training(self):
    model_list = [
        'mobilenet_v1_1.0_224_l2norm_quant.tflite',
        'mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite'
    ]
    for model in model_list:
      error_message = None
      engine = ImprintingEngine(
          test_utils.test_data_path(model), keep_classes=False)
      try:
        with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
          engine.save_model(output_model_path.name)
      except RuntimeError as e:
        error_message = str(e)
      self.assertEqual('Model without training won\'t be saved!', error_message)

  def test_imprinting_engine_invalid_model_path(self):
    error_message = None
    try:
      _ = ImprintingEngine('invalid_model_path.tflite')
    except RuntimeError as e:
      error_message = str(e)
    self.assertEqual('Failed to open file: invalid_model_path.tflite',
                     error_message)

  def test_imprinting_engine_load_extractor_with_wrong_format(self):
    error_message = None
    try:
      _ = ImprintingEngine(
          test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    except RuntimeError as e:
      error_message = str(e)
    self.assertEqual(
        'Unsupported model architecture. Input model must have an L2Norm layer.',
        error_message)

  def test_edge_tpu_not_exist(self):
    error_message = None
    try:
      _ = BasicEngine(
          test_utils.test_data_path('mobilenet_v1_1.0_224_quant_edgetpu.tflite'),
          'invalid_edgetpu_device_path')
    except RuntimeError as e:
      error_message = str(e)
    self.assertEqual(
        'Path invalid_edgetpu_device_path does not map to an Edge TPU device.',
        error_message)

  def test_exhaust_all_edge_tpus(self):
    edge_tpus = edgetpu_utils.ListEdgeTpuPaths(
        edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
    # No need to test if there's only one Edge TPU available.
    if len(edge_tpus) <= 1:
      return
    model_path = test_utils.test_data_path(
        'mobilenet_v1_1.0_224_quant_edgetpu.tflite')
    unused_basic_engines = []
    for _ in edge_tpus:
      unused_basic_engines.append(BasicEngine(model_path))

    # Request one more Edge TPU to trigger the exception.
    error_message = None
    expected_message = (
        'Multiple Edge TPUs detected and all have been mapped to at least one '
        'model. If you want to share one Edge TPU with multiple models, '
        'specify `device_path` name.')
    try:
      _ = BasicEngine(model_path)
    except RuntimeError as e:
      error_message = str(e)
    self.assertEqual(expected_message, error_message)


if __name__ == '__main__':
  unittest.main()
