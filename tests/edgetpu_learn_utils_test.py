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
import os
import shutil
import tempfile
import unittest

from . import test_utils
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.learn.utils import AppendFullyConnectedAndSoftmaxLayerToModel
import numpy as np


class EdgeTpuLearnUtilsTest(unittest.TestCase):

  def test_append_fully_connected_and_softmax_layer_to_model(self):
    in_model_name = (
        'mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite')
    in_model_path = test_utils.test_data_path(in_model_name)
    in_engine = BasicEngine(in_model_path)

    # Generate random input tensor.
    np.random.seed(12345)
    input_tensor = np.random.randint(
        0, 255, size=in_engine.get_input_tensor_shape(),
        dtype=np.uint8).flatten()

    # Set up weights, biases and FC output tensor range.
    _, embedding_vector = in_engine.run_inference(input_tensor)
    embedding_vector_dim = embedding_vector.shape[0]
    num_classes = 10
    weights = np.random.randn(embedding_vector_dim,
                              num_classes).astype(np.float32)
    biases = np.random.randn(num_classes).astype(np.float32)

    fc_output = embedding_vector.dot(weights) + biases
    fc_output_min = float(np.min(fc_output))
    fc_output_max = float(np.max(fc_output))

    try:
      # Create temporary directory, and remove this folder when test
      # finishes. Otherwise, test may fail because of generated files from
      # previous run.
      tmp_dir = tempfile.mkdtemp()
      # Append FC and softmax layers and save model.
      out_model_path = os.path.join(tmp_dir,
                                    in_model_name[:-7] + '_bp_retrained.tflite')
      AppendFullyConnectedAndSoftmaxLayerToModel(in_model_path, out_model_path,
                                                 weights.transpose().flatten(),
                                                 biases, fc_output_min,
                                                 fc_output_max)
      self.assertTrue(os.path.exists(out_model_path))

      # Run with saved model on same input.
      out_engine = BasicEngine(out_model_path)
      _, result = out_engine.run_inference(input_tensor)
      # Calculate expected result.
      expected = np.exp(fc_output - np.max(fc_output))
      expected = expected / np.sum(expected)
      np.testing.assert_almost_equal(result, expected, decimal=2)
    finally:
      shutil.rmtree(tmp_dir)

  def test_append_fully_connected_and_softmax_layer_with_invalid_input_path(self):
    in_model_name = 'invalid_file_path.tflite'
    in_model_path = test_utils.test_data_path(in_model_name)
    embedding_vector_dim = 1024
    num_classes = 10
    weights = np.random.randn(embedding_vector_dim,
                              num_classes).astype(np.float32)
    biases = np.random.randn(num_classes).astype(np.float32)
    fc_output_min = float(-1.0)
    fc_output_max = float(1.0)
    out_model_path = '/tmp/output_path_not_used'

    try:
      AppendFullyConnectedAndSoftmaxLayerToModel(in_model_path, out_model_path,
                                                 weights.transpose().flatten(),
                                                 biases, fc_output_min,
                                                 fc_output_max)
    except RuntimeError as e:
      error_message = str(e)

    expected_message = 'Failed to open file: {}'.format(in_model_path)
    self.assertEqual(error_message, expected_message)
