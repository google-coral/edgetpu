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

"""Evaluates the accuracy of imprinting based transfer learning model."""

import collections
import contextlib
import os
import unittest

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
from PIL import Image

from . import test_utils

@contextlib.contextmanager
def test_image(path):
  with open(path, 'rb') as f:
    with Image.open(f) as image:
      yield image

class ImprintingEngineEvaluationTest(unittest.TestCase):

  @staticmethod
  def _get_input_tensor_shape(model_path):
    """Gets input tensor shape of given model.

    Args:
      model_path: string, path of the model.

    Returns:
      List of integers.
    """
    tmp = BasicEngine(model_path)
    shape = tmp.get_input_tensor_shape()
    return shape.copy()

  @staticmethod
  def _get_output_number_classes(model_path):
    """Gets the number of output classes.

    Args:
      model_path: string, path of the model.

    Returns:
      int, number of the output classes.
    """
    tmp = BasicEngine(model_path)
    assert tmp.get_num_of_output_tensors() == 1
    return tmp.total_output_array_size()

  def _transfer_learn_and_evaluate(self, model_path, keep_classes, dataset_path,
                                test_ratio, top_k_range):
    """Transfer-learns with given params and returns the evaluatoin result.

    Args:
      model_path: string, path of the base model.
      keep_classes: bool, whether to keep base model classes.
      dataset_path: string, path to the directory of dataset. The images
        should be put under sub-directory named by category.
      test_ratio: float, the ratio of images used for test.
      top_k_range: int, top_k range to be evaluated. The function will return
        accuracy from top 1 to top k.

    Returns:
      list of float numbers.
    """
    print('---------------      Parsing dataset      ----------------')
    print('Dataset path:', dataset_path)

    # train in fixed order to ensure the same evaluation result.
    train_set, test_set = test_utils.prepare_data_set_from_directory(
        dataset_path, test_ratio, True)

    print('Image list successfully parsed! Number of Categories = ',
          len(train_set))
    input_shape = self._get_input_tensor_shape(model_path)
    required_image_shape = (input_shape[2], input_shape[1])  # (width, height)
    print('---------------  Processing training data ----------------')
    print('This process may take more than 30 seconds.')
    num_classes = self._get_output_number_classes(model_path) if keep_classes else 0
    train_input = []
    labels_map = {}
    for class_id, (category, image_list) in enumerate(train_set.items()):
      print('Processing {} ({} images)'.format(category, len(image_list)))
      train_input.append(
          test_utils.prepare_images(
            image_list,
            os.path.join(dataset_path, category),
            required_image_shape
          )
      )
      labels_map[num_classes + class_id] = category

    # train
    print('----------------      Start training     -----------------')
    imprinting_engine = ImprintingEngine(model_path, keep_classes)
    imprinting_engine.train_all(train_input)
    print('----------------     Training finished   -----------------')
    with test_utils.TemporaryFile(suffix='.tflite') as output_model_path:
      imprinting_engine.save_model(output_model_path.name)

      # Evaluate
      print('----------------     Start evaluating    -----------------')
      classification_engine = ClassificationEngine(output_model_path.name)
      # top[i] represents number of top (i+1) correct inference.
      top_k_correct_count = [0] * top_k_range
      image_num = 0
      for category, image_list in test_set.items():
        n = len(image_list)
        print('Evaluating {} ({} images)'.format(category, n))
        for image_name in image_list:
          with test_image(os.path.join(dataset_path, category, image_name)) as raw_image:
            # Set threshold as a negative number to ensure we get top k candidates
            # even if its score is 0.
            candidates = classification_engine.classify_with_image(
                raw_image, threshold=-0.1, top_k=top_k_range)
            for i in range(len(candidates)):
              if candidates[i][0] in labels_map and labels_map[candidates[i][0]] == category:
                top_k_correct_count[i] += 1
                break
        image_num += n
      for i in range(1, top_k_range):
        top_k_correct_count[i] += top_k_correct_count[i-1]

    return [top_k_correct_count[i] / image_num for i in range(top_k_range)]

  def _test_oxford17_flowers_single(self, model_path, keep_classes, expected):
    top_k_range = len(expected)
    ret = self._transfer_learn_and_evaluate(
        test_utils.test_data_path(model_path),
        keep_classes,
        test_utils.test_data_path('oxford_17flowers'),
        0.25,
        top_k_range
    )
    for i in range(top_k_range):
      self.assertGreaterEqual(ret[i], expected[i])

  # Evaluate with L2Norm full model, not keeping base model classes.
  def test_oxford17_flowers_l2_norm_model_not_keep_classes(self):
    self._test_oxford17_flowers_single(
        'mobilenet_v1_1.0_224_l2norm_quant.tflite',
        keep_classes=False,
        expected=[0.86, 0.94, 0.96, 0.97, 0.97]
    )

  # Evaluate with L2Norm full model, keeping base model classes.
  def test_oxford17_flowers_l2_norm_model_keep_classes(self):
    self._test_oxford17_flowers_single(
        'mobilenet_v1_1.0_224_l2norm_quant.tflite',
        keep_classes=True,
        expected=[0.86, 0.94, 0.96, 0.96, 0.97]
    )

if __name__ == '__main__':
  unittest.main()
