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
import tempfile
import unittest
from . import test_utils
from edgetpu.classification.engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
from PIL import Image


class TestImprintingEnginePythonAPI(unittest.TestCase):

  _MODEL_LIST = [
      test_utils.test_data_path(
          'imprinting/mobilenet_v1_1.0_224_l2norm_quant.tflite'),
      test_utils.test_data_path(
          'imprinting/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite'),
  ]

  @staticmethod
  def _get_image_shape(model_path):
    """Gets image shape required by given model.

    Args:
      model_path: string, path of the model.

    Returns:
      A tuple of width and height.
    """
    tmp = BasicEngine(model_path)
    input_shape = tmp.get_input_tensor_shape()
    return input_shape[2], input_shape[1]

  def _classify_image(self, engine, data_dir, image_name, label_id, score):
    with open(os.path.join(data_dir, image_name), 'rb') as f:
      with Image.open(f) as img:
        ret = engine.classify_with_image(img, top_k=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0][0], label_id)
        self.assertGreater(ret[0][1], score)

  def _classify_image_by_inference(self, engine, image_shape, data_dir, image_name, label_id, score):
    with open(os.path.join(data_dir, image_name), 'rb') as f:
      with Image.open(f) as img:
        img = img.resize((image_shape[0], image_shape[1]), Image.NEAREST)
        ret = engine.classify_with_resized_image(img, top_k=1)
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0][0], label_id)
        self.assertGreater(ret[0][1], score)

  def _train_and_test(self, model_path, output_model_path, training_datapoints,
                    test_datapoints, keep_classes=False):
    engine = ImprintingEngine(model_path, keep_classes)
    image_shape = self._get_image_shape(model_path)
    data_dir = test_utils.test_data_path('imprinting')
    # train.
    for training_datapoint in training_datapoints:
      engine.train(test_utils.prepare_images(training_datapoint['image_names'],
                                            data_dir, image_shape),
                   training_datapoint['label_id'])
    engine.save_model(output_model_path)

    # Test.
    engine = ClassificationEngine(output_model_path)
    self.assertEqual(1, engine.get_num_of_output_tensors())
    if not keep_classes:
      self.assertEqual(len(training_datapoints), engine.get_output_tensor_size(0))
    for test_datapoint in test_datapoints:
      self._classify_image(engine, data_dir, test_datapoint['image_name'],
                           test_datapoint['label_id'], test_datapoint['score'])

  def _train_and_test_run_inference(self, model_path, training_datapoints,
                    test_datapoints, keep_classes=False):
    engine = ImprintingEngine(model_path, keep_classes)
    image_shape = self._get_image_shape(model_path)
    data_dir = test_utils.test_data_path('imprinting')
    # train.
    for training_datapoint in training_datapoints:
      engine.train(test_utils.prepare_images(training_datapoint['image_names'],
                                            data_dir, image_shape),
                   training_datapoint['label_id'])
    # Test with running inference.
    for test_datapoint in test_datapoints:
      self._classify_image_by_inference(engine, image_shape, data_dir, test_datapoint['image_name'],
                                        test_datapoint['label_id'], test_datapoint['score'])

  # Test full model using run_inference, not keeping base model classes.
  def test_run_inference_training_l2_norm_model_not_keep_classes(self):
    training_datapoints = [
        {'image_names': ['cat_train_0.bmp'], 'label_id': 0},
        {'image_names': ['dog_train_0.bmp'], 'label_id': 1},
        {'image_names': ['hotdog_train_0.bmp', 'hotdog_train_1.bmp'], 'label_id': 2}
    ]
    test_datapoints = [
        {'image_name': 'cat_test_0.bmp', 'label_id': 0, 'score': 0.99},
        {'image_name': 'dog_test_0.bmp', 'label_id': 1, 'score': 0.99},
        {'image_name': 'hotdog_test_0.bmp', 'label_id': 2, 'score': 0.99}
    ]
    for model_path in self._MODEL_LIST:
      with self.subTest():
        self._train_and_test_run_inference(model_path, training_datapoints,
                                           test_datapoints, keep_classes=False)

  # Test full model, not keeping base model classes.
  def test_training_l2_norm_model_not_keep_classes(self):
    training_datapoints = [
        {'image_names': ['cat_train_0.bmp'], 'label_id': 0},
        {'image_names': ['dog_train_0.bmp'], 'label_id': 1},
        {'image_names': ['hotdog_train_0.bmp', 'hotdog_train_1.bmp'], 'label_id': 2}
    ]
    test_datapoints = [
        {'image_name': 'cat_test_0.bmp', 'label_id': 0, 'score': 0.99},
        {'image_name': 'dog_test_0.bmp', 'label_id': 1, 'score': 0.99},
        {'image_name': 'hotdog_test_0.bmp', 'label_id': 2, 'score': 0.99}
    ]
    for model_path in self._MODEL_LIST:
      with self.subTest():
        with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
          self._train_and_test(model_path, output_model_path.name, training_datapoints,
                               test_datapoints, keep_classes=False)

  # Test full model, keeping base model classes.
  def test_training_l2_norm_model_keep_classes(self):
    training_datapoints = [
        {'image_names': ['cat_train_0.bmp'], 'label_id': 1001},
        {'image_names': ['dog_train_0.bmp'], 'label_id': 1002},
        {'image_names': ['hotdog_train_0.bmp', 'hotdog_train_1.bmp'], 'label_id': 1003}
    ]
    test_datapoints = [
        {'image_name': 'cat_test_0.bmp', 'label_id': 1001, 'score': 0.99},
        {'image_name': 'dog_test_0.bmp', 'label_id': 1002, 'score': 0.93},
        {'image_name': 'hotdog_test_0.bmp', 'label_id': 1003, 'score': 0.95}
    ]
    for model_path in self._MODEL_LIST:
      with self.subTest():
        with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
          self._train_and_test(model_path,
                               output_model_path.name,
                               training_datapoints,
                               test_datapoints,
                               keep_classes=True)

  def test_incremental_training(self):
    training_datapoints = [
        {'image_names': ['cat_train_0.bmp'], 'label_id': 0},
    ]
    retrain_training_datapoints = [
        {'image_names': ['dog_train_0.bmp'], 'label_id': 1},
        {'image_names': ['hotdog_train_0.bmp', 'hotdog_train_1.bmp'], 'label_id': 2}
    ]
    test_datapoints = [
        {'image_name': 'cat_test_0.bmp', 'label_id': 0, 'score': 0.99},
        {'image_name': 'dog_test_0.bmp', 'label_id': 1, 'score': 0.99},
        {'image_name': 'hotdog_test_0.bmp', 'label_id': 2, 'score': 0.99}
    ]
    for model_path in self._MODEL_LIST:
      with self.subTest():
        with tempfile.NamedTemporaryFile(suffix='.tflite') as cat_only_model_path:
          self._train_and_test(model_path, cat_only_model_path.name,
                               training_datapoints, [], keep_classes=False)
          # Retrain based on cat only model.
          with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
            self._train_and_test(cat_only_model_path.name,
                                 output_model_path.name,
                                 retrain_training_datapoints,
                                 test_datapoints,
                                 keep_classes=True)

  def test_train_all(self):
    for model_path in self._MODEL_LIST:
      with self.subTest():
        with tempfile.NamedTemporaryFile(suffix='.tflite') as output_model_path:
          data_dir = test_utils.test_data_path('imprinting')
          engine = ImprintingEngine(model_path, keep_classes=False)
          image_shape = self._get_image_shape(model_path)

          # train.
          train_set = [
              ['cat_train_0.bmp'],
              ['dog_train_0.bmp'],
              ['hotdog_train_0.bmp', 'hotdog_train_1.bmp']
          ]
          train_input = [(
              test_utils.prepare_images(image_list, data_dir, image_shape)
          ) for image_list in train_set]
          engine.train_all(train_input)
          engine.save_model(output_model_path.name)

          # Test.
          engine = ClassificationEngine(output_model_path.name)
          self.assertEqual(1, engine.get_num_of_output_tensors())
          self.assertEqual(3, engine.get_output_tensor_size(0))

          label_to_id_map = {'cat': 0, 'dog': 1, 'hot_dog': 2}
          self._classify_image(
              engine, data_dir, 'cat_test_0.bmp', label_to_id_map['cat'], 0.99)
          self._classify_image(
              engine, data_dir, 'dog_test_0.bmp', label_to_id_map['dog'], 0.99)
          self._classify_image(
              engine, data_dir, 'hotdog_test_0.bmp', label_to_id_map['hot_dog'],
              0.99)


if __name__ == '__main__':
  unittest.main()
