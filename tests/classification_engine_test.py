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

import unittest
from edgetpu.classification.engine import ClassificationEngine
import numpy as np
from PIL import Image

from . import test_utils

def mobilenet_v1_engine():
  return ClassificationEngine(
      test_utils.test_data_path('mobilenet_v1_1.0_224_quant_edgetpu.tflite'))


class ClassificationEngineTestCase(unittest.TestCase):

  def _test_classify_cat(self, model_name, expected):
    labels = test_utils.read_label_file(test_utils.test_data_path('imagenet_labels.txt'))
    engine = ClassificationEngine(test_utils.test_data_path(model_name))
    with test_utils.test_image('cat.bmp') as img:
      ret = engine.classify_with_image(img, top_k=1)
      self.assertEqual(len(ret), 1)
      # Some models recognize it as egyptian cat while others recognize it as
      # tabby cat.
      self.assertTrue(labels[ret[0][0]] == 'tabby, tabby cat' or
                      labels[ret[0][0]] == 'Egyptian cat')
      ret = engine.classify_with_image(img, top_k=3)
      self.assertEqual(len(expected), len(ret))
      for i in range(len(expected)):
        # Check label.
        self.assertEqual(labels[ret[i][0]], expected[i][0])
        # Check score.
        self.assertGreater(ret[i][1], expected[i][1])


class TestClassificationEnginePythonAPI(ClassificationEngineTestCase):

  def test_random_input(self):
    engine = mobilenet_v1_engine()
    random_input = test_utils.generate_random_input(1, 224 * 224 * 3)
    ret = engine.classify_with_input_tensor(random_input, top_k=1)
    self.assertEqual(len(ret), 1)
    ret = engine.classify_with_input_tensor(random_input, threshold=1.0)
    self.assertEqual(len(ret), 0)

  def test_top_k(self):
    engine = mobilenet_v1_engine()
    random_input = test_utils.generate_random_input(1, 224 * 224 * 3)
    engine.classify_with_input_tensor(random_input, top_k=1)
    # top_k = number of categories
    engine.classify_with_input_tensor(random_input, top_k=1001)
    # top_k > number of categories
    engine.classify_with_input_tensor(random_input, top_k=1234)

  def test_image_object(self):
    engine = mobilenet_v1_engine()
    with test_utils.test_image('cat.bmp') as img:
      ret = engine.classify_with_image(img, threshold=0.4, top_k=10)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0][0], 286)  # Egyptian cat
      self.assertGreater(ret[0][1], 0.79)
      # Try with another resizing method.
      ret = engine.classify_with_image(
          img, threshold=0.4, top_k=10, resample=Image.BICUBIC)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0][0], 286)  # Egyptian cat
      self.assertGreater(ret[0][1], 0.83)

  def test_raw_input(self):
    engine = mobilenet_v1_engine()
    with test_utils.test_image('cat.bmp') as img:
      img = img.resize((224, 224), Image.NEAREST)
      input_tensor = np.asarray(img).flatten()
      ret = engine.classify_with_input_tensor(input_tensor, threshold=0.4, top_k=10)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0][0], 286)  # Egyptian cat
      self.assertGreater(ret[0][1], 0.79)

  def test_get_raw_ouput(self):
    engine = mobilenet_v1_engine()
    with test_utils.test_image('cat.bmp') as img:
      engine.classify_with_image(img, top_k=3)
    raw_output = engine.get_raw_output()
    self.assertGreater(raw_output[282], 0.05)  # tabby, tabby cat
    self.assertGreater(raw_output[283], 0.12)  # tiger cat
    self.assertGreater(raw_output[286], 0.79)  # Egyptian cat

  def test_various_models(self):
    # Mobilenet V1.
    self._test_classify_cat(
        'mobilenet_v1_1.0_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.78), ('tiger cat', 0.128)]
    )
    # Mobilenet V2.
    self._test_classify_cat(
        'mobilenet_v2_1.0_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.84)]
    )
    # Inception V1.
    self._test_classify_cat(
        'inception_v1_224_quant_edgetpu.tflite',
        [('tabby, tabby cat', 0.41),
         ('Egyptian cat', 0.35),
         ('tiger cat', 0.156)]
    )
    # Inception V2.
    self._test_classify_cat(
        'inception_v2_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.85)]
    )
    # Inception V3.
    self._test_classify_cat(
        'inception_v3_299_quant_edgetpu.tflite',
        [('tabby, tabby cat', 0.45),
         ('Egyptian cat', 0.2),
         ('tiger cat', 0.15)]
    )
    # Inception V4.
    self._test_classify_cat(
        'inception_v4_299_quant_edgetpu.tflite',
        [('Egyptian cat', 0.45),
         ('tabby, tabby cat', 0.3),
         ('tiger cat', 0.15)]
    )

if __name__ == '__main__':
  unittest.main()
