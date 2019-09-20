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

"""Tests image classification accuracy with ImageNet validation data set.

Please download the validation image data from to edgetpu/test_data/imagenet/
"""

import unittest
from edgetpu.classification.engine import ClassificationEngine

from PIL import Image

from . import test_utils


class TestImagenetClassification(unittest.TestCase):

  def _crop_image(self, image, cropping_fraction=0.75):
    """Crops an image in the center.
    Args:
      image: PIL image object.
      cropping_fraction: Fraction of cropped image.

    Returns:
      Cropped image as PIL image object.
    """
    init_width, init_height = image.size
    new_width = round(init_width * cropping_fraction)
    new_height = round(init_height * cropping_fraction)
    width_offset = round((init_width - init_width * cropping_fraction) / 2.0)
    height_offset = round((init_height - init_height * cropping_fraction) / 2.0)
    return image.crop((width_offset, height_offset,
                       width_offset + new_width, height_offset + new_height))

  def _test_model(self, model_name, expected_top_1=None, expected_top_5=None):
    engine = ClassificationEngine(test_utils.test_data_path(model_name))
    with open(test_utils.test_data_path('imagenet/val.txt'), 'r') as gt_file:
      gt = [line .strip().split(' ') for line in gt_file.readlines()]

    top_1_count = 0
    top_5_count = 0
    print('Running inference for model %s...' % model_name)
    for i in range(50000):
      label = int(gt[i][1]) + 1
      image_name = 'imagenet/ILSVRC2012_val_%s.JPEG' % str(i + 1).zfill(8)
      with test_utils.test_image(image_name) as image:
        image = self._crop_image(image.convert('RGB'))
        prediction = engine.classify_with_image(image, threshold=0.0, top_k=5)
        if prediction[0][0] == label:
          top_1_count += 1
          top_5_count += 1
        else:
          for j in range(1, len(prediction)):
            if prediction[j][0] == label:
              top_5_count += 1

    top_1_accuracy = top_1_count / 50000.0
    top_5_accuracy = top_5_count / 50000.0
    print('Top 1 accuracy: %.2f%%' % (top_1_accuracy * 100))
    print('Top 5 accuracy: %.2f%%' % (top_5_accuracy * 100))
    if expected_top_1 is not None:
      self.assertAlmostEqual(top_1_accuracy, expected_top_1, places=4)
    if expected_top_5 is not None:
      self.assertAlmostEqual(top_5_accuracy, expected_top_5, places=4)

  def test_mobilenet_v1(self):
    self._test_model('mobilenet_v1_1.0_224_quant_edgetpu.tflite',
                     expected_top_1=0.6854, expected_top_5=0.8772)

  def test_mobilenet_v2(self):
    self._test_model('mobilenet_v2_1.0_224_quant_edgetpu.tflite',
                     expected_top_1=0.6912, expected_top_5=0.8829)


if __name__ == '__main__':
  unittest.main()
