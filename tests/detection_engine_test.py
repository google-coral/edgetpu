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
from edgetpu.detection.engine import DetectionEngine
import numpy as np
from PIL import Image

from . import test_utils

def ssd_mobilenet_v1_coco_engine():
  return DetectionEngine(
      test_utils.test_data_path('ssd_mobilenet_v1_coco_quant_postprocess.tflite'))

class TestDetectionEnginePythonAPI(unittest.TestCase):

  def _test_gray_face(self, model_name):
    engine = DetectionEngine(test_utils.test_data_path(model_name))
    with test_utils.test_image('grace_hopper.bmp') as img:
      # Convert image to grayscale.
      img = img.convert('L')
      ret = engine.detect_with_image(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 0)
      self.assertGreater(ret[0].score, 0.95)
      self.assertGreater(
          test_utils.iou(
              np.array([[0.28, 0.07], [0.74, 0.60]]), ret[0].bounding_box), 0.9)

      ret = engine.detect_with_image(img, top_k=1, relative_coord=False)
      # Check coordinates in pixels.
      self.assertGreater(
          test_utils.iou(
              np.array([[144, 41], [382, 365]]), ret[0].bounding_box), 0.9)

  def test_image_object(self):
    engine = ssd_mobilenet_v1_coco_engine()
    with test_utils.test_image('cat.bmp') as img:
      ret = engine.detect_with_image(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.iou(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.86)

      # Detect with different resample algorithm.
      ret = engine.detect_with_image(
          img, top_k=1, resample=Image.BICUBIC)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.iou(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.90)

      # No error when top_k > number limit of detection candidates.
      engine.detect_with_image(img, top_k=100000)

  def test_image_object_without_labels(self):
    engine = ssd_mobilenet_v1_coco_engine()
    with test_utils.test_image('cat.bmp') as img:
      ret = engine.detect_with_image(img, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.iou(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.86)

  def test_raw_input(self):
    engine = ssd_mobilenet_v1_coco_engine()
    with test_utils.test_image('cat.bmp') as img:
      input_tensor = np.asarray(img.resize((300, 300), Image.NEAREST)).flatten()
      ret = engine.detect_with_input_tensor(input_tensor, top_k=1)
      self.assertEqual(len(ret), 1)
      self.assertEqual(ret[0].label_id, 16)  # cat
      self.assertGreater(ret[0].score, 0.79)
      self.assertGreater(
          test_utils.iou(
              np.array([[0.1, 0.1], [0.7, 1.0]]), ret[0].bounding_box), 0.86)

  def test_gray_face_detection(self):
    self._test_gray_face('ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')

if __name__ == '__main__':
  unittest.main()
