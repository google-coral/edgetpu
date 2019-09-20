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

import json
import unittest
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from pycocotools import coco
from pycocotools import cocoeval

from . import test_utils

class TestCocoObjectDetection(unittest.TestCase):

  @staticmethod
  def absolute_to_relative_bbox(bbox):
    """Converts the model output bounding box to the format for evaluation.

    The model output bounding box is in format [[x1, y1], [x2, y2]], where
    (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner of the
    bounding box. The COCO evaluation tool expects the bounding box to be in
    format [x1, y1, width, height].

    Args:
      bbox: List, [x1, y1, x2, y2].

    Returns:
      List of [x1, y1, width, height].
    """
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

  def _test_model(self, model_name, expected_ap=None, expected_ar=None,
                 resample=Image.NEAREST):
    engine = DetectionEngine(test_utils.test_data_path(model_name))
    ground_truth_file = 'coco/annotations/instances_val2017.json'
    coco_gt = coco.COCO(test_utils.test_data_path(ground_truth_file))
    detection_results = []
    print('Running inference for model %s...' % model_name)
    for _, img in coco_gt.imgs.items():
      with test_utils.test_image('coco', 'val2017', img['file_name']) as image:
        ret = engine.detect_with_image(image.convert('RGB'), threshold=0, top_k=100,
                                       relative_coord=False, resample=resample)
        for detection in ret:
          detection_results.append({
              'image_id': img['id'],
              # Model label id and ground truth label id are 1 off.
              'category_id': detection.label_id + 1,
              'bbox': self.absolute_to_relative_bbox(
                  detection.bounding_box.flatten().tolist()),
              'score': detection.score.item()})

    detection_file = '/tmp/%s.json' % model_name
    with open(detection_file, 'w') as f:
      json.dump(detection_results, f, separators=(',', ':'))

    coco_dt = coco_gt.loadRes(detection_file)
    coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if expected_ap is not None:
      self.assertGreaterEqual(coco_eval.stats[0], expected_ap)
    if expected_ar is not None:
      self.assertGreaterEqual(coco_eval.stats[6], expected_ar)

  def test_mobilenet_ssd_v1(self):
    self._test_model('mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite',
                     expected_ap=0.173, expected_ar=0.174)

  def test_mobilenet_ssd_v2(self):
    self._test_model('mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
                     expected_ap=0.215, expected_ar=0.199)


if __name__ == '__main__':
  unittest.main()
