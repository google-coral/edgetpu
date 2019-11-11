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
r"""A demo for object detection.

Example (Running under edgetpu repo's root directory):

  - Face detection:
    python3 examples/object_detection.py \
    --model='test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite' \
    --input='test_data/face.jpg' \
    --keep_aspect_ratio

  - Pet detection:
    python3 examples/object_detection.py \
    --model='test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite' \
    --label='test_data/pet_labels.txt' \
    --input='test_data/pets.jpg' \
    --keep_aspect_ratio

'--output' is an optional flag to specify file name of output image.
At this moment we only support SSD model with postprocessing operator. Other
models such as YOLO won't work.
"""

import argparse
import platform
import subprocess
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model',
      help='Path of the detection model, it must be a SSD model with postprocessing operator.',
      required=True)
  parser.add_argument('--label', help='Path of the labels file.')
  parser.add_argument(
      '--input', help='File path of the input image.', required=True)
  parser.add_argument('--output', help='File path of the output image.')
  parser.add_argument(
      '--keep_aspect_ratio',
      dest='keep_aspect_ratio',
      action='store_true',
      help=(
          'keep the image aspect ratio when down-sampling the image by adding '
          'black pixel padding (zeros) on bottom or right. '
          'By default the image is resized and reshaped without cropping. This '
          'option should be the same as what is applied on input images during '
          'model training. Otherwise the accuracy may be affected and the '
          'bounding box of detection result may be stretched.'))
  parser.set_defaults(keep_aspect_ratio=False)
  args = parser.parse_args()

  if not args.output:
    output_name = 'object_detection_result.jpg'
  else:
    output_name = args.output

  # Initialize engine.
  engine = DetectionEngine(args.model)
  labels = dataset_utils.read_label_file(args.label) if args.label else None

  # Open image.
  img = Image.open(args.input)
  draw = ImageDraw.Draw(img)

  # Run inference.
  ans = engine.detect_with_image(
      img,
      threshold=0.05,
      keep_aspect_ratio=args.keep_aspect_ratio,
      relative_coord=False,
      top_k=10)

  # Save result.
  if ans:
    for obj in ans:
      print('-----------------------------------------')
      if labels:
        print(labels[obj.label_id])
      print('score = ', obj.score)
      box = obj.bounding_box.flatten().tolist()
      print('box = ', box)
      # Draw a rectangle.
      draw.rectangle(box, outline='red')
    img.save(output_name)
    print('Please check ', output_name)
  else:
    print('No object detected!')


if __name__ == '__main__':
  main()
