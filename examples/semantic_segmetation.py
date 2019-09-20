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
r"""A demo for semantic segmentation.

For Raspberry Pi, you need to install 'feh' as image viewer:
sudo apt-get install feh

Example (Running under edgetpu repo's root directory):

  python3 examples/semantic_segmentation.py \
  --model='test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite' \
  --input='test_data/bird.bmp' \
  --keep_aspect_ratio

'--output' is an optional flag to specify file name of output image.
"""

import argparse
import platform
import subprocess
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils import dataset_utils, image_processing
from PIL import Image
from PIL import ImageDraw
import numpy as np


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model',
      help='Path of the segmentation model.',
      required=True)
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
    output_name = 'semantic_segmentation_result.jpg'
  else:
    output_name = args.output

  # Initialize engine.
  engine = BasicEngine(args.model)
  _, height, width, _ = engine.get_input_tensor_shape()

  # Open image.
  img = Image.open(args.input)
  if args.keep_aspect_ratio:
    resized_img, ratio = image_processing.resampling_with_original_ratio(
        img, (width, height), Image.NEAREST)
  else:
    resized_img = img.resize((width, height))
    ratio = (1., 1.)

  input_tensor = np.asarray(resized_img).flatten()
  _, raw_result = engine.run_inference(input_tensor)
  result = np.reshape(raw_result, (height, width))
  new_width, new_height = int(width * ratio[0]), int(height * ratio[1])

  # If keep_aspect_ratio, we need to remove the padding area.
  result = result[:new_height, :new_width]
  vis_result = label_to_color_image(result.astype(int)).astype(np.uint8)
  vis_result = Image.fromarray(vis_result)

  vis_img = resized_img.crop((0, 0, new_width, new_height))

  # Concat resized input image and processed segmentation results.
  concated_image = Image.new('RGB', (new_width*2, new_height))
  concated_image.paste(vis_img, (0, 0))
  concated_image.paste(vis_result, (width, 0))

  concated_image.save(output_name)
  # Display result.
  if platform.machine() == 'x86_64':
    # For gLinux, simply show the image.
    concated_image.show()
  elif platform.machine() == 'armv7l':
    # For Raspberry Pi, you need to install 'feh' to display image.
    subprocess.Popen(['feh', output_name])
  else:
    print('Please check ', output_name)


if __name__ == '__main__':
  main()
