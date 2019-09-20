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

"""Utilities to process your images before performing an inference."""

from edgetpu.utils.warning import deprecated
from PIL import ImageOps


def resampling_with_original_ratio(img, required_size, sample):
  """Resizes the image to maintain the original aspect ratio by adding pixel padding where needed.

  For example, if your model's input tensor requires a square image but your image is landscape (and
  you don't want to reshape the image to fit), pass this function your image and the required square
  dimensions, and it returns a square version by adding the necessary amount of black pixels on the
  bottom-side only. If the original image is portrait, it adds black pixels on the right-side
  only.

  Args:
    img (:obj:`PIL.Image`): The image to resize.
    required_size (list): The pixel width and height [x, y] that your model requires for input.
    sample (int): A resampling filter for image resizing.
      This can be one of :attr:`PIL.Image.NEAREST` (recommended), :attr:`PIL.Image.BOX`,
      :attr:`PIL.Image.BILINEAR`, :attr:`PIL.Image.HAMMING`, :attr:`PIL.Image.BICUBIC`,
      or :attr:`PIL.Image.LANCZOS`. See `Pillow filters
      <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters>`_.

  Returns:
    A 2-tuple with a :obj:`PIL.Image` object for the resized image, and a tuple of floats
    representing the aspect ratio difference between the original image and the returned image
    (x delta-ratio, y delta-ratio).
  """
  old_size = img.size
  # Resizing image with original ratio.
  resampling_ratio = min(
      required_size[0] / old_size[0],
      required_size[1] / old_size[1]
  )
  new_size = (
      int(old_size[0] * resampling_ratio),
      int(old_size[1] * resampling_ratio)
  )
  new_img = img.resize(new_size, sample)
  # Expand it to required size.
  delta_w = required_size[0] - new_size[0]
  delta_h = required_size[1] - new_size[1]
  padding = (0, 0, delta_w, delta_h)
  ratio = (new_size[0] / required_size[0], new_size[1] / required_size[1])
  return (ImageOps.expand(new_img, padding), ratio)

@deprecated
def ResamplingWithOriginalRatio(img, required_size, sample):
  resampling_with_original_ratio(img, required_size, sample)
