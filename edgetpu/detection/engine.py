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

"""An inference engine that performs object detection."""

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils.warning import deprecated
from edgetpu.utils import image_processing
import numpy as np
from PIL import Image


class DetectionCandidate(object):
  """A data structure that represents one detection candidate (label id, score, and bounding box).

     This is returned by methods :func:`~DetectionEngine.detect_with_image` and
     :func:`~DetectionEngine.detect_with_input_tensor`."""

  __slots__ = ['label_id', 'score', 'bounding_box']

  def __init__(self, label_id, score, x1, y1, x2, y2):
    #: An :obj:`int` for the label id.
    self.label_id = label_id
    #: A :obj:`float` for the confidence score.
    self.score = score
    #: A 2-D aray (:obj:`numpy.ndarray`) that describes the bounding box for the detected object.
    #:
    #: The format is [[x1, y1], [x2, y2]], where [x1, y1] is the top-left corner and [x2, y2]
    #: is the bottom-right corner of the bounding box. The values can be either floats (relative
    #: coordinates) or integers (pixel coordinates), depending on the ``relative_coord`` bool you
    #: pass to the :func:`~DetectionEngine.detect_with_image` or
    #: :func:`~DetectionEngine.detect_with_input_tensor` method. [0, 0] is always the top-left corner.
    self.bounding_box = np.array([[x1, y1], [x2, y2]])


class DetectionEngine(BasicEngine):
  """Extends :class:`~edgetpu.basic.basic_engine.BasicEngine` to perform object detection
     with a given model.

     This API assumes the given model is trained for object detection. Now this
     engine only supports SSD model with postprocessing operator.
  """

  def __init__(self, model_path, device_path=None):
    """
    Args:
      model_path (str): Path to a TensorFlow Lite (``.tflite``) file.
        This model must be `compiled for the Edge TPU
        <https://coral.ai/docs/edgetpu/compiler/>`_; otherwise, it simply executes
        on the host CPU.
      device_path (str): The device path for the Edge TPU this engine should use. This argument
        is needed only when you have multiple Edge TPUs and more inference engines than
        available Edge TPUs. For details, read `how to use multiple Edge TPUs
        <https://coral.ai/docs/edgetpu/multiple-edgetpu/>`_.

    Raises:
      ValueError: If the model's output tensor size is not 4.
    """
    if device_path:
      super().__init__(model_path, device_path)
    else:
      super().__init__(model_path)
    output_tensors_sizes = self.get_all_output_tensors_sizes()
    if output_tensors_sizes.size != 4:
      raise ValueError(
          ('Dectection model should have 4 output tensors!'
           'This model has {}.'.format(output_tensors_sizes.size)))
    self._tensor_start_index = [0]
    offset = 0
    for i in range(3):
      offset = offset + output_tensors_sizes[i]
      self._tensor_start_index.append(int(offset))

  def detect_with_image(self, img, threshold=0.1, top_k=3,
                      keep_aspect_ratio=False, relative_coord=True,
                      resample=Image.NEAREST):
    """Performs object detection with an image.

    Args:
      img (:obj:`PIL.Image`): The image you want to process.
      threshold (float): Minimum confidence threshold for detected objects. For example,
        use ``0.5`` to receive only detected objects with a confidence equal-to or higher-than 0.5.
      top_k (int): The maximum number of detected objects to return.
      keep_aspect_ratio (bool): If True, keep the image aspect ratio the same when down-sampling
        the image (by adding black pixel padding so it fits the input tensor's dimensions, via the
        :func:`~edgetpu.utils.image_processing.resampling_with_original_ratio()` function).
        If False, resize and reshape the image (without cropping) to match the input
        tensor's dimensions.
        (Note: This option should be the same as what is applied on input images
        during model training. Otherwise, the accuracy might be affected and the
        bounding box of detection result might be stretched.)
      relative_coord (bool): If True, provide coordinates as float values between 0 and 1,
        representing each position relative to the total image width/height. If False, provide
        coordinates as integers, representing pixel positions in the original image. [0, 0] is
        always the top-left corner.
      resample (int): A resampling filter for image resizing.
        This can be one of :attr:`PIL.Image.NEAREST`, :attr:`PIL.Image.BOX`,
        :attr:`PIL.Image.BILINEAR`, :attr:`PIL.Image.HAMMING`, :attr:`PIL.Image.BICUBIC`,
        or :attr:`PIL.Image.LANCZOS`. Default is :attr:`PIL.Image.NEAREST`. See `Pillow filters
        <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters>`_.
        (Note: A complex filter such as :attr:`PIL.Image.BICUBIC` may create slightly better
        accuracy but it also causes higher latency.)

    Returns:
      A :obj:`list` of detected objects as :class:`DetectionCandidate` objects.

    Raises:
      RuntimeError: If the model's input tensor shape doesn't match the shape expected for an
        object detection model, which is [1, height, width, channel].
      ValueError: If the input tensor channel is not 1 (grayscale) or 3 (RGB)
      ValueError: If argument values are invalid.
    """
    input_tensor_shape = self.get_input_tensor_shape()
    if (input_tensor_shape.size != 4 or
        input_tensor_shape[0] != 1):
      raise RuntimeError(
          'Invalid input tensor shape! Expected: [1, height, width, channel]')
    _, height, width, channel = input_tensor_shape

    if keep_aspect_ratio:
      resized_img, ratio = image_processing.resampling_with_original_ratio(
          img, (width, height), resample)
    else:
      resized_img = img.resize((width, height), resample)

    # Handle color space conversion.
    if channel == 1:
      resized_img = resized_img.convert('L')
    elif channel == 3:
      resized_img = resized_img.convert('RGB')
    else:
      raise ValueError(
          'Invalid input tensor channel! Expected: 1 or 3. Actual: %d' % channel)

    input_tensor = np.asarray(resized_img).flatten()
    candidates = self.detect_with_input_tensor(input_tensor, threshold, top_k)
    for c in candidates:
      if keep_aspect_ratio:
        c.bounding_box = c.bounding_box / ratio
        c.bounding_box[0] = np.maximum([0.0, 0.0], c.bounding_box[0])
        c.bounding_box[1] = np.minimum([1.0, 1.0], c.bounding_box[1])
      if relative_coord is False:
        c.bounding_box = c.bounding_box * [img.size]
    return candidates

  def detect_with_input_tensor(self, input_tensor, threshold=0.1, top_k=3):
    """Performs object detection with a raw input tensor.

    This requires you to process the input data (the image) and convert
    it to the appropriately formatted input tensor for your model.

    Args:
      input_tensor (:obj:`numpy.ndarray`): A 1-D array as the input tensor.
      threshold (float): Minimum confidence threshold for detected objects. For example,
        use ``0.5`` to receive only detected objects with a confidence equal-to or higher-than 0.5.
      top_k (int): The maximum number of detected objects to return.

    Returns:
      A :obj:`list` of detected objects as :class:`DetectionCandidate` objects.

    Raises:
      ValueError: If argument values are invalid.
    """
    if top_k <= 0:
      raise ValueError('top_k must be positive!')
    _, raw_result = self.run_inference(input_tensor)
    result = []
    num_candidates = raw_result[self._tensor_start_index[3]]
    for i in range(int(round(num_candidates))):
      score = raw_result[self._tensor_start_index[2] + i]
      if score > threshold:
        label_id = int(round(raw_result[self._tensor_start_index[1] + i]))
        y1 = max(0.0, raw_result[self._tensor_start_index[0] + 4 * i])
        x1 = max(0.0, raw_result[self._tensor_start_index[0] + 4 * i + 1])
        y2 = min(1.0, raw_result[self._tensor_start_index[0] + 4 * i + 2])
        x2 = min(1.0, raw_result[self._tensor_start_index[0] + 4 * i + 3])
        result.append(DetectionCandidate(label_id, score, x1, y1, x2, y2))
    result.sort(key=lambda x: -x.score)
    return result[:top_k]

  @deprecated
  def DetectWithImage(self, img, threshold=0.1, top_k=3,
                      keep_aspect_ratio=False, relative_coord=True,
                      resample=Image.NEAREST):
    return self.detect_with_image(img, threshold, top_k, keep_aspect_ratio,
                                  relative_coord, resample)

  @deprecated
  def DetectWithInputTensor(self, input_tensor, threshold=0.1, top_k=3):
    return self.detect_with_input_tensor(input_tensor, threshold, top_k)
