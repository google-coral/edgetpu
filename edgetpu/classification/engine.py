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

"""An inference engine that performs image classification."""

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils.warning import deprecated
import numpy
from PIL import Image


class ClassificationEngine(BasicEngine):
  """Extends :class:`~edgetpu.basic.basic_engine.BasicEngine` to perform image classification
     with a given model.

     This API assumes the given model is trained for image classification.
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
      ValueError: If the model's output tensor size is not 1.
    """
    if device_path:
      super().__init__(model_path, device_path)
    else:
      super().__init__(model_path)
    output_tensors_sizes = self.get_all_output_tensors_sizes()
    if output_tensors_sizes.size != 1:
      raise ValueError(
          ('Classification model should have 1 output tensor only!'
           'This model has {}.'.format(output_tensors_sizes.size)))

  def classify_with_image(
      self, img, threshold=0.1, top_k=3, resample=Image.NEAREST):
    """Performs classification with an image.

    Args:
      img (:obj:`PIL.Image`): The image you want to classify.
      threshold (float): Minimum confidence threshold for returned classifications. For example,
        use ``0.5`` to receive only classifications with a confidence equal-to or higher-than 0.5.
      top_k (int): The maximum number of classifications to return.
      resample (int): A resampling filter for image resizing.
        This can be one of :attr:`PIL.Image.NEAREST`, :attr:`PIL.Image.BOX`,
        :attr:`PIL.Image.BILINEAR`, :attr:`PIL.Image.HAMMING`, :attr:`PIL.Image.BICUBIC`,
        or :attr:`PIL.Image.LANCZOS`. Default is :attr:`PIL.Image.NEAREST`. See `Pillow filters
        <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters>`_.
        (Note: A complex filter such as :attr:`PIL.Image.BICUBIC` may create slightly better
        accuracy but it also causes higher latency.)

    Returns:
        A :obj:`list` of classifications, each of which is a list [int, float] that represents
        the label id (int) and the confidence score (float).

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
    img = img.resize((width, height), resample)
    # Handle color space conversion.
    if channel == 1:
      img = img.convert('L')
    elif channel == 3:
      img = img.convert('RGB')
    else:
      raise ValueError(
          'Invalid input tensor channel! Expected: 1 or 3. Actual: %d' % channel)

    input_tensor = numpy.asarray(img).flatten()
    return self.classify_with_input_tensor(input_tensor, threshold, top_k)

  def classify_with_input_tensor(self, input_tensor, threshold=0.0, top_k=3):
    """Performs classification with a raw input tensor.

    This requires you to process the input data (the image) and convert
    it to the appropriately formatted input tensor for your model.

    Args:
      input_tensor (:obj:`numpy.ndarray`): A 1-D array as the input tensor.
      threshold (float): Minimum confidence threshold for returned classifications. For example,
        use ``0.5`` to receive only classifications with a confidence equal-to or higher-than 0.5.
      top_k (int): The maximum number of classifications to return.

    Returns:
        A :obj:`list` of classifications, each of which is a list [int, float] that represents
        the label id (int) and the confidence score (float).

    Raises:
      ValueError: If argument values are invalid.
    """
    if top_k <= 0:
      raise ValueError('top_k must be positive!')
    _, self._raw_result = self.run_inference(
        input_tensor)
    # top_k must be less or equal to number of possible results.
    top_k = min(top_k, len(self._raw_result))
    result = []
    indices = numpy.argpartition(self._raw_result, -top_k)[-top_k:]
    for i in indices:
      if self._raw_result[i] > threshold:
        result.append((i, self._raw_result[i]))
    result.sort(key=lambda tup: -tup[1])
    return result[:top_k]

  @deprecated
  def ClassifyWithImage(
      self, img, threshold=0.1, top_k=3, resample=Image.NEAREST):
      return self.classify_with_image(img, threshold, top_k, resample)

  @deprecated
  def ClassifyWithInputTensor(self, input_tensor, threshold=0.0, top_k=3):
    return self.classify_with_input_tensor(input_tensor, threshold, top_k)

