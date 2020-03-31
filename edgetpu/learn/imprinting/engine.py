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

"""A weight imprinting engine that performs low-shot transfer-learning for image classification
models.

For more information about how to use this API and how to create the type of model required, see
`Retrain a classification model on-device with weight imprinting
<https://coral.ai/docs/edgetpu/retrain-classification-ondevice/>`_.

.. note::

  We updated :class:`ImprintingEngine` in the July 2019 library update (version
  2.11.1), which requires code changes if you used the previous version. The API changes are
  as follows:

  +   Most importantly, the input model has new architecture requirements. For details,
      read `Retrain a classification model on-device with weight imprinting
      <https://coral.ai/docs/edgetpu/retrain-classification-ondevice/>`_.
  +   The initialization function accepts a new ``keep_classes`` boolean to indicate whether you
      want to keep the pre-trained classes from the provided model.
  +   :func:`~ImprintingEngine.train` now requires a second argument for the class ID you want to
      train, thus allowing you to retrain classes with additional data. (It no longer returns the
      class ID.)
  +   :func:`~ImprintingEngine.train_all` requires a different format for the input data. It now uses
      a list in which each index corresponds to a class ID, and each list entry is an array of
      training images for that class. (It no longer returns a mapping of label IDs.)
  +   New methods :func:`~ImprintingEngine.classify_with_resized_image` and
      :func:`~ImprintingEngine.classify_with_input_tensor` allow you to immediately perform inferences,
      though you can still choose to save the trained model as a ``.tflite`` file with
      :func:`~ImprintingEngine.save_model`.
"""

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.swig.edgetpu_cpp_wrapper import ImprintingEnginePythonWrapper
from edgetpu.utils.warning import deprecated
import numpy
from PIL import Image

class ImprintingEngine(object):
  """Performs weight imprinting (transfer learning) with the given model."""

  def __init__(self, model_path, keep_classes=False):
    """
    Args:
      model_path (str): Path to the model you want to retrain. This model must be a ``.tflite``
        file output by the ``join_tflite_models`` tool. For more information about how to create a
        compatible model, read `Retrain an image classification model on-device
        <https://coral.ai/docs/edgetpu/retrain-classification-ondevice/>`_.
      keep_classes (bool): If True, keep the existing classes from the pre-trained model (and use
        training to add additional classes). If False, drop the existing classes and train the model
        to include new classes only.
    """
    self._engine = ImprintingEnginePythonWrapper.CreateFromFile(
        model_path, keep_classes)
    self._num_classes = 0
    if keep_classes:
      tmp = BasicEngine(model_path)
      assert tmp.get_num_of_output_tensors() == 1
      self._num_classes = tmp.total_output_array_size()

  def save_model(self, output_path):
    """Saves the newly trained model as a ``.tflite`` file.

    You can then use the saved model to perform inferencing with using
    :class:`~edgetpu.classification.engine.ClassificationEngine`. Alternatively, you can immediately
    perform inferences with the retrained model using the local inferencing methods,
    :func:`~ImprintingEngine.classify_with_resized_image` or
    :func:`~ImprintingEngine.classify_with_input_tensor`.

    Args:
      output_path (str): The path and filename where you'd like to save the trained model
        (must end with ``.tflite``).
    """
    self._engine.SaveModel(output_path)

  def train(self, input, class_id):
    """Trains the model with a set of images for one class.

    You can use this to add new classes to the model or retrain classes that you previously
    added using this imprinting API.

    Args:
      input (list of :obj:`numpy.array`): The images to use for training in a single class. Each
        :obj:`numpy.array` in the list represents an image as a 1-D tensor. You can convert each
        image to this format by passing it as an :class:`PIL.Image` to :func:`numpy.asarray()`. The
        maximum number of images allowed in the list is 200.
      class_id (int): The label id for this class. The index must be either
        the number of existing classes (to add a new class to the model) or the index of an existing
        class that was trained using this imprinting API (you can't retrain classes from the
        pre-trained model).
    """
    self._engine.Train(input, class_id)

  def train_all(self, input_data):
    """Trains the model with multiple sets of images for multiple classes.

    This essentially calls :func:`train` for each class of images you provide. You can use this to
    add a batch of new classes or retrain existing classes. Just beware that if you've already added
    new classes using the imprinting API, then the data input here must include the same classes in
    the same order. Alternatively, you can use :func:`train` to retrain specific classes one at a
    time.

    Args:
      input_data (list of :obj:`numpy.array`): The images to train for multiple classes.
        Each :obj:`numpy.array` in the list represents a different class, which
        itself contains a list of :obj:`numpy.array` objects, which each represent an image as a 1-D
        tensor. You can convert each image to this format by passing it as a :class:`PIL.Image` to
        :func:`numpy.asarray()`.
    """
    for class_id, tensors in enumerate(input_data):
      self.train(tensors, class_id=self._num_classes + class_id)

  def classify_with_resized_image(self, img, threshold=0.1, top_k=3):
    """Performs classification with the retrained model using the given image.

    **Note:** The given image must already be resized to match the model's input tensor size.

    Args:
      img (:obj:`PIL.Image`): The image you want to classify.
      threshold (float): Minimum confidence threshold for returned classifications. For example,
        use ``0.5`` to receive only classifications with a confidence equal-to or higher-than 0.5.
      top_k (int): The maximum number of classifications to return.

    Returns:
        A :obj:`list` of classifications, each of which is a list [int, float] that represents
        the label id (int) and the confidence score (float).

    Raises:
      ValueError: If argument values are invalid.
    """
    input_tensor = numpy.asarray(img).flatten()
    return self.classify_with_input_tensor(input_tensor, threshold, top_k)

  def classify_with_input_tensor(self, input_tensor, threshold=0.0, top_k=3):
    """Performs classification with the retrained model using the given raw input tensor.

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
    raw_result = self._engine.RunInference(input_tensor)
    # top_k must be less or equal to number of possible results.
    top_k = min(top_k, len(raw_result))
    result = []
    indices = numpy.argpartition(raw_result, -top_k)[-top_k:]
    for i in indices:
      if raw_result[i] > threshold:
        result.append((i, raw_result[i]))
    result.sort(key=lambda tup: -tup[1])
    return result[:top_k]

  @deprecated
  def SaveModel(self, output_path):
    self.save_model(output_path)

  @deprecated
  def Train(self, input, class_id):
    self.train(input, class_id)

  @deprecated
  def TrainAll(self, input_data):
    self.train_all(input_data)

  @deprecated
  def ClassifyWithResizedImage(self, img, threshold=0.1, top_k=3):
    return self.classify_with_resized_image(img, threshold, top_k)

  @deprecated
  def ClassifyWithInputTensor(self, input_tensor, threhsold=0.0, top_k=3):
    return self.classify_with_input_tensor(input_tensor, threshold, top_k)
