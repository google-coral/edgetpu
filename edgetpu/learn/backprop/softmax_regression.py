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
"""A softmax regression model for on-device backpropagation of the last layer."""
import os
import pickle

from edgetpu.learn.backprop import ops
from edgetpu.learn.utils import AppendFullyConnectedAndSoftmaxLayerToModel
import numpy as np

# Default names for weights and label map checkpoint.
_WEIGHTS_NAME = 'weights.pickle'
_LABEL_MAP_NAME = 'label_map.pickle'


class SoftmaxRegression(object):
  """An implementation of the softmax regression function (multinominal logistic regression) that
  operates as the last layer of your classification model, and allows for on-device training with
  backpropagation (for this layer only).

  The input for this layer must be an image embedding, which should be the output of your
  embedding extractor (the backbone of your model). Once given here, the input is fed to a
  fully-connected layer where weights and bias are applied, and then passed to the softmax function
  to receive the final probability distribution based on the number of classes for your model:

  training/inference input (image embedding) --> fully-connected layer --> softmax function

  When you're conducting training with :func:`train_with_sgd`, the process uses a cross-entropy loss
  function to measure the error and then update the weights of the fully-connected layer
  (backpropagation).

  When you're satisfied with the inference accuracy, call :func:`save_as_tflite_model` to create a
  new ``.tflite`` model with this retrained layer appended to your embedding extractor. You can then
  run inferences with this new ``.tflite`` file as usual (using
  :class:`~edgetpu.classification.engine.ClassificationEngine`).

  .. note::

    This last layer (FC + softmax) in the retrained model always runs on the host CPU instead of
    the Edge TPU. As long as the rest of your embedding extractor model is compiled for the Edge
    TPU, then running this last layer on the CPU should not significantly affect the inference
    speed.

  For more detail, see the
  `Stanford CS231 explanation of the softmax classifier
  <http://cs231n.github.io/linear-classify/#softmax>`_.
  """

  def __init__(self,
               feature_dim=None,
               num_classes=None,
               weight_scale=0.01,
               reg=0.0):
    """
    Args:
      feature_dim (int): The dimension of the input feature (length of the feature vector).
      num_classes (int): The number of output classes.
      weight_scale (float): A weight factor for computing new weights. The backpropagated
        weights are drawn from standard normal distribution, then multipled by this number to keep
        the scale small.
      reg (float): The regularization strength.
    """
    self.reg = reg
    self.feature_dim = feature_dim
    self.num_classes = num_classes
    self.label_map = None
    self.params = {}
    if feature_dim and num_classes:
      self.params['mat_w'] = weight_scale * np.random.randn(
          feature_dim, num_classes).astype(np.float32)
      self.params['vec_b'] = np.zeros((num_classes,)).astype(np.float32)

    # Needed to set proper quantization parameter for output tensor of FC layer.
    self.min_score = np.finfo(np.float32).max
    self.max_score = np.finfo(np.float32).min

  def _get_loss(self, mat_x, labels):
    """Calculates the loss of the current model for the given data, using a
    cross-entropy loss function.

    Args:
      mat_x (:obj:`numpy.ndarray`): The input data (image embeddings) to test, as a matrix of shape
        ``NxD``, where ``N`` is number of inputs to test and ``D`` is the dimension of the
        input feature (length of the feature vector).
      labels (:obj:`numpy.ndarray`): An array of the correct label indices that correspond to the
        test data passed in ``mat_x`` (class label index in one-hot vector). For example, if
        ``mat_x`` is just one image embedding, this array has one number for that image's correct
        label index.

    Returns:
      A 2-tuple with the cross-entropy loss (float) and gradients (a dictionary with ``'mat_w'``
      and ``'vec_b'``, for weight and bias, respectively).
    """
    mat_w = self.params['mat_w']
    vec_b = self.params['vec_b']
    scores, fc_cached = ops.fc_forward(mat_x, mat_w, vec_b)
    # Record min, max value of scores.
    self.min_score = np.min([self.min_score, np.min(scores)])
    self.max_score = np.max([self.max_score, np.max(scores)])
    loss, dscores = ops.softmax_cross_entropy_loss(scores, labels)
    loss += 0.5 * self.reg * np.sum(mat_w * mat_w)

    grads = {}
    _, grads['mat_w'], grads['vec_b'] = ops.fc_backward(dscores, fc_cached)
    grads['mat_w'] += self.reg * mat_w

    return loss, grads

  def run_inference(self, mat_x):
    """Runs an inference using the current weights.

    Args:
      mat_x (:obj:`numpy.ndarray`): The input data (image embeddings) to infer, as a matrix of shape
        ``NxD``, where ``N`` is number of inputs to infer and ``D`` is the dimension of the
        input feature (length of the feature vector). (This can be one or more image embeddings.)

    Returns:
      The inferred label index (or an array of indices if multiple embeddings given).
    """
    mat_w = self.params['mat_w']
    vec_b = self.params['vec_b']
    scores, _ = ops.fc_forward(mat_x, mat_w, vec_b)
    if len(scores.shape) == 1:
      return np.argmax(scores)
    else:
      return np.argmax(scores, axis=1)

  def save_as_tflite_model(self, in_model_path, out_model_path):
    """Appends learned weights to your TensorFlow Lite model and saves it as a copy.

    Beware that learned weights and biases are quantized from float32 to uint8.

    Args:
      in_model_path (str): Path to the embedding extractor model (``.tflite`` file).
      out_model_path (str): Path where you'd like to save the new model with learned weights
        and a softmax layer appended (``.tflite`` file).
    """
    # Note: this function assumes flattened weights, whose dimension is
    # num_classes x feature_dim. That's why the transpose is needed.
    AppendFullyConnectedAndSoftmaxLayerToModel(
        in_model_path, out_model_path,
        self.params['mat_w'].transpose().flatten(),
        self.params['vec_b'].flatten(), float(self.min_score),
        float(self.max_score))

  def get_accuracy(self, mat_x, labels):
    """Calculates the model's accuracy (percentage correct) when performing inferences on the
    given data and labels.

    Args:
      mat_x (:obj:`numpy.ndarray`): The input data (image embeddings) to test, as a matrix of shape
        ``NxD``, where ``N`` is number of inputs to test and ``D`` is the dimension of the
        input feature (length of the feature vector).
      labels (:obj:`numpy.ndarray`): An array of the correct label indices that correspond to the
        test data passed in ``mat_x`` (class label index in one-hot vector).

    Returns:
      The accuracy (the percent correct) as a float.
    """
    return np.mean(self.run_inference(mat_x) == labels)

  def train_with_sgd(self,
                     data,
                     num_iter,
                     learning_rate,
                     batch_size=100,
                     print_every=100):
    """Trains your model using stochastic gradient descent (SGD).

    The training data must be structured in a dictionary as specified in the ``data`` argument
    below. Notably, the training/validation images must be passed as image embeddings, not as the
    original image input. That is, run the images through your embedding extractor
    (the backbone of your graph) and use the resulting image embeddings here.

    Args:
      data (dict): A dictionary that maps ``'data_train'`` to an array of training image embeddings,
        ``'labels_train'`` to an array of training labels, ``'data_val'`` to an array of validation
        image embeddings, and ``'labels_val'`` to an array of validation labels.
      num_iter (int): The number of iterations to train.
      learning_rate (float): The learning rate (step size) to use in training.
      batch_size (int): The number of training examples to use in each iteration.
      print_every (int): The number of iterations for which to print the loss, and
        training/validation accuracy. For example, ``20`` prints the stats for every 20 iterations.
        ``0`` disables printing.
    """
    data_train = data['data_train']
    labels_train = data['labels_train']
    data_val = data['data_val']
    labels_val = data['labels_val']
    mat_w = self.params['mat_w']
    vec_b = self.params['vec_b']
    num_train = data_train.shape[0]
    loss_history = []
    for i in range(num_iter):
      batch_mask = np.random.choice(num_train, batch_size)
      data_batch = data_train[batch_mask]
      labels_batch = labels_train[batch_mask]
      loss, grads = self._get_loss(data_batch, labels_batch)
      # Simple SGD update rule.
      mat_w -= learning_rate * grads['mat_w']
      vec_b -= learning_rate * grads['vec_b']
      loss_history.append(loss)

      if (print_every > 0) and (i % print_every == 0):
        print('Loss %.2f, train acc %.2f%%, val acc %.2f%%' %
              (loss, 100 * self.get_accuracy(data_train, labels_train),
               100 * self.get_accuracy(data_val, labels_val)))

  def _set_label_map(self, label_map):
    """Attaches label_map with the model."""
    self.label_map = label_map

  def _get_label_map(self):
    """Gets label_map with the model."""
    return self.label_map

  def _save_ckpt(self, ckpt_dir):
    """Saves checkpoint."""
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    # Save weights
    weights_path = os.path.join(ckpt_dir, _WEIGHTS_NAME)
    with open(weights_path, 'wb') as fp:
      pickle.dump(self.params, fp)
    # Save label map
    if self.label_map:
      label_map_path = os.path.join(ckpt_dir, _LABEL_MAP_NAME)
      with open(label_map_path, 'wb') as fp:
        pickle.dump(self.label_map, fp)

  def _load_ckpt(self, ckpt_dir):
    """Loads weights and label_map from file."""
    weights_path = os.path.join(ckpt_dir, _WEIGHTS_NAME)
    with open(weights_path, 'rb') as fp:
      self.params = pickle.load(fp)
    self.feature_dim = self.params['mat_w'].shape[0]
    self.num_classes = self.params['mat_w'].shape[1]

    label_map_path = os.path.join(ckpt_dir, _LABEL_MAP_NAME)
    with open(label_map_path, 'rb') as fp:
      self.label_map = pickle.load(fp)
