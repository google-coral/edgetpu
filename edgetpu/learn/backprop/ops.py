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

"""Forward and backward pass implmenetation for operators."""
import numpy as np


def fc_forward(mat_x, mat_w, vec_b):
  """Forward pass of Fully-Connected layer.

  A good reference for this is: http://cs231n.github.io/linear-classify/#score

  Args:
    mat_x: NxD ndarray, N is number of features, D is length of feature vector
    mat_w: DxC ndarray, C is number of classes.
    vec_b: length C ndarray.

  Returns:
    a tuple of (mat_out, cached)
    mat_out: NxC ndarray, as defined by Y=X*W+b.
    cached: value stored to help calculating gradient in backward pass.
  """
  mat_out = mat_x.dot(mat_w) + vec_b
  cached = (mat_x, mat_w)
  return mat_out, cached


def fc_backward(dout, cached):
  """Backward pass of Fully-Connected layer.

  FC layer is defined by: Y=X*W+b

  In general, the gradient of a function, which has tensor input and output, is
  a high dimensional tensor. But for linear relation as Y=X*W+b, this high
  dimensional tensor has a lot of zeros and can be simplified.

  A good reference is:
    http://cs231n.stanford.edu/2017/handouts/linear-backprop.pdf

  Args:
    dout: NxC ndarray, gradient with respect to Y
    cached: cached value from fc_forward

  Returns:
    a tuple of gradients with respect to X, W, b
  """
  mat_x, mat_w = cached
  dmat_x = dout.dot(mat_w.T)
  dmat_w = mat_x.T.dot(dout)
  dvec_b = dout.T.dot(np.ones([mat_x.shape[0]]))
  return dmat_x, dmat_w, dvec_b


def softmax_cross_entropy_loss(logits, labels):
  """Calculates softmax cross entropy loss.

  Reference on softmax cross entropy:
    http://cs231n.github.io/linear-classify/#softmax
  Reference on getting gradient of softmax cross entropy:
    https://deepnotes.io/softmax-crossentropy

  Args:
    logits: NxC ndarray, unnormalized logits
    labels: length N ndarray, index of class label in one hot vector.

  Returns:
    A tuple of Cross-entropy loss and gradient with respect to logits
  """
  # Use softmax(x) = softmax(x-C) to avoid exp() overflow.
  logits -= np.max(logits, axis=1, keepdims=True)
  probs = np.exp(logits)
  probs /= np.sum(probs, axis=1, keepdims=True)
  num_input = logits.shape[0]
  loss = -np.sum(np.log(probs[range(num_input), labels])) / num_input

  dlogits = probs.copy()
  dlogits[range(num_input), labels] -= 1
  dlogits /= num_input
  return loss, dlogits
