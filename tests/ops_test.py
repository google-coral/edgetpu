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

from edgetpu.learn.backprop import ops
import numpy as np


def get_numerical_gradient(f, x, h=1e-4):
  """Gets numerical gradient of f, at point x.

  Using df = f(x+h)-f(x-h)/(2*h) for better numerical result.

  Args:
    f: function that takes a matrix/vector and returns a scalar.
    x: ndarray.
    h: how much to wiggle along each dimension.

  Returns:
    dx, ndarray of the same shape as x. Gradient at x.
  """
  # Using central difference to get better numerical result.
  dx = np.zeros(x.shape)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    # f(x+h)
    x[it.multi_index] += h
    fxph = f(x)
    # f(x-h)
    x[it.multi_index] -= 2 * h
    fxmh = f(x)
    x[it.multi_index] += h
    # dx at `it.multi_index`
    dx[it.multi_index] = (fxph - fxmh) / (2 * h)
    it.iternext()

  assert dx.shape == x.shape
  return dx


class OpsTest(unittest.TestCase):

  def test_get_numerical_gradient(self):
    # f=x0+x1+x2+x3+x4
    vec_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dx = get_numerical_gradient(np.sum, vec_x)
    self.assertTrue(np.allclose(np.ones(vec_x.shape), dx))

    # f=x0^2+x1^2+x2^2+x3^2+x4^2
    mat_x = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    dx = get_numerical_gradient(lambda x: np.sum(x**2), mat_x)
    self.assertTrue(np.allclose(2 * mat_x, dx))

  def test_fc_forward(self):
    mat_x = np.array(range(10)).reshape([2, 5])
    mat_w = mat_x.T
    vec_b = np.ones(2)
    mat_y_expected = np.array([[31, 81], [81, 256]])
    mat_y, _ = ops.fc_forward(mat_x, mat_w, vec_b)
    self.assertTrue(np.allclose(mat_y_expected, mat_y))

  def test_softmax_cross_entropy_loss(self):
    logits = np.ones((1, 10))
    labels = np.array([5])
    loss, dlogits = ops.softmax_cross_entropy_loss(logits, labels)
    self.assertTrue(np.allclose(loss, np.log(10)))

    numeric_dlogits = get_numerical_gradient(
        lambda x: ops.softmax_cross_entropy_loss(x, labels)[0], logits)
    self.assertTrue(np.allclose(numeric_dlogits, dlogits))

  def test_fc_backward(self):
    np.random.seed(12345)
    mat_x = np.random.randn(5, 3)
    mat_w = np.random.randn(3, 10)
    vec_b = np.random.randn(10)
    mat_y, cached = ops.fc_forward(mat_x, mat_w, vec_b)
    labels = np.random.randint(10, size=5)
    _, dlogits = ops.softmax_cross_entropy_loss(mat_y, labels)
    dmat_x, dmat_w, dvec_b = ops.fc_backward(dlogits, cached)

    # Chain FC layer and softmax loss together.
    # `i` for internal.
    def chained_loss(i_mat_x, i_mat_w, i_vec_b, i_labels):
      i_mat_y, _ = ops.fc_forward(i_mat_x, i_mat_w, i_vec_b)
      loss, _ = ops.softmax_cross_entropy_loss(i_mat_y, i_labels)
      return loss

    numeric_dmat_x = get_numerical_gradient(
        lambda var_mat_x: chained_loss(var_mat_x, mat_w, vec_b, labels), mat_x)
    self.assertTrue(np.allclose(numeric_dmat_x, dmat_x))

    numeric_dmat_w = get_numerical_gradient(
        lambda var_mat_w: chained_loss(mat_x, var_mat_w, vec_b, labels), mat_w)
    self.assertTrue(np.allclose(numeric_dmat_w, dmat_w))

    numeric_dvec_b = get_numerical_gradient(
        lambda var_vec_b: chained_loss(mat_x, mat_w, var_vec_b, labels), vec_b)
    self.assertTrue(np.allclose(numeric_dvec_b, dvec_b))
