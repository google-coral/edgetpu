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
"""Tests SoftmaxRegression class.

Generates some fake data and tries to overfit the data with SoftmaxRegression.
"""
import unittest

from edgetpu.learn.backprop.softmax_regression import SoftmaxRegression
import numpy as np


def generate_fake_data(class_sizes, means, cov_mats):
  """Generates fake data for training and testing.

  Examples from same class is drawn from the same MultiVariate Normal (MVN)
  distribution.

   # classes = len(class_sizes) = len(means) = len(cov_mats)
   dim of MVN = cov_mats[0].shape[0]

  Args:
    class_sizes: list of ints, number of examples to draw from each class.
    means: list of list of floats, mean value of each MVN distribution.
    cov_mats: list of ndarray, each element is a k by k ndarray, which
      represents the covariance matrix in MVN distribution, k is the dimension
      of MVN distribution.

  Returns:
    a tuple of data and labels. data and labels are shuffled.
  """
  # Some sanity checks.
  assert len(class_sizes) == len(means)
  assert len(class_sizes) == len(cov_mats)

  num_data = np.sum(class_sizes)
  feature_dim = len(means[0])
  data = np.empty((num_data, feature_dim))
  labels = np.empty((num_data), dtype=int)

  start_idx = 0
  class_idx = 0
  for size, mean, cov_mat in zip(class_sizes, means, cov_mats):
    data[start_idx:start_idx + size] = np.random.multivariate_normal(
        mean, cov_mat, size)
    labels[start_idx:start_idx + size] = np.ones(size, dtype=int) * class_idx
    start_idx += size
    class_idx += 1

  perm = np.random.permutation(data.shape[0])
  data = data[perm, :]
  labels = labels[perm]

  return data, labels


class SoftmaxRegressionTest(unittest.TestCase):

  def test_softmax_regression_linear_separable_data(self):
    # Fake data is generated from 3 MVN distributions, these MVN distributionss
    # are tuned to be well-separated, such that it can be separated by
    # SoftmaxRegression model (which is a linear classifier).
    num_train = 200
    num_val = 30
    # Let's distribute data evenly among different classes.
    num_classes = 3
    class_sizes = ((num_train + num_val) // num_classes) * np.ones(
        num_classes, dtype=int)
    class_sizes[-1] = (num_train + num_val) - np.sum(class_sizes[0:-1])

    # 3 is chosen, such that each pair of mean is over 6 `sigma` distance
    # apart. Which makes classes harder to `touch` each other.
    # https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    means = np.array([[1, 1], [-1, -1], [1, -1]]) * 3
    feature_dim = len(means[0])
    cov_mats = [np.eye(feature_dim)] * num_classes

    model = SoftmaxRegression(feature_dim, num_classes)
    np.random.seed(12345)
    all_data, all_labels = generate_fake_data(class_sizes, means, cov_mats)

    dataset = {}
    dataset['data_train'] = all_data[0:num_train]
    dataset['labels_train'] = all_labels[0:num_train]
    dataset['data_val'] = all_data[num_train:]
    dataset['labels_val'] = all_labels[num_train:]
    # train with SGD.
    num_iter = 20
    learning_rate = 0.01
    model.train_with_sgd(
        dataset, num_iter, learning_rate, batch_size=100, print_every=5)
    self.assertGreater(
        model.get_accuracy(dataset['data_train'], dataset['labels_train']),
        0.99)

  def test_softmax_regression_linear_non_separable_data(self):
    # Fake data is generated from 3 MVN distributions, these MVN distributions
    # are NOT well-separated.
    num_train = 200
    num_val = 30
    # Let's distribute data evenly among different classes.
    num_classes = 3
    class_sizes = ((num_train + num_val) // num_classes) * np.ones(
        num_classes, dtype=int)
    class_sizes[-1] = (num_train + num_val) - np.sum(class_sizes[0:-1])

    means = np.array([[1, 1], [-1, -1], [1, -1]])
    feature_dim = len(means[0])
    cov_mats = [np.eye(feature_dim)] * num_classes

    model = SoftmaxRegression(feature_dim, num_classes)
    np.random.seed(54321)
    all_data, all_labels = generate_fake_data(class_sizes, means, cov_mats)

    dataset = {}
    dataset['data_train'] = all_data[0:num_train]
    dataset['labels_train'] = all_labels[0:num_train]
    dataset['data_val'] = all_data[num_train:]
    dataset['labels_val'] = all_labels[num_train:]
    # train with SGD.
    num_iter = 50
    learning_rate = 0.1
    model.train_with_sgd(
        dataset, num_iter, learning_rate, batch_size=100, print_every=5)
    self.assertGreater(
        model.get_accuracy(dataset['data_train'], dataset['labels_train']), 0.8)
