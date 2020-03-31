# Copyright 2020 Google LLC
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

"""Benchmark backprop on small fake data set."""

from edgetpu.learn.backprop.softmax_regression import SoftmaxRegression

import numpy as np
import test_utils
import time


def _benchmark_for_training(num_classes, feature_dim):
  """Measures training time for given data set parameters.

  Args:
    num_classes: int, number of classes.
    feature_dim: int, dimension of the feature vector.

  Returns:
    float, training time.
  """

  num_train = 1024
  num_val = 256
  num_total = num_train + num_val
  class_sizes = (num_total // num_classes) * np.ones(num_classes, dtype=int)

  print('Preparing data set for num_classes=%d, feature_dim=%d' % (
      num_classes, feature_dim))
  np.random.seed(12345)
  all_data = np.random.rand(num_total, feature_dim)
  all_labels = np.tile(np.arange(num_classes), class_sizes[0])
  np.random.shuffle(all_labels)

  dataset = {}
  dataset['data_train'] = all_data[0:num_train]
  dataset['labels_train'] = all_labels[0:num_train]
  dataset['data_val'] = all_data[num_train:]
  dataset['labels_val'] = all_labels[num_train:]

  model = SoftmaxRegression(feature_dim, num_classes)

  # Train with SGD.
  num_iter = 500
  learning_rate = 0.01
  batch_size = 100
  print('Start backprop')
  start_time = time.perf_counter()
  model.train_with_sgd(
      dataset, num_iter, learning_rate, batch_size, print_every=-1)
  training_time = time.perf_counter() - start_time
  print('Backprop time: ', training_time, 's')
  return training_time

if __name__ == '__main__':
  args = test_utils.parse_args()
  machine = test_utils.machine_info()
  # cases are defined by parameter pairs [num_classes, feature_dim].
  cases = [[4, 256], [16, 256], [4, 1024], [16, 1024]]
  results = [('CASE', 'TRAINING_TIME(s)')]
  for params in cases:
    num_classes = params[0]
    feature_dim = params[1]
    print('-------- num_classes=%d / feature_dim=%d --------' % (
        num_classes, feature_dim))
    results.append((":".join(str(i) for i in params),
        _benchmark_for_training(num_classes, feature_dim)))
  test_utils.save_as_csv('softmax_regression_benchmarks_%s_%s.csv' %
                         (machine, time.strftime('%Y%m%d-%H%M%S')), results)
