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

"""Benchmark on small data set."""

import numpy as np
import os
import time
import tempfile
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
import test_utils


def input_tensor_size(model):
  """Returns model input tensor size."""
  engine = BasicEngine(test_utils.test_data_path(model))
  batch, height, width, depth = engine.get_input_tensor_shape()
  return batch * height * width * depth


def run_benchmark(model):
  """Measures training time for given model with random data.

  Args:
    model: string, file name of the input model.

  Returns:
    float, training time.
  """
  input_size = input_tensor_size(model)
  engine = ImprintingEngine(test_utils.test_data_path(model), keep_classes=False)

  np.random.seed(12345)
  data_by_category = {}
  # 10 Categories, each has 20 images.
  for i in range(0, 10):
    data_by_category[i] = []
    for j in range(0, 20):
      data_by_category[i].append(np.random.randint(0, 255, input_size))

  start = time.perf_counter()
  for class_id, tensors in enumerate(data_by_category.values()):
    engine.train(tensors, class_id)
  with tempfile.NamedTemporaryFile() as f:
    engine.save_model(f.name)
  training_time = time.perf_counter() - start

  print('Model: %s' % model)
  print('Training time: %.2fs' % training_time)
  return training_time


if __name__ == '__main__':
  args = test_utils.parse_args()
  machine = test_utils.machine_info()
  models, reference = test_utils.read_reference('imprinting_reference_%s.csv' % machine)
  results = [('MODEL', 'DATA_SET', 'INFERENCE_TIME')]
  for i, name in enumerate(models, start=1):
    print('---------------- %d / %d ----------------' % (i, len(models)))
    results.append((name, 'random', run_benchmark(name)))
  test_utils.save_as_csv('imprinting_benchmarks_%s_%s.csv' %
                             (machine, time.strftime('%Y%m%d-%H%M%S')),
                         results)
  test_utils.check_result(reference, results, args.enable_assertion)
