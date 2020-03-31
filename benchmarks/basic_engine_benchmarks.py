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

"""Benchmark of models.

Benchmark are measured with CPU 'performance' mode. To enable it, you need to
install 'cpupower' and run:
sudo cpupower frequency-set --governor performance

The reference number is measured on:
  - 'x86_64': Intel Xeon E5-1650 v3(3.50GHz) + Edge TPU accelarator + USB 3.0
  - 'rp3b': Raspberry Pi 3 B (version1.2)+ Edge TPU accelarator + USB 2.0
  - 'rp3b+': Raspberry Pi 3 B+ (version1.3)+ Edge TPU accelarator + USB 2.0
  - 'aarch64': Edge TPU dev board.
"""

import time
import timeit

from edgetpu.basic.basic_engine import BasicEngine
import numpy as np
import test_utils


def run_benchmark(model):
  """Returns average inference time in ms on specified model on random input."""

  print('Benchmark for [%s]' % model)
  print('model path = %s' % test_utils.test_data_path(model))
  engine = BasicEngine(test_utils.test_data_path(model))
  print('input tensor shape = %s' % engine.get_input_tensor_shape())

  iterations = 200 if 'edgetpu' in model else 20
  input_size = engine.required_input_array_size()
  random_input = test_utils.generate_random_input(1, input_size)
  input_data = np.array(random_input, dtype=np.uint8)
  result = 1000 * timeit.timeit(
      lambda: engine.run_inference(input_data),
      number=iterations) / iterations

  print('%.2f ms (iterations = %d)' % (result, iterations))
  return result


if __name__ == '__main__':
  args = test_utils.parse_args()
  machine = test_utils.machine_info()
  test_utils.check_cpu_scaling_governor_status()
  models, reference = test_utils.read_reference(
      'basic_engine_reference_%s.csv' % machine)
  results = [('MODEL', 'INFERENCE_TIME')]
  for i, model in enumerate(models, start=1):
    print('-------------- Model %d / %d ---------------' % (i, len(models)))
    results.append((model, run_benchmark(model)))
  test_utils.save_as_csv('basic_engine_benchmarks_%s_%s.csv' %
                             (machine, time.strftime('%Y%m%d-%H%M%S')),
                         results)
  test_utils.check_result(reference, results, args.enable_assertion)
