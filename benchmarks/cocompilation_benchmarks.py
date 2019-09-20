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

"""Benchmark of cocompiled models.

Benchmark are measured with CPU 'performance' mode. To enable it, you need to
install 'cpupower' and run:
sudo cpupower frequency-set --governor performance

The reference number is measured on:
  - 'x86_64': Intel Xeon W-2135(4.50GHz) + Edge TPU accelarator + USB 3.0
  - 'rp3b': Raspberry Pi 3 B (version1.2)+ Edge TPU accelarator + USB 2.0
  - 'rp3b+': Raspberry Pi 3 B+ (version1.3)+ Edge TPU accelarator + USB 2.0
  - 'aarch64': Edge TPU dev board.
"""

import time
import timeit

from edgetpu.basic import edgetpu_utils
from edgetpu.basic.basic_engine import BasicEngine
import numpy as np
import test_utils


def _run_inferences(engines, input_data_list):
  """Runs an iteration of inferences for each engine with a random inpt.

  Args:
    engines: list of basic engines.
    input_data_list: list of random input data.
  """

  for engine, input_data in zip(engines, input_data_list):
    engine.run_inference(input_data)

def _run_benchmark_for_cocompiled_models(model_names):
  """Runs benchmark for a given model set with random inputs. Models run
  inferences alternately with random inputs. It benchmarks the total time
  running each model once.

  Args:
    model_names: list of string, file names of the models.

  Returns:
    float, average sum of inferences times.
  """
  iterations = 200
  print('Benchmark for ', model_names)

  engines = []
  input_data_list = []
  edge_tpus = edgetpu_utils.ListEdgeTpuPaths(
      edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)

  for model_name in model_names:
    # Run models on a single edgetpu to achieve accurate benchmark results.
    engine = BasicEngine(test_utils.test_data_path(model_name), edge_tpus[0])

    # Prepare a random generated input.
    input_size = engine.required_input_array_size()
    random_input = test_utils.generate_random_input(1, input_size)

    # Convert it to a numpy.array.
    input_data = np.array(random_input, dtype=np.uint8)

    engines.append(engine)
    input_data_list.append(input_data)

  benchmark_time = timeit.timeit(
      lambda: _run_inferences(engines, input_data_list),
      number=iterations)

  # Time consumed for each iteration (milliseconds).
  time_per_inference = (benchmark_time / iterations) * 1000
  print(time_per_inference, 'ms (iterations = ', iterations, ')')
  return time_per_inference

if __name__ == '__main__':
  args = test_utils.parse_args()
  machine = test_utils.machine_info()
  test_utils.check_cpu_scaling_governor_status()
  # Read references from csv file.
  modelsets_list, reference = test_utils.read_reference(
      'cocompilation_reference_%s.csv' % machine)
  total_modelsets = len(modelsets_list)
  # Put column names in first row.
  results = [('MODELS', 'INFERENCE_TIME')]
  for cnt, modelsets in enumerate(modelsets_list, start=1):
    print('-------------- Models ', cnt, '/', total_modelsets, ' ---------------')
    results.append((modelsets, _run_benchmark_for_cocompiled_models(modelsets.split(','))))
  test_utils.save_as_csv(
      'cocompilation_benchmarks_%s_%s.csv' % (
          machine, time.strftime('%Y%m%d-%H%M%S')),
      results)
  test_utils.check_result(reference, results, args.enable_assertion)
