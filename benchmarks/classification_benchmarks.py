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

"""Benchmark for Classification Engine Python API."""

import time
import timeit

from edgetpu.classification.engine import ClassificationEngine
import test_utils


def run_benchmark(model, image):
  """Returns average inference time in ms on specified model and image."""
  print('Benchmark for [%s] on %s' % (model, image))
  engine = ClassificationEngine(test_utils.test_data_path(model))
  iterations = 200 if 'edgetpu' in model else 10

  with test_utils.test_image(image) as img:
    result = 1000 * timeit.timeit(
        lambda: engine.classify_with_image(img, threshold=0.4, top_k=10),
        number=iterations) / iterations

  print('%.2f ms (iterations = %d)' % (result, iterations))
  return result

if __name__ == '__main__':
  args = test_utils.parse_args()
  machine = test_utils.machine_info()
  test_utils.check_cpu_scaling_governor_status()
  models, reference = test_utils.read_reference(
      'classification_reference_%s.csv' % machine)
  results = [('MODEL', 'IMAGE_NAME', 'INFERENCE_TIME')]
  for i, model in enumerate(models, start=1):
    print('-------------- Model %d / %d ---------------' % (i, len(models)))
    for image in ['cat.bmp', 'cat_720p.jpg', 'cat_1080p.jpg']:
      results.append((model, image, run_benchmark(model, image)))
  test_utils.save_as_csv('classification_benchmarks_%s_%s.csv' %
                             (machine, time.strftime('%Y%m%d-%H%M%S')),
                         results)
  test_utils.check_result(reference, results, args.enable_assertion)
