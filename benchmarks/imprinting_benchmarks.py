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

import os
import subprocess
import time
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.learn.imprinting.engine import ImprintingEngine
import test_utils


def _get_shape(model):
  """Gets images shape required by model.

  Args:
    model: string, file name of the input model.

  Returns:
    (width, height)
  """
  basic_engine = BasicEngine(test_utils.test_data_path('imprinting', model))
  _, height, width, _ = basic_engine.get_input_tensor_shape()
  return (width, height)


def _benchmark_for_training(model, data_set):
  """Measures training time for given model and data set.

  Args:
    model: string, file name of the input model.
    data_set: string, name of the folder storing images. Labels file is also
      named as '[data_set].csv'.

  Returns:
    float, training time.
  """
  shape = _get_shape(model)
  engine = ImprintingEngine(test_utils.test_data_path('imprinting', model), keep_classes=False)
  output_model_path = '/tmp/model_for_benchmark.tflite'

  data_dir = test_utils.test_data_path(data_set)

  # The labels file is named as '[data_set].csv'.
  image_list_by_category = test_utils.prepare_classification_data_set(
      test_utils.test_data_path(data_set + '.csv'))

  start_time = time.monotonic()
  for category, image_list in image_list_by_category.items():
    category_dir = os.path.join(data_dir, category)
    image_list_by_category[category] = test_utils.prepare_images(
        image_list, category_dir, shape)
  end_time = time.monotonic()
  print('Image pre-processing time: ', end_time - start_time, 's')
  start_time = end_time
  for class_id, tensors in enumerate(image_list_by_category.values()):
    engine.train(tensors, class_id)
  engine.save_model(output_model_path)
  training_time = time.monotonic() - start_time
  print('Model: ', model)
  print('Data set : ', data_set)
  print('Training time : ', training_time, 's')
  # Remove the model.
  subprocess.call(['rm', output_model_path])
  return training_time


if __name__ == '__main__':
  args = test_utils.parse_args()
  machine = test_utils.machine_info()
  models, reference = test_utils.read_reference(
      'imprinting_reference_%s.csv' % machine)
  model_num = len(models)
  results = [('MODEL', 'DATA_SET', 'INFERENCE_TIME')]
  for cnt, name in enumerate(models, start=1):
    # 10 Categories, each has 20 images.
    data = 'open_image_v4_subset'
    print('---------------- ', cnt, '/', model_num, ' ----------------')
    results.append((name, data, _benchmark_for_training(name, data)))
  test_utils.save_as_csv('imprinting_benchmarks_%s_%s.csv' %
                         (machine, time.strftime('%Y%m%d-%H%M%S')), results)
  test_utils.check_result(reference, results, args.enable_assertion)
