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

"""Script to analyze performance speedup when using multiple Edge TPUs.

Basically, this script times how long it costs to run certain amount of
inferences, `num_inferences`, with 1, 2, ..., (max # Edge TPU) Edge TPU.

It then reports the speedup one can get by using multiple Edge TPU
devices. Speedup is defined as:
    k_tpu_speedup = 1_tpu_time / k_tpu_time

Note:
*) This is timing a particular usage pattern, and real use case might vary a
   lot. But it gives a rough idea about the speedup.
*) Python interpreter has GIL, so code cannot take advantage of multiple cores
   of host CPU. And if the task is CPU bounded, it is not advised to use
   multiple threads inside python interpreter.
*) This script can be quite time-consuming. Adjust `num_inferences` accordingly.
"""

import logging
import threading
import time

from edgetpu.basic import edgetpu_utils
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
import numpy as np
from PIL import Image
import test_utils


def run_inference_job(model_name,
                      input_filename,
                      num_inferences,
                      num_threads,
                      task_type,
                      check_result=None):
  """Runs classification or detection job with `num_threads`.

  Args:
    model_name: string
    input_filename: string
    num_inferences: int
    num_threads: int
    task_type: string, `classification` or `detection`
    check_result: callback function to check the result.

  Returns:
    double, wall time (in seconds) for running the job.
  """

  def thread_job(model_name,
                 input_filename,
                 num_inferences,
                 task_type,
                 check_result=None):
    """Runs classification or detection job on one Python thread."""
    tid = threading.get_ident()
    logging.info('Thread: %d, # inferences: %d, model: %s', tid, num_inferences,
                 model_name)
    if task_type == 'classification':
      task_engine = ClassificationEngine
      inference_func = ClassificationEngine.classify_with_input_tensor
    else:
      assert task_type == 'detection'
      task_engine = DetectionEngine
      inference_func = DetectionEngine.detect_with_input_tensor

    engine = task_engine(test_utils.test_data_path(model_name))
    logging.info('Thread: %d, using device %s', tid, engine.device_path())
    with test_utils.test_image(input_filename) as img:
      _, height, width, _ = engine.get_input_tensor_shape()
      resized_img = np.asarray(img.resize((width, height),
                                          Image.NEAREST)).flatten()
      for _ in range(num_inferences):
        result = inference_func(engine, resized_img, top_k=1)

    # Check result once.
    if check_result:
      check_result(result)
    assert len(result) == 1
    logging.info('Thread: %d, model: %s done', tid, model_name)

  start_time = time.perf_counter()
  # Round up a bit if not divisible.
  num_inferences_per_thread = (num_inferences + num_threads - 1) // num_threads
  workers = []
  for _ in range(num_threads):
    workers.append(
        threading.Thread(
            target=thread_job,
            args=(model_name, input_filename, num_inferences_per_thread,
                  task_type, check_result)))

  for worker in workers:
    worker.start()
  for worker in workers:
    worker.join()
  return time.perf_counter() - start_time


def main():
  num_inferences = 30000
  input_filename = 'cat.bmp'
  num_tpus = len(
      edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_NONE))
  model_names = [
      'mobilenet_v1_1.0_224_quant_edgetpu.tflite',
      'mobilenet_v2_1.0_224_quant_edgetpu.tflite',
      'mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite',
      'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
      'inception_v1_224_quant_edgetpu.tflite',
      'inception_v2_224_quant_edgetpu.tflite',
      'inception_v3_299_quant_edgetpu.tflite',
      'inception_v4_299_quant_edgetpu.tflite',
  ]

  def show_speedup(inference_costs):
    logging.info('Single Edge TPU base time: %f seconds', inference_costs[0])
    for i in range(1, len(inference_costs)):
      logging.info('# TPUs: %d, speedup: %f', i + 1,
                   inference_costs[0] / inference_costs[i])

  inference_costs_map = {}
  for model_name in model_names:
    task_type = 'classification'
    if 'ssd' in model_name:
      task_type = 'detection'
    inference_costs_map[model_name] = [0.0] * num_tpus
    for num_threads in range(num_tpus, 0, -1):
      cost = run_inference_job(model_name, input_filename, num_inferences,
                               num_threads, task_type)
      inference_costs_map[model_name][num_threads - 1] = cost
      logging.info('model: %s, # threads: %d, cost: %f seconds', model_name,
                   num_threads, cost)
    show_speedup(inference_costs_map[model_name])

  logging.info('============Summary==========')
  for model_name in model_names:
    inference_costs = inference_costs_map[model_name]
    logging.info('---------------------------')
    logging.info('Model: %s', model_name)
    show_speedup(inference_costs)


if __name__ == '__main__':
  logging.basicConfig(
      format=(
          '%(asctime)s.%(msecs)03d p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
      ),
      level=logging.INFO,
      datefmt='%Y-%m-%d,%H:%M:%S')
  main()
