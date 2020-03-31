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

"""Test utils for benchmark and manual tests."""
import argparse
import collections
import contextlib
import csv
import os
import platform
import random
import urllib.parse

import numpy as np
from PIL import Image


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--enable_assertion', dest='enable_assertion',
                      action='store_true', default=False)
  return parser.parse_args()


def check_cpu_scaling_governor_status():
  """Checks whether CPU scaling enabled."""
  try:
    with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor') as f:
      status = f.read()

    if  status.strip() != 'performance':
      print('************************ WARNING *****************************')
      print('CPU scaling is enabled! Please switch to \'performance\' mode ')
      print('**************************************************************')
  except FileNotFoundError:
    pass


def machine_info():
  """Gets platform info to choose reference value."""
  machine = platform.machine()
  if machine == 'armv7l':
    with open('/proc/device-tree/model') as model_file:
      board_info = model_file.read()
    if 'Raspberry Pi 3 Model B Rev' in board_info:
      machine = 'rp3b'
    elif 'Raspberry Pi 3 Model B Plus Rev' in board_info:
      machine = 'rp3b+'
    elif 'Raspberry Pi 4 Model B Rev 1.1' in board_info:
      machine = 'rp4b'
    else:
      machine = 'unknown'
  return machine


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', 'test_data')

REFERENCE_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'reference')

BENCHMARK_RESULT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    'result')


def test_data_path(path, *paths):
  """Returns absolute path for a given test file."""
  return os.path.abspath(os.path.join(TEST_DATA_DIR, path, *paths))


def reference_path(path, *paths):
  """Returns absolute path for a given benchmark reference file."""
  return os.path.abspath(os.path.join(REFERENCE_DATA_DIR, path, *paths))


def benchmark_result_path(path, *paths):
  """Returns absolute path for a given benchmark result file."""
  return os.path.abspath(os.path.join(BENCHMARK_RESULT_DIR, path, *paths))


@contextlib.contextmanager
def test_image(path, *paths):
  """Returns opened test image."""
  with open(test_data_path(path, *paths), 'rb') as f:
    with Image.open(f) as image:
      yield image


def generate_random_input(seed, n):
  """Generates a list with n uint8 numbers."""
  random.seed(a=seed)
  return [random.randint(0, 255) for _ in range(n)]


def prepare_classification_data_set(filename):
  """Prepares classification data set.

  Args:
    filename: name of the csv file. It contains filenames of images and the
      categories they belonged.
  Returns:
    Dict with format {category_name : list of filenames}
  """
  ret = collections.defaultdict(list)
  with open(filename, mode='r') as csv_file:
    for row in csv.DictReader(csv_file):
      if not row['URL']:
        continue
      url = urllib.parse.urlparse(row['URL'])
      filename = os.path.basename(url.path)
      ret[row['Category']].append(filename)
  return ret


def prepare_images(image_list, directory, shape):
  """Reads images and converts them to numpy array with specified shape.

  Args:
    image_list: a list of strings storing file names.
    directory: string, path of directory storing input images.
    shape: a 2-D tuple represents the shape of required input tensor.
  Returns:
    A list of numpy.array.
  """
  ret = []
  for filename in image_list:
    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
      continue
    with Image.open(file_path) as img:
      img = img.resize(shape, Image.NEAREST)
      flat_img = np.asarray(img).flatten()
      if flat_img.shape[0] == shape[0] * shape[1] * 3:
        ret.append(flat_img)
  return np.array(ret)


def read_reference(file_name):
  """Reads reference from csv file.

  Args:
    file_name: string, name of the reference file.

  Returns:
    model_list: list of string.
    reference: { environment : reference_time}, environment is a string tuple
      while reference_time is a float number.
  """
  model_list = set()
  reference = {}
  with open(reference_path(file_name), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    # Drop first line(column names).
    next(reader)
    for row in reader:
      reference[tuple(row[:-1])] = float(row[-1])
      model_list.add(row[0])
  return sorted(model_list), reference


def check_result(reference, result_list, enable_assertion):
  """Checks result, warns when latency is abnormal.

  Args:
    reference: { environment : reference_time}, environment is a string tuple
      while reference_time is a float number.
    result_list: a list of tuple.
    enable_assertion: bool, throw assertion when unexpected latencty detected.
  """
  # Allow 30% variance.
  variance_threshold = 0.30
  print('******************** Check results *********************')
  cnt = 0
  # Drop first line(column name).
  for result in result_list[1:]:
    environment = result[:-1]
    inference_time = result[-1]

    if environment not in reference:
      print(' * No matching record for [%s].' % (','.join(environment)))
      cnt += 1
    reference_latency = reference[environment]
    up_limit = reference_latency * (1 + variance_threshold)
    down_limit = reference_latency * (1 - variance_threshold)

    if inference_time > up_limit:
      msg = ((' * Unexpected high latency! [%s]\n'
              '   Inference time: %s ms  Reference time: %s ms') %
             (','.join(environment), inference_time, reference_latency))
      print(msg)
      cnt += 1

    if inference_time < down_limit:
      msg = ((' * Unexpected low latency! [%s]\n'
              '   Inference time: %s ms  Reference time: %s ms') %
             (','.join(environment), inference_time, reference_latency))
      print(msg)
      cnt += 1
  print('******************** Check finished! *******************')
  if enable_assertion:
    assert cnt == 0, 'Benchmark test failed!'


def save_as_csv(file_name, result):
  """Saves benchmark result as csv files.

  Args:
    file_name: string, name of the saved file.
    result: A list of tuple.
  """
  os.makedirs(BENCHMARK_RESULT_DIR, exist_ok=True)
  with open(benchmark_result_path(file_name), 'w', newline='') as csv_file:
    writer = csv.writer(
        csv_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for line in result:
      writer.writerow(line)
  print(file_name, ' saved!')
