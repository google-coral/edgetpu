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

from . import test_utils
from edgetpu.basic import edgetpu_utils
from edgetpu.basic.basic_engine import BasicEngine


class EdgeTpuUtilsTest(unittest.TestCase):

  def test_list_edge_tpu_paths(self):
    num_all = len(
        edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_NONE))
    unused_engine = BasicEngine(
        test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
    num_assigned = len(
        edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_ASSIGNED))
    self.assertEqual(num_assigned, 1)
    num_available = len(
        edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED))
    self.assertEqual(num_available, num_all - 1)

  def test_use_all_edge_tpu(self):
    available_tpus = edgetpu_utils.ListEdgeTpuPaths(
        edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
    recorded_tpus = []
    engine_list = []
    for _ in available_tpus:
      engine = BasicEngine(
          test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))
      recorded_tpus.append(engine.device_path())
      engine_list.append(engine)

    remaining_tpus = edgetpu_utils.ListEdgeTpuPaths(
        edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
    self.assertEqual(0, len(remaining_tpus))
    self.assertTupleEqual(tuple(recorded_tpus), available_tpus)
