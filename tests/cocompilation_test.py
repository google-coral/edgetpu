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
from . import classification_engine_test


class TestCocompilationModelPythonAPI(classification_engine_test.ClassificationEngineTestCase):

  def test_various_cocompiled_models(self):
    # Mobilenet V1 and Mobilenet V2.
    self._test_classify_cat(
        'cocompilation/mobilenet_v1_1.0_224_quant_cocompiled_with_mobilenet_v2_1.0_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.78), ('tiger cat', 0.128)]
    )
    self._test_classify_cat(
        'cocompilation/mobilenet_v2_1.0_224_quant_cocompiled_with_mobilenet_v1_1.0_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.84)]
    )
    # Inception V1 and Inception V4.
    self._test_classify_cat(
        'cocompilation/inception_v1_224_quant_cocompiled_with_inception_v4_299_quant_edgetpu.tflite',
        [('tabby, tabby cat', 0.41),
         ('Egyptian cat', 0.35),
         ('tiger cat', 0.156)]
    )
    self._test_classify_cat(
        'cocompilation/inception_v4_299_quant_cocompiled_with_inception_v1_224_quant_edgetpu.tflite',
        [('Egyptian cat', 0.45),
         ('tabby, tabby cat', 0.3),
         ('tiger cat', 0.15)]
    )

if __name__ == '__main__':
  unittest.main()
