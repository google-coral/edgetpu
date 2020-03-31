#!/bin/bash
#
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

# Single Edge TPU examples.
# Download data set for last layer backprop example.
if [ ! -d "/tmp/retrain/flower_photos/" ]
then
  mkdir -p /tmp/retrain
  curl http://download.tensorflow.org/example_images/flower_photos.tgz \
      | tar xz -C /tmp/retrain
  mogrify -format bmp /tmp/retrain/flower_photos/*/*.jpg
fi
run_env "${CPP_EXAMPLES_DIR}/backprop_last_layer" \
  --embedding_extractor_path="${TEST_DATA_DIR}/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite"

run_env "${CPP_EXAMPLES_DIR}/two_models_one_tpu" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/bird.bmp" \
  "${TEST_DATA_DIR}/sunflower.bmp"

# Multiple Edge TPU examples.
while [[ $(count_edgetpus) -lt 2 ]]; do
  echo "You need at least two Edge TPU devices plugged in to run the following tests."
  echo "Press Enter when ready."
  read LINE
done

run_env "${CPP_EXAMPLES_DIR}/two_models_two_tpus_threaded" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/bird.bmp" \
  "${TEST_DATA_DIR}/sunflower.bmp"
