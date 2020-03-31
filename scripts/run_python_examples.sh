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
source "${SCRIPT_DIR}/setup_python_env.sh"

EXAMPLES_DIR="${SCRIPT_DIR}/../examples"

# Download data
DOWNLOAD=${SCRIPT_DIR}/../scripts/download.sh
DOWNLOADS_DIR=${SCRIPT_DIR}/../.downloads

"${DOWNLOAD}" "http://download.tensorflow.org/example_images/flower_photos.tgz" \
              "${DOWNLOADS_DIR}/flower_photos.tgz" \
              "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
if [[ ! -d "${DOWNLOADS_DIR}/flower_photos" ]]; then
  tar -C "${DOWNLOADS_DIR}" -xzf "${DOWNLOADS_DIR}/flower_photos.tgz"
fi

RETRAIN_DIR="/tmp/coral_retrain"
rm -rf "${RETRAIN_DIR}" && mkdir -p "${RETRAIN_DIR}"
ln -s "${DOWNLOADS_DIR}/flower_photos" "${RETRAIN_DIR}/flower_photos"

"${TEST_DATA_DIR}/download_imprinting_test_data.sh"

# Run examples
function run_example {
  pushd /tmp
  run_env python3 "$@"
  popd
}

# Run single Edge TPU examples.
if [[ "${FILTER_TESTS_BY_EDGETPU_NUM}" == "n" ]] || [[ "${NUM_EDGETPUS}" -gt 0 ]]; then
  run_example "${EXAMPLES_DIR}/classify_image.py" \
    --model="${TEST_DATA_DIR}/mobilenet_v1_1.0_224_quant_edgetpu.tflite" \
    --label="${TEST_DATA_DIR}/imagenet_labels.txt" \
    --image="${TEST_DATA_DIR}/cat.bmp"

  run_example "${EXAMPLES_DIR}/classify_image.py" \
    --model="${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
    --label="${TEST_DATA_DIR}/inat_bird_labels.txt" \
    --image="${TEST_DATA_DIR}/parrot.jpg"

  run_example "${EXAMPLES_DIR}/object_detection.py" \
    --model="${TEST_DATA_DIR}/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite" \
    --input="${TEST_DATA_DIR}/face.jpg" \
    --keep_aspect_ratio

  run_example "${EXAMPLES_DIR}/object_detection.py" \
    --model="${TEST_DATA_DIR}/ssd_mobilenet_v1_fine_tuned_pet_edgetpu.tflite" \
    --input="${TEST_DATA_DIR}/pets.jpg"

  run_example "${EXAMPLES_DIR}/imprinting_learning.py" \
    --model_path="${TEST_DATA_DIR}/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite" \
    --data="${TEST_DATA_DIR}/open_image_v4_subset" \
    --output="/tmp/imprinting_learning_test_model_$(date +%s).tflite"

  run_example "${EXAMPLES_DIR}/backprop_last_layer.py" \
    --data_dir "${RETRAIN_DIR}/flower_photos" \
    --embedding_extractor_path "${TEST_DATA_DIR}/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite" \
    --output_dir "${RETRAIN_DIR}/output"

  run_example "${EXAMPLES_DIR}/classify_image.py" \
    --model="${RETRAIN_DIR}/output/retrained_model_edgetpu.tflite" \
    --label="${RETRAIN_DIR}/output/label_map.txt" \
    --image="${TEST_DATA_DIR}/sunflower.bmp"

  run_example "${EXAMPLES_DIR}/semantic_segmetation.py" \
    --model="${TEST_DATA_DIR}/deeplabv3_mnv2_pascal_quant_edgetpu.tflite" \
    --input="${TEST_DATA_DIR}/bird.bmp" \
    --keep_aspect_ratio
fi

# Run multiple Edge TPU examples.
if [[ "${FILTER_TESTS_BY_EDGETPU_NUM}" == "n" ]] || [[ "${NUM_EDGETPUS}" -gt 1 ]]; then
  run_example "${EXAMPLES_DIR}/two_models_inference.py" \
    --classification_model="${TEST_DATA_DIR}/mobilenet_v1_1.0_224_quant_edgetpu.tflite" \
    --detection_model="${TEST_DATA_DIR}/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite" \
    --image="${TEST_DATA_DIR}/cat.bmp"
fi
