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
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EDGETPU_DIR="${SCRIPT_DIR}/../"
source "${SCRIPT_DIR}/prepare_test_env.sh"


function run_example {
  if [[ "${TEST_TYPE}" == "installed" ]]; then
    pushd /tmp
      python3 $@
    popd
  else
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" PYTHONPATH="${SCRIPT_DIR}/.." \
      `which ${PYTHON}` $@
  fi
}

run_example "${EDGETPU_DIR}/examples/classify_image.py" \
  --model="${EDGETPU_DIR}/test_data/mobilenet_v1_1.0_224_quant_edgetpu.tflite" \
  --label="${EDGETPU_DIR}/test_data/imagenet_labels.txt" \
  --image="${EDGETPU_DIR}/test_data/cat.bmp"

run_example "${EDGETPU_DIR}/examples/classify_image.py" \
  --model="${EDGETPU_DIR}/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
  --label="${EDGETPU_DIR}/test_data/inat_bird_labels.txt" \
  --image="${EDGETPU_DIR}/test_data/parrot.jpg"

run_example "${EDGETPU_DIR}/examples/object_detection.py" \
  --model="${EDGETPU_DIR}/test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite" \
  --input="${EDGETPU_DIR}/test_data/face.jpg" \
  --keep_aspect_ratio

run_example "${EDGETPU_DIR}/examples/object_detection.py" \
  --model="${EDGETPU_DIR}/test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite" \
  --input="${EDGETPU_DIR}/test_data/pets.jpg"

run_example "${EDGETPU_DIR}/examples/two_models_inference.py" \
  --classification_model="${EDGETPU_DIR}/test_data/mobilenet_v1_1.0_224_quant_edgetpu.tflite" \
  --detection_model="${EDGETPU_DIR}/test_data/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite" \
  --image="${EDGETPU_DIR}/test_data/cat.bmp"

"${EDGETPU_DIR}/test_data/download_imprinting_test_data.sh"
run_example "${EDGETPU_DIR}/examples/imprinting_learning.py" \
  --model_path="${EDGETPU_DIR}/test_data/imprinting/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite" \
  --data="${EDGETPU_DIR}/test_data/open_image_v4_subset" \
  --output="/tmp/imprinting_learning_test_model_$(date +%s).tflite"

RETRAIN_FOLDER="/tmp/retrain_$(date +%s)"
rm -rf "${RETRAIN_FOLDER}"
mkdir -p "${RETRAIN_FOLDER}"
curl http://download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C "${RETRAIN_FOLDER}"
run_example "${EDGETPU_DIR}/examples/backprop_last_layer.py" \
  --data_dir "${RETRAIN_FOLDER}/flower_photos" \
  --embedding_extractor_path "${EDGETPU_DIR}/test_data/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite" \
  --output_dir "${RETRAIN_FOLDER}/output"

run_example "${EDGETPU_DIR}/examples/classify_image.py" \
  --model="${RETRAIN_FOLDER}/output/retrained_model_edgetpu.tflite" \
  --label="${RETRAIN_FOLDER}/output/label_map.txt" \
  --image="${EDGETPU_DIR}/test_data/sunflower.bmp"

run_example "${EDGETPU_DIR}/examples/semantic_segmetation.py" \
  --model="${EDGETPU_DIR}/test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite" \
  --input="${EDGETPU_DIR}/test_data/bird.bmp" \
  --keep_aspect_ratio

