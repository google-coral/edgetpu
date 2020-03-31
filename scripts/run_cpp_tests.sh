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

# color scheme finish
red='tput setaf 1'
green='tput setaf 2'
ncreset='tput sgr0'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

if [[ "${CPU}" == "armv7a" ]]; then
  REQUEST_FOR_MULTI_EDGETPU_TEST=1000
else
  REQUEST_FOR_MULTI_EDGETPU_TEST=3000
fi

# Run tests.
run_env "${CPP_TESTS_DIR}/error_reporter_test"
${green}; echo "complete error_ reporter_test"; ${ncreset}

run_env "${CPP_TESTS_DIR}/version_test"
${green}; echo "complete version_test"; ${ncreset}

run_env "${CPP_TESTS_DIR}/bbox_utils_test"
${green}; echo "complete bbox_utilis_test"; ${ncreset}

run_env "${CPP_TESTS_DIR}/basic/basic_engine_native_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete basic_engine_native_test";${ncreset}

run_env "${CPP_TESTS_DIR}/basic/basic_engine_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete basic_engine_test";${ncreset}

run_env "${CPP_TESTS_DIR}/basic/edgetpu_resource_manager_test"
${green}; echo "complete edgetpu_resource_manager_test"; ${ncreset}

run_env "${CPP_TESTS_DIR}/basic/segmentation_models_test" \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete segmentation_models_test"; ${ncreset}

run_env "${CPP_TESTS_DIR}/basic/embedding_extractor_models_test" \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete embedding_extractor_models_test"; ${ncreset}

run_env "${CPP_TESTS_DIR}/basic/float_input_model_test" \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete basic/float_input_model_test";${ncreset}

run_env "${CPP_TESTS_DIR}/basic/inference_repeatability_test" \
    --stress_test_runs=20 \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete interference_repeatability_test"; ${ncreset}

run_env "${CPP_TESTS_DIR}/basic/inference_stress_test"  \
    --stress_test_runs=20 \
    --stress_with_sleep_test_runs=5 \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete interference_stress_test";${ncreset}

run_env "${CPP_TESTS_DIR}/basic/model_loading_stress_test" \
    --stress_test_runs=20 \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete model_loading_stress_test";${ncreset}

run_env "${CPP_TESTS_DIR}/classification/engine_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complte classification/engine_test";${ncreset}

run_env "${CPP_TESTS_DIR}/classification/models_test" \
  --test_data_dir="${TEST_DATA_DIR}" \
  --test_case_csv="classification_test_cases.csv"
${green}; echo "complete classfication/models_test";${ncreset}

run_env "${CPP_TESTS_DIR}/detection/engine_test" \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete detection/engine_test";${ncreset}

run_env "${CPP_TESTS_DIR}/detection/models_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete detection/models_test complete";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/imprinting/engine_native_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete imprinting/engine_native_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/imprinting/engine_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete imprinting/engine_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/backprop/cross_entropy_loss_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete backprop/cross_entropy_loss_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/backprop/sgd_updater_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete backprop/sgd_updater_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/backprop/test_utils_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete backprop/test_utils_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/backprop/multi_variate_normal_distribution_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete backprop/multi_variate_normal_distribution_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/backprop/softmax_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "backprop/softmax_test complete";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/backprop/fully_connected_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete backprop/fully_connected_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/backprop/softmax_regression_model_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete backprop/softmax_regression_model_test";${ncreset}

run_env "${CPP_TESTS_DIR}/learn/utils_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete learn/utils_test";${ncreset}

run_env "${CPP_TESTS_DIR}/posenet/models_test" \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete posenet/models_test";${ncreset}

run_env "${CPP_TESTS_DIR}/posenet/posenet_decoder_test"
${green}; echo "complete posenet/posenet_decoder_test";${ncreset}

#run_env "${CPP_TESTS_DIR}/tools/tflite_graph_util_test" \
#    --test_data_dir="${TEST_DATA_DIR}"
#${green}; echo "complete tools/tflite_graph_util_test";${ncreset}

# Multiple Edge TPU tests: need 2 TPU at least.
while [[ $(count_edgetpus) -lt 2 ]]; do
  echo "You need at least two Edge TPU devices plugged in to run the following tests."
  echo "Press Enter when ready."
  read LINE
done

run_env "${CPP_TESTS_DIR}/basic/multiple_tpus_inference_stress_test" \
    --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete multiple_tpus_inference_stress_test"; ${ncreset}

# Tools.
run_env "${CPP_TOOLS_DIR}/multiple_tpus_performance_analysis" \
    --test_data_dir="${TEST_DATA_DIR}" \
    --num_requests="${REQUEST_FOR_MULTI_EDGETPU_TEST}"
${green}; echo "multiple_tpus_performance_analysis and test complete"; ${ncreset}
# pipeline tests: need 4 TPU at least.

while [[ $(count_edgetpus) -lt 4 ]]; do
  echo "You need at least Four Edge TPU devices plugged in to run the following tests."
  echo "Press Enter when ready."
  read LINE
done

run_env "${CPP_TESTS_DIR}/pipeline/pipelined_model_runner_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete pipeline/pipelined_model_runner_test";${ncreset}

run_env "${CPP_TESTS_DIR}/pipeline/models_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete pipeline/models_test";${ncreset}

run_env "${CPP_TESTS_DIR}/pipeline/internal/segment_runner_test" \
  --test_data_dir="${TEST_DATA_DIR}"
${green}; echo "complete pipeline/internal/segment_runner_test";${ncreset}
