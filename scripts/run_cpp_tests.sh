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
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/prepare_test_env.sh"

function run_test {
  if [[ "${TEST_TYPE}" == "installed" ]]; then
    pushd /tmp
      $@
    popd
  else
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" $@
  fi
}

OUT_DIR="${SCRIPT_DIR}/../out/${CPU}"
TEST_DATA_DIR="${SCRIPT_DIR}/../test_data"

# Run tests.
TESTS_DIR="${OUT_DIR}/tests/src/cpp"

run_test "${TESTS_DIR}/error_reporter_test"
run_test "${TESTS_DIR}/version_test"
run_test "${TESTS_DIR}/bbox_utils_test"

run_test "${TESTS_DIR}/basic/basic_engine_native_test" \
     --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/basic/basic_engine_test" \
     --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/basic/edgetpu_resource_manager_test"
run_test "${TESTS_DIR}/basic/inference_repeatability_test" \
    --stress_test_runs=20 \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/basic/inference_stress_test"  \
    --stress_test_runs=20 \
    --stress_with_sleep_test_runs=5 \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/basic/model_loading_stress_test" \
    --stress_test_runs=20 \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/basic/models_test" \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/classification/engine_test" \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/classification/models_test" \
    --test_data_dir="${TEST_DATA_DIR}"

run_test "${TESTS_DIR}/detection/engine_test" \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/detection/models_test" \
    --test_data_dir="${TEST_DATA_DIR}"

run_test "${TESTS_DIR}/learn/imprinting/engine_native_test" \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/learn/imprinting/engine_test" \
    --test_data_dir="${TEST_DATA_DIR}"
run_test "${TESTS_DIR}/learn/utils_test" \
    --test_data_dir="${TEST_DATA_DIR}"

run_test "${TESTS_DIR}/posenet/models_test" \
    --test_data_dir="${TEST_DATA_DIR}"

if [[ -f "${TESTS_DIR}/experimental/experimental_models_test" ]]; then
  run_test "${TESTS_DIR}/experimental/experimental_models_test" \
      --test_data_dir="${TEST_DATA_DIR}"
fi

disable_cpu_scaling

BENCHMARKS_DIR="${OUT_DIR}/benchmarks/src/cpp"

# Run benchmarks.
run_test "${BENCHMARKS_DIR}/basic/models_benchmark" \
    --benchmark_out="${SCRIPT_DIR}/benchmark_${CPU}_${RUNTIME_PERF}.csv" \
    --benchmark_out_format=csv \
    --test_data_dir="${TEST_DATA_DIR}"

run_test "${BENCHMARKS_DIR}/basic/edgetpu_resource_manager_benchmark" \
    --benchmark_out="${SCRIPT_DIR}/resource_manager_benchmark_${CPU}_${RUNTIME_PERF}.csv" \
    --benchmark_out_format=csv \

run_test "${BENCHMARKS_DIR}/posenet/models_benchmark" \
    --benchmark_out="${SCRIPT_DIR}/posenet_benchmark_${CPU}_${RUNTIME_PERF}.csv" \
    --benchmark_out_format=csv \
    --test_data_dir="${TEST_DATA_DIR}"

if [[ -f "${BENCHMARKS_DIR}/experimental/experimental_models_benchmark" ]]; then
  run_test "${BENCHMARKS_DIR}/experimental/experimental_models_benchmark" \
      --benchmark_out="${SCRIPT_DIR}/exp_benchmark_${CPU}_${RUNTIME_PERF}.csv" \
      --benchmark_out_format=csv \
      --test_data_dir="${TEST_DATA_DIR}"
fi

# Multiple Edge TPU tests.
echo "To run the following tests, please insert additional Edge TPU if only one Edge TPU is connected right now."
echo "Press Enter when ready."
read LINE

echo "Run multiple edgetpu tests..."
run_test "${TESTS_DIR}/basic/multiple_tpus_inference_stress_test" \
    --test_data_dir="${TEST_DATA_DIR}"

# Tools.
TOOLS_DIR="${OUT_DIR}/tools"

run_test "${TOOLS_DIR}/multiple_tpus_performance_analysis" \
    --test_data_dir="${TEST_DATA_DIR}" \
    --num_requests="${REQUEST_FOR_MULTI_EDGETPU_TEST}"

# Examples.
EXAMPLES_DIR="${OUT_DIR}/examples"

run_test "${EXAMPLES_DIR}/two_models_one_tpu" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/bird.bmp" \
  "${TEST_DATA_DIR}/sunflower.bmp"

run_test "${EXAMPLES_DIR}/two_models_two_tpus_threaded" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite" \
  "${TEST_DATA_DIR}/bird.bmp" \
  "${TEST_DATA_DIR}/sunflower.bmp"
