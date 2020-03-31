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

disable_cpu_scaling

# Run benchmarks.
run_env "${CPP_BENCHMARKS_DIR}/basic/models_benchmark" \
    --benchmark_out="${SCRIPT_DIR}/benchmark_${CPU}_${RUNTIME_PERF}.csv" \
    --benchmark_out_format=csv \
    --test_data_dir="${TEST_DATA_DIR}"

run_env "${CPP_BENCHMARKS_DIR}/basic/edgetpu_resource_manager_benchmark" \
    --benchmark_out="${SCRIPT_DIR}/resource_manager_benchmark_${CPU}_${RUNTIME_PERF}.csv" \
    --benchmark_out_format=csv \

run_env "${CPP_BENCHMARKS_DIR}/posenet/models_benchmark" \
    --benchmark_out="${SCRIPT_DIR}/posenet_benchmark_${CPU}_${RUNTIME_PERF}.csv" \
    --benchmark_out_format=csv \
    --test_data_dir="${TEST_DATA_DIR}"
