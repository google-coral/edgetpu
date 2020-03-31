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

function run_test {
  pushd /
  MPLBACKEND=agg run_env python3 -m unittest -v "${SCRIPT_DIR}/../tests/$1.py"
  popd
}

# Always run the following tests no matter how many Edge TPU devices are connected.
echo -e "${BLUE}Version Test"
echo -e "Checks runtime version.${DEFAULT}"
run_test version_test

echo -e "${BLUE}Unit test of BasicEngine"
echo -e "Run unit test with BasicEngine. It will run inference on all models once.${DEFAULT}"
run_test basic_engine_test

# Run single Edge TPU tests.
if [[ "${FILTER_TESTS_BY_EDGETPU_NUM}" == "n" ]] || [[ "${NUM_EDGETPUS}" -gt 0 ]]; then
  echo -e "${BLUE}Test Exceptions${DEFAULT}"
  run_test exception_test

  echo -e "${BLUE}ClassificationEngine"
  echo -e "Now we'll run unit test of ClassificationEngine${DEFAULT}"
  run_test classification_engine_test

  echo -e "${BLUE}Edge TPU utils test${DEFAULT}"
  run_test edgetpu_utils_test

  echo -e "${BLUE}edgetpu_learn_utils_test${DEFAULT}"
  run_test edgetpu_learn_utils_test

  echo -e "${BLUE}DetectionEngine"
  echo -e "Now we'll run unit test of DetectionEngine${DEFAULT}"
  run_test detection_engine_test

  echo -e "${BLUE}ImprintingEngine"
  echo -e "Now we'll run unit test of ImprintingEngine${DEFAULT}"
  run_test imprinting_engine_test

  echo -e "${BLUE}Evaluation for ImprintingEngine${DEFAULT}"
  if [[ "${CPU}" == "x86_64" ]]; then
    ${SCRIPT_DIR}/../test_data/download_oxford_17flowers.sh
    run_test imprinting_evaluation_test
  else
    echo -e "${YELLOW}Skip.${DEFAULT}"
  fi
fi

# Run multiple Edge TPU tests.
if [[ "${FILTER_TESTS_BY_EDGETPU_NUM}" == "n" ]] || [[ "${NUM_EDGETPUS}" -gt 1 ]]; then
  echo -e "${BLUE}Multiple Edge TPUs test${DEFAULT}"
  run_test multiple_tpus_test
fi

echo -e "Tests finished!"
