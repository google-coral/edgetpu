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
source "${SCRIPT_DIR}/prepare_test_env.sh"

function run_test {
  if [[ "${TEST_TYPE}" == "installed" ]]; then
    pushd /
      MPLBACKEND=agg python3 -m unittest -v "${SCRIPT_DIR}/../tests/$1.py"
    popd
  else
    env MPLBACKEND=agg LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" PYTHONPATH="${SCRIPT_DIR}/.." \
      `which ${PYTHON}` -m unittest -v tests.$1
  fi
}

echo -e "${BLUE}Test Exceptions${DEFAULT}"
run_test exception_test

echo -e "${BLUE}Unit test of BasicEngine"
echo -e "Run unit test with BasicEngine. It will run inference on all models once.${DEFAULT}"
run_test basic_engine_test

echo -e "${BLUE}ClassificationEngine"
echo -e "Now we'll run unit test of ClassificationEngine${DEFAULT}"
run_test classification_engine_test

echo -e "${BLUE}Multiple Edge TPUs test${DEFAULT}"
run_test multiple_tpus_test

echo -e "${BLUE}Edge TPU utils test${DEFAULT}"
run_test edgetpu_utils_test

echo -e "${BLUE}edgetpu_learn_utils_test${DEFAULT}"
run_test edgetpu_learn_utils_test

echo -e "${BLUE}DetectionEngine"
echo -e "Now we'll run unit test of DetectionEngine${DEFAULT}"
run_test detection_engine_test

echo -e "${BLUE}COCO test for DetectionEngine"
if [[ "${MACHINE}" == "x86_64" ]]; then
  # Takes a long time.
  echo -e "${YELLOW}This test will take long time.${DEFAULT}"
  echo -e "${GREEN}Download dependent libraries.${DEFAULT}"
  retry "sudo apt-get install -y libfreetype6-dev libpng-dev libqhull-dev libagg-dev python3-dev pkg-config"
  python3 -m pip install matplotlib
  python3 -m pip install cython
  python3 -m pip install git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI

  echo -e "${GREEN}Download coco data set.${DEFAULT}"
  ${SCRIPT_DIR}/../test_data/download_coco_val_data.sh

  echo -e "${GREEN}Start tests.${DEFAULT}"
  run_test coco_object_detection_test
else
  echo -e "${YELLOW}Skip.${DEFAULT}"
fi

echo -e "${BLUE}ImprintingEngine"
echo -e "Now we'll run unit test of ImprintingEngine${DEFAULT}"
run_test imprinting_engine_test

echo -e "${BLUE}Evaluation for ImprintingEngine${DEFAULT}"
if [[ "${MACHINE}" == "x86_64" ]]; then
  ${SCRIPT_DIR}/../test_data/download_oxford_17flowers.sh
  run_test imprinting_evaluation_test
else
  echo -e "${YELLOW}Skip.${DEFAULT}"
fi

echo -e "${BLUE}Cocompilation"
echo -e "Now we'll run unit test of Cocompilation${DEFAULT}"
run_test cocompilation_test


echo -e "Tests finished!"
