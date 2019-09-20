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

BENCHMARK_OPTIONS= # Empty by default
if [[ -f /etc/mendel_version ]] && [[ "${RUNTIME_PERF}" == "direct" ]]; then
  BENCHMARK_OPTIONS=--enable_assertion
fi

if [[ ${MACHINE} != "x86_64" ]]; then
  echo -e "${GREEN}--------------- Enable CPU performance mode -----------------${DEFAULT}"
  retry "sudo apt-get install -y linux-cpupower"
  sudo cpupower frequency-set --governor performance
fi

function run_benchmark {
  if [[ "${TEST_TYPE}" == "installed" ]]; then
    pushd /tmp
      python3 "${SCRIPT_DIR}/../benchmarks/$1.py" ${BENCHMARK_OPTIONS}
    popd
  else
    env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" PYTHONPATH="${SCRIPT_DIR}/.." \
      `which ${PYTHON}` "${SCRIPT_DIR}/../benchmarks/$1.py" ${BENCHMARK_OPTIONS}
  fi
}

echo -e "${BLUE}Benchmark of BasicEngine"
echo -e "Benchmark all supported models with BasicEngine.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
run_benchmark basic_engine_benchmarks

echo -e "${BLUE}Benchmark for ClassificationEngine"
echo -e "Benchmark all classification models with different image size.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
run_benchmark classification_benchmarks

echo -e "${BLUE}Benchmark for DetectionEngine"
echo -e "Benchmark all detection models with different image size.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
run_benchmark detection_benchmarks

echo -e "${BLUE}Benchmark for ImprintingEngine"
echo -e "Benchmark speed of transfer learning with Imprinting Engine.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
${SCRIPT_DIR}/../test_data/download_imprinting_test_data.sh
run_benchmark imprinting_benchmarks

echo -e "${BLUE}Benchmark for Cocompilation"
echo -e "Benchmark speed of cocompilation models.${DEFAULT}"
echo -e "${YELLOW}This test will take long time.${DEFAULT}"
run_benchmark cocompilation_benchmarks

echo -e "Benchmarks finished!"
