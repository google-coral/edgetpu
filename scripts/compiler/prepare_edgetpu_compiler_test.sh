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

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MAKEFILE="${SCRIPT_DIR}/../../Makefile"
readonly TEST_DIR=${TEST_DIR:="/tmp/edgetpu_compiler_tests"}

rm -rf "${TEST_DIR}" && mkdir -p "${TEST_DIR}"

for i in "$@"; do
  if [[ "$i" == --clean ]]; then
    make -f "${MAKEFILE}" clean
  fi
done

# Build for k8 (use Ubuntu 16.04 for compatibility with most platforms).
make DOCKER_IMAGE=ubuntu:16.04 \
     DOCKER_CPUS=k8 \
     DOCKER_TARGETS="tests benchmarks" \
     -f "${MAKEFILE}" docker-build

cp -rf "${SCRIPT_DIR}/../../out" "${TEST_DIR}"
cp -rf "${SCRIPT_DIR}/../../test_data" "${TEST_DIR}/old_data"

# Models in the following subfolders will be skiped.
rm -rf "${TEST_DIR}/old_data/invalid_models"
rm -rf "${TEST_DIR}/old_data/posenet"
rm -rf "${TEST_DIR}/old_data/tools"

cp -rf "${SCRIPT_DIR}/../../compiler" "${TEST_DIR}"
cp -rf "${SCRIPT_DIR}/../../libedgetpu" "${TEST_DIR}"

cp -f "${SCRIPT_DIR}/edgetpu_compiler_tests.py" "${TEST_DIR}"
cp -f "${SCRIPT_DIR}/edgetpu_compiler_tests_config.txt" "${TEST_DIR}"
cp -f "${SCRIPT_DIR}/edgetpu_compiler_benchmarks_config.txt" "${TEST_DIR}"

echo "Edge TPU compiler test prepared at ${TEST_DIR}"
