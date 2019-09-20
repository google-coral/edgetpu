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
#
# This script will generate edgetpu_api.tar.gz which contains Edge TPU
# Python API.

set -e
set -x

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DIST_DIR="${SCRIPT_DIR}/../dist"
readonly DIR_NAME=edgetpu_runtime

rm -rf "${DIST_DIR}/${DIR_NAME}"
mkdir -p "${DIST_DIR}/${DIR_NAME}"
cp -r "${SCRIPT_DIR}/../libedgetpu" \
      "${SCRIPT_DIR}/runtime/install.sh" \
      "${SCRIPT_DIR}/runtime/uninstall.sh" \
      "${DIST_DIR}/${DIR_NAME}"
tar -C "${DIST_DIR}" \
    -zcvf "${DIST_DIR}/${DIR_NAME}_$(date '+%Y%m%d').tar.gz" \
    "${DIR_NAME}"
