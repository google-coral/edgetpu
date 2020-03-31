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

VENV_DIR="${SCRIPT_DIR}/../.env"

if [[ "${CPU}" == "x86_64" ]]; then
  # Sometimes "venv" module is not available out of the box on some Linux
  # distributions (it needs to be installed as separate python3-venv package).
  "${PYTHON}" -m pip install --user virtualenv

  if [[ "${TEST_TYPE}" == "installed" ]]; then
    # Assume 'edgetpu' package dependencies are already system-wide installed.
    "${PYTHON}" -m virtualenv --clear --system-site-packages "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
  else
    # Install 'edgetpu' package dependencies.
    "${PYTHON}" -m virtualenv --clear "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    python3 -m pip install numpy Pillow
  fi
fi
