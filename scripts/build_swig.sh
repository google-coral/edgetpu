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
readonly MAKEFILE="${SCRIPT_DIR}/../Makefile"

for i in "$@"; do
  if [[ "$i" == --clean ]]; then
    make -f "${MAKEFILE}" clean
  fi
done

# Python 3.5
make DOCKER_IMAGE=ubuntu:16.04 DOCKER_TARGETS=swig -f "${MAKEFILE}" docker-build
# Python 3.6
make DOCKER_IMAGE=ubuntu:18.04 DOCKER_TARGETS=swig -f "${MAKEFILE}" docker-build
# Python 3.7
make DOCKER_IMAGE=debian:buster DOCKER_TARGETS=swig -f "${MAKEFILE}" docker-build
