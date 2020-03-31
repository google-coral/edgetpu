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
readonly CMD="make deb deb-armhf deb-arm64 wheel && mkdir -p dist && mv -f ../*.deb dist"

"${SCRIPT_DIR}/build_swig.sh"

make DOCKER_SHELL_COMMAND="${CMD}" -f "${MAKEFILE}" docker-shell
