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
set -ex

date

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
DEFAULT='\033[0m' # No Color

RUNTIME_PERF=throttled
TEST_TYPE=local
FILTER_TESTS_BY_EDGETPU_NUM=n

PYTHON=${PYTHON:-python3}

TEST_DATA_DIR="${SCRIPT_DIR}/../test_data"

function retry {
  local TIMES=3
  for i in $(seq 0 "${TIMES}"); do
    [[ "${i}" -gt "0" ]] && sleep "$((15*$((2**$((${i}-1))))))"
    ${1} && break
    [[ "${i}" -eq "${TIMES}" ]] && exit 1
  done
}

while :; do
  case $1 in
    -n) FILTER_TESTS_BY_EDGETPU_NUM=y  # Filter tests based number of Edge TPUs, i.e. only run multiple TPU tests when there are multiple TPUs.
    ;;
    -m) RUNTIME_PERF=direct  # Use maximum frequency.
    ;;
    -i) TEST_TYPE=installed  # Test installed library.
    ;;
    *) break
  esac
  shift
done

readonly OS=$(uname -s)

function set_scaling_governor {
  for cpu in $(ls -d /sys/devices/system/cpu/cpu* | grep -P 'cpu\d+'); do
    local online="${cpu}/online"
    if [[ ! -f "${online}" || $(cat "${online}") == "1" ]]; then
      echo "${1}" | sudo tee "${cpu}/cpufreq/scaling_governor"
    fi
  done
}

function disable_cpu_scaling {
  if [[ -d /sys/devices/system/cpu ]]; then
    echo -e "${GREEN}--------------- Enabling CPU performance mode -----------------${DEFAULT}"
    set_scaling_governor performance
  fi
}

if [[ "${OS}" == "Linux" ]]; then
  case "$(uname -m)" in
    x86_64)
      readonly CPU=k8
      ;;
    armv7l)
      readonly CPU=armv7a
      ;;
    aarch64)
      readonly CPU=aarch64
      ;;
    *)
      echo "Your Linux architecture is not supported." 1>&2
      exit 1
      ;;
  esac

  if [[ "${TEST_TYPE}" != "installed" ]]; then
    for pkg in libc6 libgcc1 libstdc++6 libusb-1.0-0; do
      if ! dpkg -l "${pkg}" > /dev/null; then
        PACKAGES+=" ${pkg}"
      fi
    done
    if [[ -n "${PACKAGES}" ]]; then
      retry "sudo apt-get update"
      retry "sudo apt-get install -y ${PACKAGES}"
    fi
  fi
elif [[ "${OS}" == "Darwin" ]]; then
  readonly CPU=darwin
else
  echo "Your operating system is not supported." 1>&2
  exit 1
fi

readonly RUNTIME_DIR="${SCRIPT_DIR}/../libedgetpu/${RUNTIME_PERF}/${CPU}"

readonly CPP_OUT_DIR="${SCRIPT_DIR}/../out/${CPU}"
readonly CPP_TESTS_DIR="${CPP_OUT_DIR}/tests/src/cpp"
readonly CPP_BENCHMARKS_DIR="${CPP_OUT_DIR}/benchmarks/src/cpp"
readonly CPP_TOOLS_DIR="${CPP_OUT_DIR}/tools"
readonly CPP_EXAMPLES_DIR="${CPP_OUT_DIR}/examples"

function count_edgetpus {
  LD_LIBRARY_PATH="${RUNTIME_DIR}" "${PYTHON}" "${SCRIPT_DIR}/lstpu.py" | wc -l
}

while [[ $(count_edgetpus) -lt 1 ]]; do
  echo "You need at least one Edge TPU device plugged in."
  echo "Press Enter when ready."
  read LINE
done

NUM_EDGETPUS=$(count_edgetpus)

function run_env {
  if [[ "${TEST_TYPE}" == "installed" ]]; then
    if [[ "${CPU}" == "darwin" ]]; then
      # Use default MacPorts or Homebrew library location.
      env LD_LIBRARY_PATH=/usr/local/lib:/opt/local/lib "$@"
    else
      "$@"
    fi
  else
    env LD_LIBRARY_PATH="${RUNTIME_DIR}" PYTHONPATH="${SCRIPT_DIR}/.." "$@"
  fi
}
