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

date

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
DEFAULT='\033[0m' # No Color

RUNTIME_PERF=throttled
AUTO_CONFIRM=n
TEST_TYPE=local

PYTHON=${PYTHON:-python3}

while :; do
  case $1 in
    -y|-Y) AUTO_CONFIRM=y  # Automatic confirm.
    ;;
    -m) RUNTIME_PERF=direct  # Use maximum frequency.
    ;;
    -i) TEST_TYPE=installed  # Test installed library.
    ;;
    *) break
  esac
  shift
done

MACHINE=$(uname -m)

case "${MACHINE}" in
  x86_64)
    CPU=k8
    HOST_GNU_TYPE=x86_64-linux-gnu
    REQUEST_FOR_MULTI_EDGETPU_TEST=3000
    ;;
  armv7l)
    CPU=armv7a
    HOST_GNU_TYPE=arm-linux-gnueabihf
    REQUEST_FOR_MULTI_EDGETPU_TEST=1000
    ;;
  aarch64)
    CPU=aarch64
    HOST_GNU_TYPE=aarch64-linux-gnu
    REQUEST_FOR_MULTI_EDGETPU_TEST=3000
    ;;
  *)
    error "Your platform is not supported."
    exit 1
    ;;
esac

LD_LIBRARY_PATH="${SCRIPT_DIR}/../libedgetpu/${RUNTIME_PERF}/${CPU}"

function retry {
  local TIMES=3
  for i in $(seq 0 "${TIMES}"); do
    [[ "${i}" -gt "0" ]] && sleep "$((15*$((2**$((${i}-1))))))"
    ${1} && break
    [[ "${i}" -eq "${TIMES}" ]] && exit 1
  done
}

function disable_cpu_scaling {
  if [[ $(apt-cache search linux-cpupower) ]]; then
    retry "sudo apt-get install -y linux-cpupower"
  else
    retry "sudo apt-get install -y linux-tools-$(uname -r)"
  fi
  echo -e "${GREEN}--------------- Enable CPU performance mode -----------------${DEFAULT}"
  sudo cpupower frequency-set --governor performance
}

retry "sudo apt-get update"
retry "sudo apt-get install -y libc6 libgcc1 libstdc++6 libusb-1.0-0 python3-numpy python3-pil"

if [[ "${MACHINE}" == "x86_64" ]]; then
  rm -rf "${SCRIPT_DIR}/.env"

  if [[ "${TEST_TYPE}" == "installed" ]]; then
    python3 -m pip install virtualenv
    python3 -m virtualenv --system-site-packages "${SCRIPT_DIR}/.env"
    source "${SCRIPT_DIR}/.env/bin/activate"
  else
    "${PYTHON}" -m pip install virtualenv
    "${PYTHON}" -m virtualenv "${SCRIPT_DIR}/.env"
    source "${SCRIPT_DIR}/.env/bin/activate"
    pip install numpy Pillow
  fi
fi

if [[ ! -f /etc/mendel_version ]] && [[ "${AUTO_CONFIRM}" != "y" ]]; then
  echo -e "${GREEN}Plug in USB Accelerator and press 'Enter' to continue.${DEFAULT}"
  read LINE
fi
