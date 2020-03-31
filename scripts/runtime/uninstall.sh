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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function info {
  echo -e "\033[0;32m${1}\033[0m"  # green
}

function warn {
  echo -e "\033[0;33m${1}\033[0m"  # yellow
}

function error {
  echo -e "\033[0;31m${1}\033[0m"  # red
}

if [[ "${EUID}" != 0 ]]; then
  error "Please use sudo to run as root."
  exit 1
fi

if [[ -f /etc/mendel_version ]]; then
  error "Looks like you're using a Coral Dev Board. You should instead use Debian packages to manage Edge TPU software."
  exit 1
fi

readonly OS="$(uname -s)"
readonly MACHINE="$(uname -m)"

if [[ "${OS}" == "Linux" ]]; then
  case "${MACHINE}" in
    x86_64)
      HOST_GNU_TYPE=x86_64-linux-gnu
      CPU_DIR=k8
      ;;
    armv7l)
      HOST_GNU_TYPE=arm-linux-gnueabihf
      CPU_DIR=armv7a
      ;;
    aarch64)
      HOST_GNU_TYPE=aarch64-linux-gnu
      CPU_DIR=aarch64
      ;;
    *)
      error "Your Linux platform is not supported. There's nothing to uninstall."
      exit 1
      ;;
  esac
elif [[ "${OS}" == "Darwin" ]]; then
  CPU=darwin
else
  error "Your operating system is not supported. There's nothing to uninstall."
  exit 1
fi

if [[ "${CPU}" == "darwin" ]]; then
  LIBEDGETPU_LIB_DIR="/usr/local/lib"

  if [[ -f "${LIBEDGETPU_LIB_DIR}/libedgetpu.1.0.dylib" ]]; then
    info "Uninstalling Edge TPU runtime library..."
    rm -f "${LIBEDGETPU_LIB_DIR}/libedgetpu.1.0.dylib"
    info "Done"
  fi

  if [[ -L "${LIBEDGETPU_LIB_DIR}/libedgetpu.1.dylib" ]]; then
    info "Uninstalling Edge TPU runtime library symlink..."
    rm -f "${LIBEDGETPU_LIB_DIR}/libedgetpu.1.dylib"
    info "Done"
  fi
else
  if [[ -x "$(command -v udevadm)" ]]; then
    UDEV_RULE_PATH="/etc/udev/rules.d/99-edgetpu-accelerator.rules"
    if [[ -f "${UDEV_RULE_PATH}" ]]; then
      info "Uninstalling device rule file [${UDEV_RULE_PATH}]..."
      rm -f "${UDEV_RULE_PATH}"
      udevadm control --reload-rules && udevadm trigger
      info "Done."
    fi
  fi

  LIBEDGETPU_DST="/usr/lib/${HOST_GNU_TYPE}/libedgetpu.so.1.0"
  if [[ -f "${LIBEDGETPU_DST}" ]]; then
    info "Uninstalling Edge TPU runtime library [${LIBEDGETPU_DST}]..."
    rm -f "${LIBEDGETPU_DST}"
    ldconfig
    info "Done."
  fi
fi
