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

if [[ -d "${SCRIPT_DIR}/libedgetpu" ]]; then
  LIBEDGETPU_DIR="${SCRIPT_DIR}/libedgetpu"
else
  LIBEDGETPU_DIR="${SCRIPT_DIR}/../../libedgetpu"
fi

function info {
  echo -e "\033[0;32m${1}\033[0m"  # green
}

function warn {
  echo -e "\033[0;33m${1}\033[0m"  # yellow
}

function error {
  echo -e "\033[0;31m${1}\033[0m"  # red
}

function install_file {
  local name="${1}"
  local src="${2}"
  local dst="${3}"

  info "Installing ${name} [${dst}]..."
  if [[ -f "${dst}" ]]; then
    warn "File already exists. Replacing it..."
    rm -f "${dst}"
  fi
  cp -p "${src}" "${dst}"
}

if [[ "${EUID}" != 0 ]]; then
  error "Please use sudo to run as root."
  exit 1
fi

if [[ -f /etc/mendel_version ]]; then
  error "Looks like you're using a Coral Dev Board. You should instead use Debian packages to manage Edge TPU software."
  exit 1
fi

case `uname -m` in
  x86_64)
    HOST_GNU_TYPE=x86_64-linux-gnu
    CPU=k8
    ;;
  armv6l)
    HOST_GNU_TYPE=arm-linux-gnueabihf
    CPU=armv6
    ;;
  armv7l)
    HOST_GNU_TYPE=arm-linux-gnueabihf
    CPU=armv7a
    ;;
  aarch64)
    HOST_GNU_TYPE=aarch64-linux-gnu
    CPU=aarch64
    ;;
  *)
    error "Your platform is not supported."
    exit 1
    ;;
esac

cat << EOM
Warning: During normal operation, the Edge TPU Accelerator may heat up,
depending on the computation workloads and operating frequency. Touching the
metal part of the device after it has been operating for an extended period of
time may lead to discomfort and/or skin burns. As such, when running at the
default operating frequency, the device is intended to safely operate at an
ambient temperature of 35C or less. Or when running at the maximum operating
frequency, it should be operated at an ambient temperature of 25C or less.

Google does not accept any responsibility for any loss or damage if the device
is operated outside of the recommended ambient temperature range.
................................................................................
Would you like to enable the maximum operating frequency for the USB Accelerator? Y/N
EOM

read USE_MAX_FREQ
case "${USE_MAX_FREQ}" in
  [yY])
    info "Using maximum operating frequency for USB Accelerator."
    FREQ_DIR=direct
    ;;
  *)
    info "Using default operating frequency for USB Accelerator."
    FREQ_DIR=throttled
    ;;
esac

# Install dependent libraries.
info "Installing library dependencies..."
apt-get update && apt-get install -y libc6 libgcc1 libstdc++6 libusb-1.0-0
info "Done."

if [[ -x "$(command -v udevadm)" ]]; then
  install_file "device rule file" \
               "${LIBEDGETPU_DIR}/edgetpu-accelerator.rules" \
               "/etc/udev/rules.d/99-edgetpu-accelerator.rules"
  udevadm control --reload-rules && udevadm trigger
  info "Done."
fi

install_file "Edge TPU runtime library" \
             "${LIBEDGETPU_DIR}/${FREQ_DIR}/${CPU}/libedgetpu.so.1.0" \
             "/usr/lib/${HOST_GNU_TYPE}/libedgetpu.so.1.0"
ldconfig
info "Done."
