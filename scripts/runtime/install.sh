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
  cp -a "${src}" "${dst}"
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
      error "Your Linux platform is not supported."
      exit 1
      ;;
  esac
elif [[ "${OS}" == "Darwin" ]]; then
  CPU=darwin

  MACPORTS_PATH_AUTO="$(command -v port || true)"
  MACPORTS_PATH="${MACPORTS_PATH_AUTO:-/opt/local/bin/port}"

  BREW_PATH_AUTO="$(command -v brew || true)"
  BREW_PATH="${BREW_PATH_AUTO:-/usr/local/bin/brew}"

  if [[ -x "${MACPORTS_PATH}" ]]; then
    DARWIN_INSTALL_COMMAND="${MACPORTS_PATH}"
    DARWIN_INSTALL_USER="$(whoami)"
  elif [[ -x "${BREW_PATH}" ]]; then
    DARWIN_INSTALL_COMMAND="${BREW_PATH}"
    DARWIN_INSTALL_USER="${SUDO_USER}"
  else
    error "You need to install either Homebrew or MacPorts first."
    exit 1
  fi
else
  error "Your operating system is not supported."
  exit 1
fi

cat << EOM
Warning: If you're using the Coral USB Accelerator, it may heat up during operation, depending
on the computation workloads and operating frequency. Touching the metal part of the USB
Accelerator after it has been operating for an extended period of time may lead to discomfort
and/or skin burns. As such, if you enable the Edge TPU runtime using the maximum operating
frequency, the USB Accelerator should be operated at an ambient temperature of 25°C or less.
Alternatively, if you enable the Edge TPU runtime using the reduced operating frequency, then
the device is intended to safely operate at an ambient temperature of 35°C or less.

Google does not accept any responsibility for any loss or damage if the device
is operated outside of the recommended ambient temperature range.

Note: This question affects only USB-based Coral devices, and is irrelevant for PCIe devices.
................................................................................
Would you like to enable the maximum operating frequency for your Coral USB device? Y/N
EOM

read USE_MAX_FREQ
case "${USE_MAX_FREQ}" in
  [yY])
    info "Using the maximum operating frequency for Coral USB devices."
    FREQ_DIR=direct
    ;;
  *)
    info "Using the reduced operating frequency for Coral USB devices."
    FREQ_DIR=throttled
    ;;
esac

if [[ "${CPU}" == "darwin" ]]; then
  sudo -u "${DARWIN_INSTALL_USER}" "${DARWIN_INSTALL_COMMAND}" install libusb

  DARWIN_INSTALL_LIB_DIR="$(dirname "$(dirname "${DARWIN_INSTALL_COMMAND}")")/lib"
  LIBEDGETPU_LIB_DIR="/usr/local/lib"
  mkdir -p "${LIBEDGETPU_LIB_DIR}"

  install_file "Edge TPU runtime library" \
               "${LIBEDGETPU_DIR}/${FREQ_DIR}/darwin/libedgetpu.1.0.dylib" \
               "${LIBEDGETPU_LIB_DIR}"

  install_file "Edge TPU runtime library symlink" \
               "${LIBEDGETPU_DIR}/${FREQ_DIR}/darwin/libedgetpu.1.dylib" \
               "${LIBEDGETPU_LIB_DIR}"

  install_name_tool -id  "${LIBEDGETPU_LIB_DIR}/libedgetpu.1.dylib" \
                         "${LIBEDGETPU_LIB_DIR}/libedgetpu.1.0.dylib"

  install_name_tool -change "/opt/local/lib/libusb-1.0.0.dylib" \
                            "${DARWIN_INSTALL_LIB_DIR}/libusb-1.0.0.dylib" \
                            "${LIBEDGETPU_LIB_DIR}/libedgetpu.1.0.dylib"
else
  for pkg in libc6 libgcc1 libstdc++6 libusb-1.0-0; do
    if ! dpkg -l "${pkg}" > /dev/null; then
      PACKAGES+=" ${pkg}"
    fi
  done

  if [[ -n "${PACKAGES}" ]]; then
    info "Installing library dependencies:${PACKAGES}..."
    apt-get update && apt-get install -y ${PACKAGES}
    info "Done."
  fi

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
  ldconfig  # Generates libedgetpu.so.1 symlink
  info "Done."
fi
