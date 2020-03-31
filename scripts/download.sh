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
if [[ "${#}" -ne 3 ]]; then
   echo "$0 <url> <filename> <sha256>" >&2
   exit 1
fi

readonly URL="${1}"
readonly FILENAME="${2}"
readonly SHA256="${3}"

function check {
  local file="${1}"
  local hash="${2}"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "${hash}" "*${file}" | shasum --algorithm 256 --check -
  else
    echo "${hash}  ${file}" | sha256sum --check -
  fi
}

# Check if file already exists and has correct checksum.
if [[ -f "${FILENAME}" ]]; then
  if check "${FILENAME}" "${SHA256}"; then
    exit 0
  else
    echo "File already exists but checksum doesn't match." >&2
    exit 1
  fi
fi

# Download file.
TMP_FILE=$(mktemp -t download_$(basename "${FILENAME}").XXXXXX)
if ! wget -O "${TMP_FILE}" "${URL}"; then
  echo "Cannot download file." >&2
  exit 1
fi

if check "${TMP_FILE}" "${SHA256}"; then
  mkdir -p "$(dirname "${FILENAME}")"
  mv "${TMP_FILE}" "${FILENAME}"
  exit 0
else
  echo "Downloaded file checksum doesn't match."
  exit 1
fi
