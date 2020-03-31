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

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DOWNLOADS_DIR="${SCRIPT_DIR}/../.downloads"
readonly OUTPUT_DIR="${SCRIPT_DIR}/oxford_17flowers"
mkdir -p "${OUTPUT_DIR}"
readonly DOWNLOAD=${SCRIPT_DIR}/../scripts/download.sh

# Download images.
"${DOWNLOAD}" "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz" \
              "${DOWNLOADS_DIR}/17flowers.tgz" \
              "fe38a60f8b4a95e657551247d3e7d799a3fafcdbc595be504b12839967823d70"
tar -C "${OUTPUT_DIR}" -xzf "${DOWNLOADS_DIR}/17flowers.tgz"

# Classes.
declare -a classes=("daffodil" "snowdrop" "lily_valley" "bluebell" "crocus"
                       "iris" "tigerlily" "tulip" "fritillary" "sunflower"
                       "daisy" "colts_foot" "dandelion" "cowslip" "buttercup"
                       "windflower" "pansy")

counter=1
# Images are named in accordance with order of classes.
for class in "${classes[@]}"; do
  # Each class has 80 images, we'll store them in sub directory named by class.
  mkdir -p "${OUTPUT_DIR}/${class}"
  for i in $(seq 1 80); do
    # File name of i-th image of current class.
    id=$(printf "%04d" $counter)
    # Move the image to the sub-directory named by class.
    mv "${OUTPUT_DIR}/jpg/image_${id}.jpg" "${OUTPUT_DIR}/${class}"
    counter=$((counter+1))
  done
done

rm -r "${OUTPUT_DIR}/jpg"
