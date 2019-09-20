#!/bin/bash
# Downloads the ImageNet validation data set and labels.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/imagenet"
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# Helper function to download and unzip a .zip file
function download_and_untar() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [[ ! -f "${FILENAME}" ]]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  tar xvf "${FILENAME}"
}

# Download the validation data set.
IMAGENET_DATA_URL="http://www.image-net.org/challenges/LSVRC/2012/nnoupb"
IMAGENET_DATA_FILE="ILSVRC2012_img_val.tar"
download_and_untar ${IMAGENET_DATA_URL} ${IMAGENET_DATA_FILE}

# Download the ground truth label for classification.
IMAGENET_LABEL_URL="http://dl.caffe.berkeleyvision.org"
IMAGENET_LABEL_FILE="caffe_ilsvrc12.tar.gz"
download_and_untar ${IMAGENET_LABEL_URL} ${IMAGENET_LABEL_FILE}
