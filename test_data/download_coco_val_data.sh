#!/bin/bash
# Downloads the COCO validation data set to validate the accuracy of coco object
# detection models.
#
# Install dependencies:
#   sudo apt-get install -y libfreetype6-dev libpng-dev libqhull-dev libagg-dev python3-dev pkg-config
#   pip install matplotlib
#   pip install cython
#   pip install git+https://github.com/cocodataset/cocoapi#subdirectory=PythonAPI
#
# Setup proper matplotlib backed:
#   export MPLBACKEND=agg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/coco"
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# Helper function to download and unzip a .zip file
function download_and_unzip() {
	local BASE_URL=${1}
	local FILENAME=${2}

	if [[ ! -f "${FILENAME}" ]]; then
  	echo "Downloading ${FILENAME} to $(pwd)"
  	wget -nd -c "${BASE_URL}/${FILENAME}"
	else
  	echo "Skipping download of ${FILENAME}"
	fi
	echo "Unzipping ${FILENAME}"
	unzip -nq "${FILENAME}"
}

# Download the validation data set.
COCO_DATA_URL="http://images.cocodataset.org/zips"
VAL_DATA_FILE="val2017.zip"
download_and_unzip ${COCO_DATA_URL} ${VAL_DATA_FILE}

# Download the annotations.
COCO_ANN_URL="http://images.cocodataset.org/annotations"
ANN_FILE="annotations_trainval2017.zip"
download_and_unzip ${COCO_ANN_URL} ${ANN_FILE}