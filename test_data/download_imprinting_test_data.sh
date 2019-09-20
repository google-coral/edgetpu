#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]
  then OUTPUT_DIR="${SCRIPT_DIR}/open_image_v4_subset"
else
  OUTPUT_DIR="$1/open_image_v4_subset"
fi
mkdir -p "${OUTPUT_DIR}"

INPUT="${SCRIPT_DIR}/open_image_v4_subset.csv"
if [[ ! -f "${INPUT}" ]]; then
  echo "${INPUT} file not found"
  exit 99
fi

while IFS=',' read -r URL RAW_LABEL || [[ -n "${RAW_LABEL}" ]]; do
  LABEL="${RAW_LABEL/$'\r'/}"
  if curl --output /dev/null --silent --head --fail "${URL}"; then
      echo "Downloading ${URL} with label: ${LABEL}"
      wget -q -c "${URL}" -P "${OUTPUT_DIR}/${LABEL}"
  else
    echo "URL ${URL} does not exist, skipping"
  fi
done < "${INPUT}"

echo -e "Done!\nDownloaded files are in ${OUTPUT_DIR}"
