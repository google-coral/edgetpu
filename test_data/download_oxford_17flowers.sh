#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/oxford_17flowers"
mkdir -p "${OUTPUT_DIR}"

# Download images.
oxford_17flowers_link="http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
wget -nd -c "${oxford_17flowers_link}" -P "${SCRIPT_DIR}"

tar -zxf "${SCRIPT_DIR}/17flowers.tgz" --directory "${OUTPUT_DIR}"
mv "${OUTPUT_DIR}/jpg/"*.jpg "${OUTPUT_DIR}"
rm -r "${OUTPUT_DIR}/jpg"

# Classes.
declare -a classes=("daffodil" "snowdrop" "lily_valley" "bluebell" "crocus"
                       "iris" "tigerlily" "tulip" "fritillary" "sunflower"
                       "daisy" "colts_foot" "dandelion" "cowslip" "buttercup"
                       "windflower" "pansy")

counter=1
# Images are named in accordance with order of classes.
for class in "${classes[@]}"
do
  # Each class has 80 images, we'll store them in sub directory named by class.
  mkdir -p "${OUTPUT_DIR}/${class}"
  for i in $(seq 1 80);
  do
    # File name of i-th image of current class.
    id=$(printf "%04d" $counter)
    # Move the image to the sub-directory named by class.
    mv "${OUTPUT_DIR}/image_${id}.jpg" "${OUTPUT_DIR}/${class}"
    counter=$((counter+1))
  done
done
