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
"""Utilities to process dataset."""

from edgetpu.utils.warning import deprecated
import re

def read_label_file(file_path):
  """Reads labels from text file and store it in a dict.

  This function supports following formats:

  1. Each line contains id and description separated by colon or space.
     Example: '0:cat' or '0 cat'.
  2. Each line contains description only. Label's id equals to row number.

  Args:
    file_path: String, path to the label file.

  Returns:
    Dict of (int, string) which maps label id to description.
  """
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  ret = {}
  for row_number, content in enumerate(lines):
    pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
    if len(pair) == 2 and pair[0].strip().isdigit():
      ret[int(pair[0])] = pair[1].strip()
    else:
      ret[row_number] = pair[0].strip()
  return ret

@deprecated
def ReadLabelFile(file_path):
  return read_label_file(file_path)
