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

import edgetpu.basic.edgetpu_utils as utils
import sys

__version__ = "2.14.0"

expected=utils.SUPPORTED_RUNTIME_VERSION
installed=utils.GetRuntimeVersion()
if installed.find(expected) == -1:
  print("[WARNING] requires runtime %s, but installed runtime has version %s. "
        "It's not guaranteed that different versions of APIs and runtime "
        "could work together properly." % (expected, installed),
        file=sys.stderr)
