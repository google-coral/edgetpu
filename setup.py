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

import os

from setuptools import setup, find_packages

def read(filename):
  path = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
  with open(path , 'r') as f:
    return f.read()

setup(
  name='edgetpu',
  description='Edge TPU Python API',
  long_description=read('README.md'),
  license='Apache 2',
  version=__import__('edgetpu').__version__,
  author='Coral',
  author_email='coral-support@google.com',
  url='https://coral.googlesource.com/edgetpu',
  packages=find_packages(),
  include_package_data=True,
  install_requires=[
      'numpy>=1.12.1',
      'Pillow>=4.0.0',
  ],
  python_requires='>=3.5.2',
)
