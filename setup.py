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
import re

from setuptools import setup, find_packages
from setuptools.dist import Distribution

def read(filename):
  path = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
  with open(path , 'r') as f:
    return f.read()

def find_version(text):
  match = re.search(r"^__version__\s*=\s*['\"](.*)['\"]\s*$", text,
                    re.MULTILINE)
  return match.group(1)

# We don't build SWIG wrapper through build_clib, so specify here
# that we have external things. This gives a python-version specific
# wheel name instead of "any" (which would indicate pure python).
class BinaryDistribution(Distribution):
  def has_ext_modules(self):
    return True

setup(
  name='edgetpu',
  description='Edge TPU Python API',
  long_description=read('README.md'),
  license='Apache 2',
  version=find_version(read('edgetpu/__init__.py')),
  author='Coral',
  author_email='coral-support@google.com',
  url='https://coral.googlesource.com/edgetpu',
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ],
  packages=find_packages(),
  include_package_data=True,
  install_requires=[
      'numpy>=1.12.1',
      'Pillow>=4.0.0',
  ],
  python_requires='>=3.5.2',
  **({'distclass': BinaryDistribution} if os.name == 'nt' else {})
)
