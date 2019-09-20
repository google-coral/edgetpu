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
"""Provide error message for deprecated EdgeTpu APIs."""

import functools
import inspect
import logging
import re

def camelcase_to_lowercase(name):
  """Converts a camel case name to lower case one.

  Args:
    name: string, function name in camel case.
  Returns:
    lower case name of the function
  """
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def deprecated(func):
  """This is a decorator which can be used to mark functions as deprecated.
  """
  @functools.wraps(func)
  def new_func(*args, **kwargs):
    stack = inspect.stack()[1]
    call_location = '{}:{}'.format(stack.filename, stack.lineno)
    camelcase_name = func.__name__
    lowercase_name = camelcase_to_lowercase(camelcase_name)
    logging.warning(
        'From %s: The name %s will be deprecated. Please use %s instead.\n',
        call_location, camelcase_name, lowercase_name)
    return func(*args, **kwargs)

  new_func.__name__ = func.__name__
  new_func.__doc__ = func.__doc__
  new_func.__dict__.update(func.__dict__)
  return new_func
