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
import argparse
import ctypes
import platform

class EdgeTpuDevice(ctypes.Structure):
  _fields_ = [("type", ctypes.c_int),
              ("path", ctypes.c_char_p)]

EDGETPU_APEX_PCI = 0
EDGETPU_APEX_USB = 1

def edgetpu_type(t):
  if t == EDGETPU_APEX_PCI:
    return 'PCI'
  if t == EDGETPU_APEX_USB:
    return 'USB'
  return 'Unknown'

def edgetpulib_default():
  system = platform.system()
  if system == 'Windows':
    return 'edgetpu.dll'
  if system == 'Darwin':
    return 'libedgetpu.1.dylib'
  if system == 'Linux':
    return 'libedgetpu.so.1'
  raise Exception('This operating system is not supported.')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--edgetpulib',
                      default=edgetpulib_default(),
                      help='Path to edgetpu dynamic library')
  args = parser.parse_args()

  lib = ctypes.pydll.LoadLibrary(args.edgetpulib)
  lib.edgetpu_list_devices.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
  lib.edgetpu_list_devices.restype = ctypes.POINTER(EdgeTpuDevice)

  num_devices = ctypes.c_size_t()
  devices = lib.edgetpu_list_devices(ctypes.byref(num_devices))
  for i in range(num_devices.value):
    print(i, edgetpu_type(devices[i].type), devices[i].path.decode('utf-8'))

if __name__ == '__main__':
  main()
