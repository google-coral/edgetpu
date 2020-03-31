# Copyright 2020 Google LLC
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
"""
Takes a Windows DLL as input, and generates a new link library
to allow the original DLL to be renamed.
Because this requires dumpbin.exe and lib.exe on the PATH, it's
recommended to run it from the Visual Studio command prompt.

Args:
    input_dll: Path to original DLL
    output_dll: Path to create new DLL
"""
import argparse
import os
import sys
import re
import subprocess

def main(args):
    # Use the DUMPBIN tool to extract exported symbol names.
    dumpbin_out = subprocess.check_output(
        ["dumpbin", "/exports", args.input_dll],
        text=True
    )
    # Functions from dumpbin look like this:
    #           1    0 0003F9C0 ??0EdgeTpuManager@edgetpu@@QEAA@AEBV01@@Z
    matcher = re.compile(r'^\s*\d+\s+\w+\s+\w{8}\s+([^ ]+)')

    # Build a DEF file from the output of DUMPBIN.
    output = 'EXPORTS' + os.linesep
    for line in dumpbin_out.splitlines():
        matches = matcher.search(line)
        if matches:
            fn_name = matches.group(1)
            output += fn_name + os.linesep
    def_file_name = args.output_dll[:-4] + '.def'
    lib_file_name = args.output_dll + '.if.lib'
    exp_file_name = args.output_dll + '.if.exp'
    with open(def_file_name, 'w') as output_def_file:
        output_def_file.write(output)

    # Use the LIB tool to generate a new link library.
    subprocess.check_output(
        [
            'lib',
            '/machine:x64',
            '/def:%s' % os.path.basename(def_file_name),
            '/out:%s' % os.path.basename(lib_file_name)
        ],
        cwd=os.path.dirname(args.output_dll)
    )

    # Move original DLL to new name.
    os.rename(args.input_dll, args.output_dll)

    # Clean up intermediates.
    os.remove(def_file_name)
    os.remove(exp_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dll",
        help="Path to library to rename",
        required=True
    )
    parser.add_argument(
        "--output_dll",
        help="Path to output renamed library",
        required=True
    )
    args = parser.parse_args()
    if not args.output_dll.endswith('.dll') or not args.input_dll.endswith('.dll'):
        print('ERROR: input_dll and output_dll must end with .dll')
        sys.exit(-1)
    main(args)