# Edge TPU Python API (DEPRECATED)

**NOTICE:** This Python library is no longer maintained. We recommend you
instead use [PyCoral](https://github.com/google-coral/pycoral).

This library remains available if your project depends on it—either as a [Python
wheel](https://coral.ai/software/#edgetpu-python-api) or a Debian package
(`python3-edgetpu`)—but we will not release any more updates.

If you run `apt-get update python3-edgetpu`, it will print a warning and will
not install the library because it depends on a now-out-dated version of the
libedgetpu package. So we've created a "legacy" version of the Edge TPU Runtime
package that maintains compatibility with the Edge TPU Python library. If you
need to keep using the Edge TPU Python library, refer to the instructions in the
warning message to install the legacy package.


## Build the library

The Edge TPU Python API has a swig-based native layer so an
architecture-specific build is required. Using `build_swig.sh `provides support
for ARM64, ARMv7, and x86_64 for Python 3.5, 3.6, 3.7, and 3.8 (note that only
one wheel is generated supporting all of these).

1. Be sure to edit the `WORKSPACE` file at the root of the repository to specify
a `TENSORFLOW_COMMIT` that matches the version used by your Edge TPU Runtime.

1. Run `scripts/build_swig.sh` to build SWIG-based native layer for different
   Linux architectures. Build is Docker-based, so you need to have it installed.

1. Run `make wheel` to generate Python library wheel and then
   `pip3 install $(ls dist/*.whl)` to install it
