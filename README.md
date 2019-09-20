# Edge TPU Python API

This repository contains an easy-to-use Python API to work with Coral devices:

* [Dev Board](https://coral.withgoogle.com/products/dev-board/)
* [USB Accelerator](https://coral.withgoogle.com/products/accelerator/)

You can run inference and do transfer learning.

## Build and install from source

1. Sync the source code as per the [Mendel get started guide](
https://coral.googlesource.com/docs/+/refs/heads/master/GettingStarted.md).

1. Run `cd packages/edgetpu/` to switch current working directory

1. Run `scripts/runtime/install.sh` to install Edge TPU runtime

1. Run `scripts/build_swig.sh` to build SWIG-based native layer

1. Run `make wheel` to generate Python wheel and then
   `pip3 install $(ls dist/*.whl)` to install it


## Native C++ code

All native code is inside `src` folder. You can build everything using Makefile.
For example, run `make tests` to build all C++ unit tests or `make benchmarks`
to build all C++ benchmarks. To get the list of all available make targets run
`make help`.

You can cross-compile code for different platforms by setting CPU variable:

```
make CPU=k8      tests  # Builds for x86_64 (default CPU value)
make CPU=armv7a  tests  # Builds for ARMv7-A, e.g. Pi 3 or Pi 4
make CPU=aarch64 tests  # Builds for ARMv8, e.g. Coral Dev Board
```

Docker allows to avoid complicated environment setup and run the same Makefile
targets:
```
make DOCKER_IMAGE=debian:buster DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=tests docker-build
make DOCKER_IMAGE=ubuntu:18.04  DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=tests docker-build
```

All output goes to `out` directory.
