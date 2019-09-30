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
SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
PY3_VER ?= $(shell python3 -c "import sys;print('%d%d' % sys.version_info[:2])")
# Allowed CPU values: k8, armv7a, aarch64
CPU ?= k8
# Allowed COMPILATION_MODE values: opt, dbg
COMPILATION_MODE ?= opt

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS := --crosstool_top=@crosstool//:toolchains \
                     --compilation_mode=$(COMPILATION_MODE) \
                     --copt=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
                     --verbose_failures \
                     --sandbox_debug \
                     --subcommands \
                     --define PY3_VER=$(PY3_VER) \
                     --linkopt=-L$(MAKEFILE_DIR)/libedgetpu/direct/$(CPU) \
                     --linkopt=-l:libedgetpu.so.1.0 \
                     --compiler=gcc \
                     --cpu=$(CPU)

ifeq ("$(COMPILATION_MODE)", "opt")
BAZEL_BUILD_FLAGS += --linkopt=-Wl,--strip-all
endif

ifeq ("$(CPU)", "k8")
BAZEL_BUILD_FLAGS += --copt=-includeglibc_compat.h
SWIG_WRAPPER_BASENAME := _edgetpu_cpp_wrapper.cpython-$(PY3_VER)m-x86_64-linux-gnu
else ifeq ("$(CPU)", "aarch64")
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
SWIG_WRAPPER_BASENAME := _edgetpu_cpp_wrapper.cpython-$(PY3_VER)m-aarch64-linux-gnu
else ifeq ("$(CPU)", "armv7a")
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
SWIG_WRAPPER_BASENAME := _edgetpu_cpp_wrapper.cpython-$(PY3_VER)m-arm-linux-gnueabihf
else
$(error Unknown value $(CPU) of CPU variable)
endif

# $(1): pattern, $(2) destination directory
define copy_out_files =
	for f in `find $(BAZEL_OUT_DIR) -name $(1) -type f -printf '%P\n'`; do \
		mkdir -p $(2)/`dirname $$f`; \
		cp -f $(BAZEL_OUT_DIR)/$$f $(2)/$$f; \
	done
endef

SWIG_OUT_DIR        := $(MAKEFILE_DIR)/edgetpu/swig
EXAMPLES_OUT_DIR    := $(MAKEFILE_DIR)/out/$(CPU)/examples
TOOLS_OUT_DIR       := $(MAKEFILE_DIR)/out/$(CPU)/tools
TESTS_OUT_DIR       := $(MAKEFILE_DIR)/out/$(CPU)/tests
BENCHMARKS_OUT_DIR  := $(MAKEFILE_DIR)/out/$(CPU)/benchmarks

.PHONY: all \
        tests \
        benchmarks \
        tools \
        examples \
        swig \
        clean \
        deb \
        deb-armhf \
        deb-arm64 \
        wheel \
        help

all: tests benchmarks tools examples swig

tests:
	bazel build $(BAZEL_BUILD_FLAGS) $(shell bazel query 'kind(cc_.*test, //src/cpp/...)')
	$(call copy_out_files,"*_test",$(TESTS_OUT_DIR))

benchmarks:
	bazel build $(BAZEL_BUILD_FLAGS) $(shell bazel query 'kind(cc_binary, //src/cpp/...)' | grep benchmark)
	$(call copy_out_files,"*_benchmark",$(BENCHMARKS_OUT_DIR))

tools:
	bazel build $(BAZEL_BUILD_FLAGS) //src/cpp/tools:join_tflite_models \
	                                 //src/cpp/tools:multiple_tpus_performance_analysis
	mkdir -p $(TOOLS_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/src/cpp/tools/join_tflite_models \
	      $(BAZEL_OUT_DIR)/src/cpp/tools/multiple_tpus_performance_analysis \
	      $(TOOLS_OUT_DIR)

examples:
	bazel build $(BAZEL_BUILD_FLAGS) //src/cpp/examples:two_models_one_tpu \
	                                 //src/cpp/examples:two_models_two_tpus_threaded \
	                                 //src/cpp/examples:classify_image \
	                                 //src/cpp/examples:object_detection
	mkdir -p $(EXAMPLES_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/src/cpp/examples/two_models_one_tpu \
	      $(BAZEL_OUT_DIR)/src/cpp/examples/two_models_two_tpus_threaded \
	      $(BAZEL_OUT_DIR)/src/cpp/examples/classify_image \
	      $(BAZEL_OUT_DIR)/src/cpp/examples/object_detection \
	      $(EXAMPLES_OUT_DIR)

swig:
	bazel build $(BAZEL_BUILD_FLAGS) //src/cpp/swig:all
	mkdir -p $(SWIG_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/src/cpp/swig/_edgetpu_cpp_wrapper.so $(SWIG_OUT_DIR)/$(SWIG_WRAPPER_BASENAME).so
	cp -f $(BAZEL_OUT_DIR)/src/cpp/swig/*.py $(SWIG_OUT_DIR)/edgetpu_cpp_wrapper.py

clean:
	rm -rf $(MAKEFILE_DIR)/bazel-* \
	       $(MAKEFILE_DIR)/build \
	       $(MAKEFILE_DIR)/dist \
	       $(MAKEFILE_DIR)/edgetpu/swig/*.so \
	       $(MAKEFILE_DIR)/edgetpu/swig/edgetpu_cpp_wrapper.py \
	       $(MAKEFILE_DIR)/out

DOCKER_WORKSPACE=$(MAKEFILE_DIR)
DOCKER_CPUS=k8 armv7a aarch64
DOCKER_TAG_BASE=coral-edgetpu
include $(MAKEFILE_DIR)/docker/docker.mk

deb:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b

deb-armhf:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a armhf -d

deb-arm64:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a arm64 -d

wheel:
	python3 $(MAKEFILE_DIR)/setup.py bdist_wheel -d $(MAKEFILE_DIR)/dist

help:
	@echo "make all               - Build everything"
	@echo "make tests             - Build all tests"
	@echo "make benchmarks        - Build all benchmarks"
	@echo "make tools             - Build all tools"
	@echo "make examples          - Build all examples"
	@echo "make swig              - Build python SWIG wrapper"
	@echo "make clean             - Remove generated files"
	@echo "make deb               - Build Debian package for amd64"
	@echo "make deb-armhf         - Build Debian package for armhf"
	@echo "make deb-arm64         - Build Debian package for arm64"
	@echo "make wheel             - Build Python wheel"
	@echo "make help              - Print help message"

# Debugging util, print variable names. For example, `make print-ROOT_DIR`.
print-%:
	@echo $* = $($*)
