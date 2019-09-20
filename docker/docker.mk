DOCKER_MK_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# Docker
DOCKER_CPUS ?= k8 armv7a armv6 aarch64
DOCKER_TARGETS ?=
DOCKER_IMAGE ?= debian:stretch
DOCKER_IMAGE_NAME = $(word 1, $(subst :, ,$(DOCKER_IMAGE)))
DOCKER_IMAGE_VERSION = $(word 2, $(subst :, ,$(DOCKER_IMAGE)))
DOCKER_TAG_BASE ?= "bazel-cross"
DOCKER_TAG := "$(DOCKER_TAG_BASE)-$(DOCKER_IMAGE_NAME)-$(DOCKER_IMAGE_VERSION)"
DOCKER_SHELL_COMMAND ?= /bin/bash

docker-image:
	docker build -t $(DOCKER_TAG) \
	             --build-arg VERSION=$(DOCKER_IMAGE_VERSION) \
	             -f $(DOCKER_MK_DIR)/Dockerfile.$(DOCKER_IMAGE_NAME) $(DOCKER_MK_DIR)

docker-shell: docker-image
	docker run --rm -it --workdir /workspace -v $(DOCKER_WORKSPACE):/workspace \
	           $(DOCKER_TAG) $(DOCKER_SHELL_COMMAND)

docker-build: docker-image
	docker run --rm -t -v $(DOCKER_WORKSPACE):/workspace $(DOCKER_TAG) /bin/bash -c "\
	    groupadd --gid $(shell id -g) $(shell id -g -n); \
	    useradd -m -e '' -s /bin/bash --gid $(shell id -g) --uid $(shell id -u) $(shell id -u -n); \
	    su - $(shell id -u -n) -c '\
	        for cpu in $(DOCKER_CPUS); do \
	            make CPU=\$${cpu} -C /workspace $(DOCKER_TARGETS) || exit 1; \
	        done \
	    ' \
	"
