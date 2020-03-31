ARG IMAGE
FROM ${IMAGE}

COPY update_sources.sh /
RUN /update_sources.sh

RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN apt-get update && apt-get install -y \
  sudo \
  debhelper \
  python \
  python-future \
  python3-all \
  python3-numpy \
  python3-setuptools \
  python3-six \
  python3-wheel \
  libpython3-dev \
  libpython3-dev:armhf \
  libpython3-dev:arm64 \
  build-essential \
  crossbuild-essential-armhf \
  crossbuild-essential-arm64 \
  libusb-1.0-0-dev \
  libusb-1.0-0-dev:arm64 \
  libusb-1.0-0-dev:armhf \
  zlib1g-dev \
  zlib1g-dev:armhf \
  zlib1g-dev:arm64 \
  pkg-config \
  zip \
  unzip \
  curl \
  wget \
  git \
  vim

RUN git clone https://github.com/raspberrypi/tools.git && \
    cd tools && \
    git reset --hard 4a335520900ce55e251ac4f420f52bf0b2ab6b1f

ARG BAZEL_VERSION=2.1.0
RUN wget -O /bazel https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    bash /bazel && \
    rm -f /bazel

