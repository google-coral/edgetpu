#!/bin/bash
. /etc/os-release

[[ "${NAME}" == "Ubuntu" ]] || exit 0

sed -i "s/deb\ /deb \[arch=amd64\]\ /g" /etc/apt/sources.list

cat <<EOT >> /etc/apt/sources.list
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME} main universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME}-updates main universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME}-security main universe
EOT
