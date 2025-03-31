#!/bin/sh
# This script installs cmake and gcc-8 on the Jetson Nano.

sudo apt update
sudo apt install python3-pip nvidia-tensorrt nano -y
sudo apt install nvidia-jetpack -y
sudo pip3 install -U jetson-stats
sudo apt install build-essential software-properties-common manpages-dev -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install gcc-8 g++-8 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo apt-get install libssl-dev -y
wget https://cmake.org/files/v3.27/cmake-3.27.1.tar.gz
tar -xzvf cmake-3.27.1.tar.gz
cd cmake-3.27.1
./bootstrap
make -j4
sudo make install
gcc --version
g++ --version
cmake --version
