# Jetson Nano Developer Kit A02 4GB

I got the [Developer Kit A02](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) (only one CSI camera interface, the B01 has two) of the [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano) in early 2020. In 2021 I added the case, proper power supply and Wifi Card. Early 2024 I started to use it again, but run into limitations very early:

## Ubuntu Distribution limited to 18.04

While some updated images with 20.04 exist, officially Nvidia only supports [18.04 LTS](https://en.wikipedia.org/wiki/Ubuntu_version_history#1804) from the [official website](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write). This version is 7 years old in 2025 and ended support in May 2023. This adds severe software limitations to the already existing hardware limitations:

- Kernel GNU/Linux 4.9.201-tegra
- GNU Compiler Collection 7.5.0 (G++ 7.5.0) from 2019
- NVIDIA Cuda Compiler nvcc 10.3.200
- Jetpack 4.6.1-b110 `sudo apt-cache show nvidia-jetpack`
- Python 3.6.9
- No [OpenCL](https://en.wikipedia.org/wiki/OpenCL) (try fix with PoCL)

## Hardware limitations

The hardware was released 2019, based on the [Maxwell architecture](https://en.wikipedia.org/wiki/Maxwell_(microarchitecture)) of Nvidia from 2014. That's 11 years by 2025.

- Quadcore [Cortex-A57](https://en.wikipedia.org/wiki/ARM_Cortex-A57) at 1428 MHz in [Tegra X1](https://en.wikipedia.org/wiki/Tegra#Tegra_X1)
- 4GB LPDDR4 with 25.60 GB/s bandwidth
- 128 core Maxwell (CC 5.3) GPU with 472 GFLOPS at 921 MHz
- [Cuda CC 5.3](https://www.techpowerup.com/gpu-specs/jetson-nano.c3643)

## Running an LLM on the Jetson

You can install [ollama](https://ollama.com/) on this machine. And it does run llama3.2:1b with 3.77 token/s at 100% CPU. Is it possible to get GPU acceleration? The hardware should be able to, since Cuda CC >= 5.0 is required, and the Jetson has CC 5.3.

### llama.cc as an alternative?

An [article by Flor Sanders](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f) from April 2024 describes the process of running llama.cpp on a 2GB Jetson Nano. With 4GB it should be possible to run a complete llama3.2:1b model file. But you can't use the provided gcc 7.5 compiler, you need at least 8.5 - so you compile it yourself over night.

Since I have a 32GB SDcard and 11 GB free it should be possible to compile GCC 8.5 overnight. But first we have to [add the link to nvcc](https://forums.developer.nvidia.com/t/cuda-nvcc-not-found/118068) the path. in `~/.bashrc` I have to add (with `nano`, that has to be installed too): 

```
$ export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```



### POCL - Portable CL on the Jetson?

An [article on Medium](https://yunusmuhammad007.medium.com/build-and-install-opencl-on-jetson-nano-10bf4a7f0e65) from September 2021 describes the installation of [PoCL](https://github.com/pocl/pocl) 1.7 on the Jetson Nano. By 2024 version 6.0 is the latest one, but it's not supported by PoCL. The reason [is described here](https://largo.lip6.fr/monolithe/admin_pocl/), and related to the old Ubuntu 18.04. That's why the latest version that can be installed is PoCL 3.0. This has been done successfully in October 2022

On "old" Jetson boards (TX2, Xavier NX, AGX Xavier & Nano), it is not possible to install recent PoCL version 5 because the OS is too old (Ubuntu 18.04) and it is complicated to install a recent version of the required Clang compiler (version 17). This is why on these specific boards we will install PoCL version 3. One of the main drawback is that there is no GPU 16-bit float support in this version :-(.

#### Install LLVM

Install the LLVM for ARM64 and Jetson Nano on Ubuntu 18.04 ([source](https://largo.lip6.fr/monolithe/admin_pocl/)):

```
export LLVM_VERSION=10
sudo apt install -y build-essential ocl-icd-libopencl1 cmake git pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} llvm-${LLVM_VERSION} make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils libxml2-dev libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} llvm-${LLVM_VERSION}-dev libncurses5
cd /opt
sudo wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.1.0/clang+llvm-11.1.0-aarch64-linux-gnu.tar.xz
sudo tar -xvvf clang+llvm-11.1.0-aarch64-linux-gnu.tar.xz
sudo mv clang+llvm-11.1.0-aarch64-linux-gnu llvm-11.1.0
sudo rm clang+llvm-11.1.0-aarch64-linux-gnu.tar.xz
```

#### Compile and Install PoCL

```
cd ~/
mkdir softwares
cd softwares
git clone -b release_3_0 https://github.com/pocl/pocl.git pocl_3.0
cd pocl_3.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/pocl-3.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-funroll-loops -march=native" -DCMAKE_C_FLAGS="-funroll-loops -march=native" -DWITH_LLVM_CONFIG=/opt/llvm-11.1.0/bin/llvm-config -DSTATIC_LLVM=ON -DENABLE_CUDA=ON ..
make -j6
sudo make install
sudo mkdir -p /etc/OpenCL/vendors/
sudo touch /etc/OpenCL/vendors/pocl.icd
echo "/opt/pocl-3.0/lib/libpocl.so" | sudo tee --append /etc/OpenCL/vendors/pocl.icd
```

Now OpenCL should be successfully installed on the system. You can check if it works with the following command:

```
clinfo
```

My result:

#### Run `clpeak` Benchmark

```
cd ~/
mkdir workspace
cd workspace
git clone https://github.com/krrishnarraj/clpeak.git
cd clpeak
git submodule update --init --recursive --remote
mkdir build
cd build
cmake ..
make -j6
./clpeak
```
