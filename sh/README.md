# Install llama.cpp on the Jetson Nano Development Kit 2019 with 4 GB RAM

Time requirements:

1. Download [image](https://developer.nvidia.com/jetson-nano-sd-card-image) from Nvidia website 6.1 GB, then write image to SD card - **48 minutes**
2. Start system, set up machine name, user and password **5 min**
3. Update to 4.9.337 from November 2024 - **35 minutes**
4. Install ollama and the reference model [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF?local-app=ollama) - **7 minutes**
5. Install additional software like `cmake` and `gcc-8` - **3 hours**
7. Compile llama.cpp b1618 from December 2023 with GPU support

## 1. Download Ubuntu 18.04.6 LTS image from Nvidia and write to SD Card

Get the latest supported image [18.04.6 LTS from the Nvidia Website](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write) (6.11 GByte). Write it to an SD card with at least 16 GB using tools like [rufus](https://rufus.ie/en/) or [balena etcher](https://etcher.balena.io/). Depending on the SD card this could take 48 minutes.



## 2. First system start

Plug in the SD card, answer a few questions like expanding the SD card to maximum size, agree EULA and name for machine, user and password. The SSH daemon is started automatically, so after a reboot you can login remotely and continue the configuration. Your system now is:

Software support for 18.04.6 LTS ended in May 2023. Newer kernels like 4.9.337 are provided by Nvidia in November 2024. The downloaded image contains:

- Kernel GNU/Linux 4.9.253-tegra `uname -a`
- GNU Compiler Collection 7.5.0 (G++ 7.5.0) from 2019 `gcc --version`
- NVIDIA Cuda Compiler nvcc 10.3.200 cuda_10.2_r440.TC440_70.29663091_0 `/usr/local/cuda/bin/nvcc --version`
- Python 3.6.9 `python3 --version`
- 12,529,712 Bytes used `df`

``` sh
mk@nano:~$ sudo apt update
Fetched 41,1 MB in 27s (1.503 kB/s)
Building dependency tree
348 packages can be upgraded. Run 'apt list --upgradable' to see them.
```



## 3. Update system to kernel 4.9.337 from November 2024

This will take **35 minutes**.

```
sudo apt upgrade -n
```

- Kernel GNU/Linux 4.9.337-tegra from November 2024



## 4. Run your first LLM with ollama

At this point you can already run your first LLM with pure CPU support on your system. The fastest way is [ollama](https://ollama.com/). You can install it with

``` sh
sudo apt install curl libcurl4-openssl-dev
curl -fsSL https://ollama.com/install.sh | sh
>>> Installing ollama to /usr/local
>>> Downloading Linux arm64 bundle
WARNING: Unsupported JetPack version detected.  GPU may not be supported
>>> NVIDIA JetPack ready.
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
```

Then install and run the first model with

```
ollama run hf.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M --verbose
```

And after **7 minutes** its ready for questions like `How many R's are in the word STRAWBERRY?"

> prompt eval rate: 9.04 tokens/s - eval rate: 6.62 tokens/s



## 5. Install additional packages

Install `jetpack`, `cmake` and `gcc-8` with just 

```
curl -fsSL https://kreier.github.io/jetson/sh/step5.sh | sh
```

or add the following sofware packages with these three lines in the next **10 minutes**:

```
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
```

Add the following lines to ~/.bashrc at the end with `nano ~/.bashrc`

```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```


- Kernel GNU/Linux 4.9.337-tegra
- JetPack 32.7.6-20241104234540 [L4T 32.7.6] `dpkg-query --show nvidia-l4t-core`
- Compiler gcc Ubuntu/Linaro 8.4.0-1ubuntu1~18.04) 8.4.0 `gcc --version`
- cmake 3.27.1 `cmake --version`



Some good ideas are written at https://github.com/dnovischi/jetson-tutorials/blob/main/jetson-nano-ubuntu-18-04-install.md

- Jetpack 4.6.1-b110 `sudo apt-cache show nvidia-jetpack`



## 6. Compile llama.cpp for CPU

You need to use gcc-9, with gcc-8 you get errors regarding `llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h:313:27: error: invalid initializer` and in `llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c:8615:40:` regarding `ggml_uint8x16x4_t q6bits = ggml_vld1q_u8_x4(q6); q6 += 64;`. Using gcc-9 all these errors disappear. Installation takes 5 minutes with

``` sh
sudo apt install gcc-9 g++-9 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo apt install libcurl4-openssl-dev
```

After that you can just follow the instructions from the llama.cpp website below. We added the `-DLLAMA_CURL=ON` option to the build process. It allows to download models directly from the huggingface website with the `-hf` option (see example below). The last compilation step takes 60 minutes.

``` sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release
```

Now let's download and run our first model:

``` sh
./build/bin/llama-cli -hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M
```

## 7. Compile with GPU support b

Download the specific version with

``` sh
git clone https://github.com/ggerganov/llama.cpp.git  
cd llama.cpp
git checkout 81bc921
git checkout -b llamaForJetsonNano
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
make -j 2
```

You likely get an error message `"identifier 'CUBLAS_TF32_TENSOR_OP_MATH' not found"` but this can be fized to include in file `llama.cpp/ggml-cuda.cu` above the `#include` statement:

```
#if CUDA_VERSION < 1100
  #define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
  #define CUBLAS_COMPUTE_16F CUDA_R_16F
  #define CUBLAS_COMPUTE_32F CUDA_R_32F
#endif
```

Now again `make -j 2`.




## 95. Compile llama.cpp b1618 with GPU support

Let's start with b1618 and gcc 8.5.




## 99. remnants

Install `cmake` and `gcc-8` with just 

```
curl -fsSL https://kreier.github.io/jetson/sh/step5.sh | sh
```

or step by step with

```
sudo apt update
sudo apt install build-essential software-properties-common manpages-dev -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
gcc --version
g++ --version
```



This part will take 5 hours when executing all the steps outlined below. I wrote a script to run all steps automatically, just enter the following line:

```
sh https://kreier.github.io/jetson/sh/step4.sh
```

After completing the

Install gcc 8.5.0

```
sudo apt update
sudo apt install build-essential software-properties-common manpages-dev -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
gcc --version
g++ --version
```

Install gcc 9.5 is much faster:

```
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9
sudo apt install g++-9
```

Or with alternatives:

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo apt install  gcc-9  g++-9
```
 