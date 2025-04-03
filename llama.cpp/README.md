# Setup Guide for `llama.cpp` with CUDA on Nvidia Jetson Nano 4GB

As of April 2025 the current version of llama.cpp can be compiled for the Jetson Nano from 2019 with GPU/CUDA support using `gcc 8.5` and `nvcc 10.2`. A few variants are described here by their build date, and later compared by their performance in benchmarks:

- 2025-04-04 **b5050** Some extra steps had to be included to handle the new support of `bfloat16` in llama.cpp since January 2025. Procedure is described in [this gist](https://github.com/ggml-org/llama.cpp/releases/tag/b4400).
- 2024-12-31 [b4400](https://github.com/ggml-org/llama.cpp/releases/tag/b4400) Following the steps from the [gist](https://github.com/ggml-org/llama.cpp/releases/tag/b4400) above, step 6 can be ommited.
- 2024-02-26 [b2275](https://github.com/ggml-org/llama.cpp/tree/b2275) A [gist by Flor Sanders](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f) from 2024-04-11 describes the procedure to combile a version with GPU acceleration.
- 2023-12-07 [b1618](https://github.com/ggml-org/llama.cpp/tree/b1618) A medium.com article from Anurag Dogra from 2025-03-26 describes the modification needed to compile llama.cpp with `gcc 8.5` and CUDA support.

Parts of the gist are copied here:

- Prerequisites
- Procedure
- Benchmark
- Compile llama.cpp for CPU mode - 24 minutes
- Install prerequisites
- Choosing the right compiler
- Sources

## Prerequisites

You will need the following software packages installed. The section "[Install prerequisites](https://gist.github.com/kreier/6871691130ec3ab907dd2815f9313c5d#install-prerequisites)" describes the process in detail. The installation of `gcc 8.5` and `cmake 3.31` of these might take several hours.

- Nvidia CUDA Compiler nvcc 10.2 - `nvcc --version`
- GCC and CXX (g++) 8.5 - `gcc --version`
- cmake >= 3.14 - `cmake --version`
- `nano`, `curl`, `libcurl4-openssl-dev`, `python3-pip` and `jtop`

## Procedure

To ensure this gist keeps working in the future, while llama.cpp gets new versions, we're cloning the repository and then check out a version known to be working. If you want to try a more recent variant omit the steps `git checkout 3f9da22` and `git checkout -b llamaJetsonNanoCUDA` in the following instructions:

### 1. Clone repository

``` sh
git clone https://github.com/ggml-org/llama.cpp llama5050gpu.cpp
cd llama5050gpu.cpp
git checkout 3f9da22
git checkout -b llamaJetsonNanoCUDA
```

Now we have to make changes to these 6 files before calling cmake to start compiling:

- CMakeLists.txt 14
- ggml/CMakeLists.txt 274
- ggml/src/ggml-cuda/common.cuh 455
- ggml/src/ggml-cuda/fattn-common.cuh 623
- ggml/src/ggml-cuda/fattn-vec-f32.cuh 71
- ggml/src/ggml-cuda/template-instances/../fattn-vec-f16.cuh 73

Early 2025 llama.cpp started supporting and using `bfloat16`, a feature not included in nvcc 10.2. We have two options:

- Option A: Create two new files
    - /usr/local/cuda/include/cuda_bf16.h
    - /usr/local/cuda/include/cuda_bf16.hpp
- Option B: Edit 3 files
    - ggml/src/ggml-cuda/vendors/cuda.h
    - ggml/src/ggml-cuda/convert.cu
    - ggml/src/ggml-cuda/mmv.cu

Details are described in step 2 to 7:

### 2. Add a limit to the CUDA architecture in `CMakeLists.txt`

Edit the file *CMakeLists.txt* with `nano CMakeLists.txt`. Add the following 3 lines after line 14 (with Ctrl + "\_"):

```
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 50 61)
endif()
```


### 3. Add two linker instructions after line 274 in `ggml/CMakeLists.txt`

Edit the file with `nano ggml/CMakeLists.txt` and enter two new lines after `set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")`. It should then look like:

``` h
set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
target_link_libraries(ggml PRIVATE stdc++fs)
add_link_options(-Wl,--copy-dt-needed-entries)
#if (GGML_METAL)
#    set_target_properties(ggml PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-metal.metal")
#endif()
```

### 4. Remove *cpmstexpr* from line 455 in `ggml/src/ggml-cuda/common.cuh`

This feature from CUDA C++ 17 we don't support anyway, just remove the **constexpr** after the *static* in line 455.1 Use `nano ggml/src/ggml-cuda/common.cuh`. After that it looks like:

``` h
// TODO: move to ggml-common.h
static __device__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
```

### 5. Comment lines containing *__buildin_assume* with // to avoid compiler error '"__builtin_assume" is undefined':

- line 623, `nano ggml/src/ggml-cuda/fattn-common.cuh` - 532
- line 71, `nano ggml/src/ggml-cuda/fattn-vec-f32.cuh` - 70
- line 73, `nano ggml/src/ggml-cuda/template-instances/../fattn-vec-f16.cuh`

In January 2025 with version $> 4400$ llama.cpp started including support for bfloat16. There is a standard library `cuda_bf16.h` in the folder `/usr/local/cuda-10.2/targets/aarch64-linux/include` for nvcc 11.0 and larger. With more than 5000 lines one can not simply copy a later version this file into this folder (with its companion `cuda_bf16.hpp` and 3800 lines) and hope it would work. Since it is linked to version 11 or 12, the error messages keep expanding (e.g. `/usr/local/cuda/include/cuda_bf16.h:4322:10: fatal error: nv/target: No such file or directory`). We have two working options

### 6. Option A: Create a `cuda_bf16.h` that redefines `nv_bfloat16` as `half`

Create two new files in the folder `/usr/local/cuda/include/`. The first one is `cuda_bf16.h`, give it the following content:

``` h
#ifndef CUDA_BF16_H
#define CUDA_BF16_H

#include <cuda_fp16.h>

// Define nv_bfloat16 as half
typedef half nv_bfloat16;

#endif // CUDA_BF16_H
```

The second file is `cuda_bf16.hpp` with the content

``` hpp
#ifndef CUDA_BF16_HPP
#define CUDA_BF16_HPP

#include "cuda_bf16.h"

namespace cuda {

    class BFloat16 {
    public:
        nv_bfloat16 value;

        __host__ __device__ BFloat16() : value(0) {}
        __host__ __device__ BFloat16(float f) { value = __float2half(f); }
        __host__ __device__ operator float() const { return __half2float(value); }
    };

} // namespace cuda

#endif // CUDA_BF16_HPP
```


### 6. Option B: Comment all code related to `nv_float16` (*bfloat16`) in 3 files

The second solution is to remove all references of `nv_float16` in the 3 files referencing them. First we have to __NOT__ include the nonexisting `cuda_bf16.h`. Just add two // in front of line 6 with `nano ggml/src/ggml-cuda/vendors/cuda.h`. After that it looks like this:

``` h
#include <cuda.h>
#include <cublas_v2.h>
//#include <cuda_bf16.h>
#include <cuda_fp16.h>
```

That is not enough, the new data type `nv_bfloat16` is referenced in many files. Replace them with `half`

- 684 in `ggml/src/ggml-cuda/convert.cu`
- 60 in `ggml/src/ggml-cuda/mmv.cu`
- 67 in `ggml/src/ggml-cuda/mmv.cu`
- 68 in `ggml/src/ggml-cuda/mmv.cu`
- 235 in `ggml/src/ggml-cuda/mmv.cu` (2x)
- 282 in `ggml/src/ggml-cuda/mmv.cu` (2x)

in `ggml/src/ggml-cuda/convert.cu` there are several instances of bfloat16 use cases:


### 7. Add a flag to avoid the *Target "ggml-cuda" requires the language dialect "CUDA17" (with compiler   extensions).* error

Now it's time to finally call cmake and compile. We add some flags to avoid CUDA17 error messages.



```
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=off
cmake --build build --config Release
```

























## Benchmark

### TinyLlama-1.1B-Chat 2023-12-31

In earlier works and referenced projects we often used the [TinyLlama-1.1B-Chat](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF?show_file_info=tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf) in Q4 quantization from 2023-12-31 with 669 MB in the model file. The test prompt is "The american civil war".

```
./build/bin/llama -hf 
```


llama.cpp has also a build-in benchmark program, here tested with the CUDA version b1618 from December 2023:

| ngl | pp512 | tg128 |
|-----|-------|-------|
| 0   | 17.80 | 2.59  |
| 5   | 20.57 | 3.00  |
| 10  | 24.09 | 2.83  |
| 15  | 31.69 | 3.39  |
| 20  | 39.35 | 3.54  |
| 24  | 55.52 | 3.68  |

Using just the CPU version with a newer llama.cpp b5017 from April 2025 we get a much faster token generation just with the CPU

| ngl | pp512 | tg128 |
|-----|-------|-------|
| 0   |  6.73 | 5.18  |

### Gemma3:1b 2025-03-12

This much more recent [model from March 2025](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF?local-app=llama.cpp) is slightly larger with 806 MB but much more capable than TinyLlama, and comparable in speed. The prompt is "Explain quantum entanglement"

```
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
llama-cli -hf unsloth/gemma-3-1b-it-GGUF:Q4_K_M
```

## Compile llama.cpp for CPU mode

This can be done with `gcc 8.5` or `gcc 9.4` in 24 minutes and was tested with a version as recent as April 2025. You can follow the [instructions from llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). We added the parameter `-DLLAMA_CURL=ON` to support an easy model download from huggingface with the `-hf` command:

``` sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release
```

After finishing the compilation its time for the first model and AI chat:

```
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```



## Install prerequisites


- JetPack 4.6.6 (L4T 32.7.6-20241104234540) - `dpkg-query --show nvidia-l4t-core`

Most of the prerequisites can be installed rather fast. But compiling gcc 8.5.0 will take 3 hours on the Jetson Nano. And the installation of cmake 3.31.5 will also take 45 minutes.

``` sh
sudo apt update
sudo apt install nano curl libcurl4-openssl-dev python3-pip
pip3 install jetson-top
```

### Install `cmake >=3.14`

Purge any old `cmake` installation and install a newer `3.27`

``` sh
sudo apt-get remove --purge cmake
sudo apt-get isntall libssl-dev
wget https://cmake.org/files/v3.27/cmake-3.27.1.tar.gz
tar -xzvf cmake-3.27.1.tar.gz
cd cmake-3.27.1.tar.gz
./bootstrap
make -j4
sudo make install
```


## Choosing the right compiler

### GCC 9.4

This compiler from June 1, 2021 can be easily installed from an apt repository in a few minutes, using

``` sh
sudo apt install build-essential software-properties-common manpages-dev -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install gcc-9 g++-9 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
```

But it is not compatible with `nvcc 10.2` and shows `error: #error -- unsupported GNU version! gcc versions later than 8 are not supported!`. The reasons are found in line 136 of 

> /usr/local/cuda/targets/aarch64-linux/include/crt/host_config.h

``` h
#if defined (__GNUC__)
#if __GNUC__ > 8
#error -- unsupported GNU version! gcc versions later than 8 are not supported!
#endif /* __GNUC__ > 8 */ 
```

### GCC 8.4

This compiler version 8.4 from March 4, 2020 can be installed in the same fast fashion as the mentioned 9.4 above. Just replace three lines:

``` sh
sudo apt install gcc-8 g++-8 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
```

But it throws an error on `llama.cpp/ggml-quants.c` line 407 with:

``` sh
~/llama.cpp/ggml-quants.c: In function ‘ggml_vec_dot_q3_K_q8_K’:
~/llama.cpp/ggml-quants.c:407:27: error: implicit declaration of function ‘vld1q_s8_x4’; did you mean ‘vld1q_s8_x’? [-Werror=implicit-function-declaration]
 #define ggml_vld1q_s8_x4  vld1q_s8_x4
```

It seems that in version 8.4 the ARM NEON intrinsic `vld1q_s8_x4` is treated as a built-in function that cannot be replaced by a macro. It might be related to a fix from [ktkachov on 2020-10-13](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=97349) as one of the [199 bug fixes](https://gcc.gnu.org/bugzilla/buglist.cgi?bug_status=RESOLVED&resolution=FIXED&target_milestone=8.5) leading to 8.5. Let's use the next version:

### GCC 8.5

This version was released May 14, 2021. Unfortunately this version is not yet available for ubuntu 18.04 on `ppa:ubuntu-toolchain-r/test`. We have to compile it by ourselves, and this takes some 3 hours (for the `make -j$(nproc)` step). The steps are:

``` sh
sudo apt-get install -y build-essential software-properties-common
sudo apt-get install -y libgmp-dev libmpfr-dev libmpc-dev
wget http://ftp.gnu.org/gnu/gcc/gcc-8.5.0/gcc-8.5.0.tar.gz
tar -xvzf gcc-8.5.0.tar.gz
cd gcc-8.5.0
./contrib/download_prerequisites
mkdir build && cd build
../configure --enable-languages=c,c++ --disable-multilib
make -j$(nproc)  # Use all CPU cores
sudo make install
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++ 100
```

## Sources

- 2025-03-26 [LLAMA.CPP on NVIDIA Jetson Nano: A Complete Guide](https://medium.com/@anuragdogra2192/llama-cpp-on-nvidia-jetson-nano-a-complete-guide-fb178530bc35), Running LLAMA.cpp on Jetson Nano 4 GB with CUDA 10.2 by Anurag Dogra on medium.com. His modifications compile an older version of llama.cpp with `gcc 8.5` successfully. Because the codebase for llama.cpp is rather old, the performance with GPU support is significantly worse than current versions running purely on the CPU. This motivated to get a more recent llama.cpp version to be compiled. He uses the version [81bc921](https://github.com/ggml-org/llama.cpp/tree/81bc9214a389362010f7a57f4cbc30e5f83a2d28) from December 7, 2023 - [b1618](https://github.com/ggml-org/llama.cpp/tree/b1618) of llama.cpp.
- 2025-01-13 Guide to compile a recent llama.cpp with CUDA support for the Nintendo Switch at [nocoffei.com](https://nocoffei.com/?p=352), titled "Switch AI ✨". The Nintendo Switch 1 has the same Tegra X1 CPU and Maxwell GPU as the Jetson Nano, but 256 CUDA cores instead of just 128, and a higher clock rate. This article was the main source for this gist.
- 2024-04-11 [Setup Guide for `llama.cpp` on Nvidia Jetson Nano 2GB](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f) by Flor Sanders in a gist. He describes the steps to install the `gcc 8.5` compiler on the Jetson. In step 5 he checks out the version [a33e6a0](https://github.com/ggml-org/llama.cpp/commit/a33e6a0d2a66104ea9a906bdbf8a94d050189d91) from February 26, 2024 - [b2275](https://github.com/ggml-org/llama.cpp/tree/b2275)
- 2024-05-04 [Add binary support for Nvidia Jetson Nano- JetPack 4 #4140](https://github.com/ollama/ollama/issues/4140) on issues for ollama. In his initial statement dtischler assumes llama.cpp would require gcc-11, but it actually compiles fine with gcc-8 in version 8.5 from May 14, 2021 as shown in this gist.













## History - content until March 2025

This is a full account of the steps I ran to get `llama.cpp` running on the Nvidia Jetson Nano 2GB. It accumulates multiple different fixes and tutorials, whose contributions are referenced at the bottom of this README.

[Github Gist FlorSanders/JetsonNano2GB_LlamaCpp_SetupGuide.md](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f)

__Remark 2025-01-21:__ This gist is from April 2024. The current version of llama.cpp should be able to compile on the Jetson Nano out of the box. Or you can directly run ollama on the Jetson nano, it just works. But the inference is only done on the CPU, the GPU is not utilized - and probably never will. See [ollama issue 4140](https://github.com/ollama/ollama/issues/4140) regarding JetPack 4, CUDA 10.2 and gcc-11.

Read more in this gist: [kreier/JetsonNano2GB_LlamaCpp_SetupGuide.md](https://gist.github.com/kreier/c64815fd2fd3c15ca9d84ab2cfa58ff9)

## Compile llama.cpp from source

My Jetson Nano Developer Kit A02 from 2019 has 4GB RAM. The [latest software](https://developer.nvidia.com/embedded/downloads) support from Nvidia is Ubuntu 18.04 LTS, support ended in May 2023. On the Nvidia website its the [Jetson Nano Developer Kit SD Card Image](https://developer.nvidia.com/embedded/l4t/r32_release_v7.1/jp_4.6.1_b110_sd_card/jeston_nano/jetson-nano-jp461-sd-card-image.zip) 4.6.1 from 2022/02/23.  It includes the GNU Compiler Collection gcc and g++ 7.5.0 from 2019. You can [compile version 8.5 in about 3 hours](https://kreier.github.io/jetson/#2-llamacpp-as-an-alternative-probably-only-on-cpu-2024-04-11) from source, but you get several error messages when trying to compile llama.cpp. The first step `cmake -B build` works, but the second step `cmake --build build --config Release` exits after 6% with 

```
...
cc1: some warnings being treated as errors
ggml/src/CMakeFiles/ggml-cpu.dir/build.make:134: recipe for target 'ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-quants.c.o' failed
make[2]: *** [ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-quants.c.o] Error 1
CMakeFiles/Makefile2:2181: recipe for target 'ggml/src/CMakeFiles/ggml-cpu.dir/all' failed
make[1]: *** [ggml/src/CMakeFiles/ggml-cpu.dir/all] Error 2
Makefile:145: recipe for target 'all' failed
make: *** [all] Error 2
```

You need at least gcc-9 and can install it with

``` sh
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9 g++-9
```

You later have to add `-DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9` to cmake. The version of cmake provided with ubuntu 18.04 is rather old, only 3.10.2 and you need at least 3.14. Fortunately you can install version 3.31.6 with `sudo snap install cmake --classic`

Now you can follow the [llama.cpp instructions to compile it for CPU](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md):

```
cmake -B build -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9
cmake --build build --config Release
```

Note that we included the gcc version for the build cmake. There are a few warnings, but it continues to build until 100%. Now you can test your build:

## Running llama.cpp

Let's have a look at a few small LLMs we coud run:

- https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
- https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF

We forgot to build it with support for huggingface, so we get an error

```
mk@jetson:~/Downloads/llama.cpp$ ./build/bin/llama-cli -hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
common_get_hf_file: llama.cpp built without libcurl, downloading from Hugging Face not supported.
```

### Download models

We can download it with `wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` and then run it with `./build/bin/llama-cli -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Solar System" --n-gpu-layers 5 --ctx-size 512 --threads 4 --temp 0.7 --top-k 40 --top-p 0.9 --batch-size 16`.

### Compile llama.cpp with huggingface support

A common problem, as [noted in October 2024 on github](https://github.com/ggml-org/llama.cpp/discussions/9835). We have to 

```
$ sudo apt install libcurl4-openssl-dev
```

And then we can compile llama.cpp with the additional flag `-DLLAMA_CURL=ON`

```
cmake -B build -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 -DLLAMA_CURL=ON
cmake --build build --config Release
```

It jumps right away to 20% and needs only 5 minutes to compile. Now we can directly write  

```
llama-cli -hf bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_L
```

There is also an uncensored one:

```
llama-cli -hf mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-uncensored-GGUF:Q4_K_M
```

## Benchmark

The reference model is the small [TinyLlama-1.1B-Chat](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf). You install it with `llama-cli -hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M`.

The way to run the benchmark is `./build/bin/llama-bench -m ../.cache/llama.cpp/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF_tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

### Jetson Nano - memory bandwidth 6 GB/s

```
| model                  |       size | params | backend | threads |  test |         t/s |
| ---------------------- | ---------: | -----: | ------- | ------: | ----: | ----------: |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |       4 | pp512 | 6.71 ± 0.00 |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |       4 | tg128 | 4.98 ± 0.01 |

build: c7b43ab6 (4970)
```

### Intel i7-13700T - memory bandwidth 54 GB/s

For comparison I ran the same model and benchmark on an i7-13700T:

```
| model                  |       size | params | backend | threads |  test |           t/s |
| ---------------------- | ---------: | -----: | ------- | ------: | ----: | ------------: |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |      12 | pp512 | 156.84 ± 8.99 |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |      12 | tg128 |  47.38 ± 0.88 |

build: d5c6309d (4975)
```

### Nvidia RTX 3070 Ti - memory bandwidth [574 GB/s](https://kreier.github.io/benchmark/gpu/)

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3070 Ti, compute capability 8.6, VMM: yes
| model                  |       size | params | backend | ngl |   test |               t/s |
| ---------------------- | ---------: | -----: | ------- | --: | -----: | ----------------: |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CUDA    |  99 |  pp512 | 12830.34 ± 186.18 |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CUDA    |  99 |  tg128 |    325.35 ± 11.50 |

build: f125b8dc (4977)
```

## History

In November 2023 a [bug report #4099 for llama.cpp](https://github.com/ggml-org/llama.cpp/issues/4099) was created for the Jetson Nano. It was closed in March 2024, followed by the [github gist from FlorSanders](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f) explaining the successful steps.
