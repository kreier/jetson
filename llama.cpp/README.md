# Setup Guide for `llama.cpp` on Nvidia Jetson Nano 4GB

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
