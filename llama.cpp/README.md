# Setup Guide for `llama.cpp` on Nvidia Jetson Nano 4GB

This is a full account of the steps I ran to get `llama.cpp` running on the Nvidia Jetson Nano 2GB. It accumulates multiple different fixes and tutorials, whose contributions are referenced at the bottom of this README.

[Github Gist FlorSanders/JetsonNano2GB_LlamaCpp_SetupGuide.md](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f)

__Remark 2025-01-21:__ This gist is from April 2024. The current version of llama.cpp should be able to compile on the Jetson Nano out of the box. Or you can directly run ollama on the Jetson nano, it just works. But the inference is only done on the CPU, the GPU is not utilized - and probably never will. See [ollama issue 4140](https://github.com/ollama/ollama/issues/4140) regarding JetPack 4, CUDA 10.2 and gcc-11.

Read more in this gist: [kreier/JetsonNano2GB_LlamaCpp_SetupGuide.md](https://gist.github.com/kreier/c64815fd2fd3c15ca9d84ab2cfa58ff9)

## Compile llama.cpp from source

My Jetson Nano Developer Kit A02 from 2019 has 4GB RAM. The latest software support from Nvidia is Ubuntu 18.04 LTS, support ended in May 2023. It includes the GNU Compiler Collection gcc and g++ 7.5.0 from 2019. You can [compile version 8.5 in about 3 hours](https://kreier.github.io/jetson/#2-llamacpp-as-an-alternative-probably-only-on-cpu-2024-04-11) from source, but you get several error messages when trying to compile llama.cpp. The first step `cmake -B build` works, but the second step `cmake --build build --config Release` exits after 6% with 

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

