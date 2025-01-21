# Setup Guide for `llama.cpp` on Nvidia Jetson Nano 4GB

This is a full account of the steps I ran to get `llama.cpp` running on the Nvidia Jetson Nano 2GB. It accumulates multiple different fixes and tutorials, whose contributions are referenced at the bottom of this README.

__Remark 2025-01-21:__ This gist is from April 2024. The current version of llama.cpp should be able to compile on the Jetson Nano out of the box. Or you can directly run ollama on the Jetson nano, it just works. But the inference is only done on the CPU, the GPU is not utilized - and probably never will. See [ollama issue 4140](https://github.com/ollama/ollama/issues/4140) regarding JetPack 4, CUDA 10.2 and gcc-11.

Read more in this gist: [kreier/JetsonNano2GB_LlamaCpp_SetupGuide.md](https://gist.github.com/kreier/c64815fd2fd3c15ca9d84ab2cfa58ff9)
