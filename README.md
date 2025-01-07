# Jetson Nano Developer Kit A02 4GB

I got the [Developer Kit A02](https://hshop.vn/may-tinh-ai-nvidia-jetson-nano-developer-kit-b01-upgrade-version-with-2-cameras) (only one CSI camera interface, the B01 has two) of the [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano) in early 2020. In 2021 I added the case, proper power supply and Wifi Card. Early 2024 I started to use it again, but run into limitations very early:

## Ubuntu Distribution limited to 18.04

While some updated images with 20.04 exist, officially Nvidia only supports 18.04. This version is 7 years old in 2025. This adds severe software limitations to the already existing hardware limitations:

- No OpenCL
- NVIDIA Cuda Compiler nvcc 10.3.200
- Python 3.6.9

## Hardware limitations

The hardware was released 2019, based on the [Maxwell architecture](https://en.wikipedia.org/wiki/Maxwell_(microarchitecture)) of Nvidia from 2014. That's 11 years by 2025.

- Quadcore
- 4GB LPDDR4 with 25.60 GB/s bandwidth
- [Cuda CC 5.3](https://www.techpowerup.com/gpu-specs/jetson-nano.c3643)

## Running an LLM on the Jetson

You can install [ollama](https://ollama.com/) on this machine. And it does run llama3.2:1b with 3.77 token/s at 100% CPU. Is it possible to get GPU acceleration? The hardware should be able to, since Cuda CC >= 5.0 is required, and the Jetson has CC 5.3.

### llama.cc as an alternative?

An [article by Flor Sanders](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f) from April 2024 describes the process of running llama.cpp on a 2GB Jetson Nano. With 4GB it should be possible to run a complete llama3.2:1b model file.

Since I have a 32GB SDcard and 11 GB free it should be possible to compile GCC 8.5 overnight. But first we have to [add the link to nvcc](https://forums.developer.nvidia.com/t/cuda-nvcc-not-found/118068) the path. in `~/.bashrc` I have to add (with `nano`, that has to be installed too): 

```
$ export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### POCL - Portable CL on the Jetson?


