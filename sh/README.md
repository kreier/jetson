# Install llama.cpp on the Jetson Nano Development Kit 2019 with 4 GB RAM

Time requirements:

1. Download [image](https://developer.nvidia.com/jetson-nano-sd-card-image) from Nvidia website 6.1 GB, then write image to SD card - **48 minutes**
2.1 Start system, set up machine name, user and password **5 min**
2.2 pdate to 4.9.337 from November 2024 - **30 minutes**
3. Install ollama and the reference model [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF?local-app=ollama)
4. Update the system, including gcc 8.5.0 - **3 hours**
5. Compile llama.cpp b1618 from December 2023 with GPU support

## 1. Download Ubuntu 18.04.6 LTS image from Nvidia and write to SD Card

Get the latest supported image [18.04.6 LTS from the Nvidia Website](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write) (6.11 GByte). Write it to an SD card with at least 16 GB using tools like [rufus](https://rufus.ie/en/) or [balena etcher](https://etcher.balena.io/). Depending on the SD card this could take 48 minutes.



## 2.1 First system start

Plug in the SD card, answer a few questions like expanding the SD card to maximum size, agree EULA and name for machine, user and password. The SSH daemon is started automatically, so after a reboot you can login remotely and continue the configuration. Your system now is:

Software support for 18.04.6 LTS ended in May 2023. Newer kernels like 4.9.337 are provided by Nvidia in November 2024. The downloaded image contains:

- Kernel GNU/Linux 4.9.253-tegra `uname -a`
- GNU Compiler Collection 7.5.0 (G++ 7.5.0) from 2019 `gcc --version`
- NVIDIA Cuda Compiler nvcc 10.3.200 cuda_10.2_r440.TC440_70.29663091_0 `/usr/local/cuda/bin/nvcc --version`
- Python 3.6.9 `python3 --version`
- Jetpack 4.6.1-b110 `sudo apt-cache show nvidia-jetpack`
- 12,529,712 Bytes used `df`

```
mk@nano:~$ sudo apt update
Fetched 41,1 MB in 27s (1.503 kB/s)
Building dependency tree
348 packages can be upgraded. Run 'apt list --upgradable' to see them.
```

## 2.2 Update 

- Kernel GNU/Linux 4.9.337-tegra from November 2024
- GNU Compiler Collection 7.5.0 (G++ 7.5.0) from 2019
- NVIDIA Cuda Compiler nvcc 10.3.200 cuda_10.2_r440.TC440_70.29663091_0 
- Jetpack 4.6.6-b110 sudo apt-cache show nvidia-jetpack
- Python 3.6.9

You can update the kernel to 4.9.337-tegra from November 2024. After the update your system is

- Kernel GNU/Linux 4.9.337-tegra from November 2024
- GNU Compiler Collection 7.5.0 (G++ 7.5.0) from 2019
- NVIDIA Cuda Compiler nvcc 10.3.200
- Jetpack 4.6.6-b110 sudo apt-cache show nvidia-jetpack
- Python 3.6.9

Some good ideas are written at https://github.com/dnovischi/jetson-tutorials/blob/main/jetson-nano-ubuntu-18-04-install.md

## 2.3 Install additional packages

This will take 10 minutes.

```
sudo apt upgrade -y
```


```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo apt install python3-pip nvidia-tensorrt nvidia-jetpack
sudo pip3 install -U jetson-stats

```

- Kernel GNU/Linux 4.9.337-tegra
- JetPack 32.7.6 `dpkg-query --show nvidia-l4t-core`
- Jetpack4.6.6-b24 `apt-cache show nvidia-jetpack`
- 





## 3. Run your first LLM

At this point you can already run your first LLM with pure CPU support on your system. The fastest way is [ollama](https://ollama.com/). You can install it with

```
curl -fsSL https://ollama.com/install.sh | sh
```

Then install and run the first model with

```
ollama run hf.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M --verbose
```

## 4. Update the system

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

## 5. Compile llama.cpp b1618 with GPU support

Let's start with b1618 and gcc 8.5.
