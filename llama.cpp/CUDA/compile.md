# Compile llama.cpp

Make sure you have `sudo apt install libcurl4-openssl-dev` done before for libcurl to download directly from huggingface.

```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CURL=ON
cmake --build build --config Release
```

Testrun with [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF?show_file_info=tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)

```
./build/bin/llama-cli -hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M
```

The 636.18 MiB gguf file should be located in ~/.cache/llama.cpp/filename. Run the benchmark:

```
./build/bin/llama-bench -m ../.cache/llama.cpp/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF_tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

The expected result is something like

| model                  |       size | params | backend | threads |  test |            t/s |
| ---------------------- | ---------: | -----: | ------- | ------: | ----: | -------------: |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |       4 | pp512 | 102.88 ± 19.54 |
| llama 1B Q4_K - Medium | 636.18 MiB | 1.10 B | CPU     |       4 | tg128 |   40.96 ± 0.49 |

> build: f125b8dc (4977)

## Compile with CUDA support

Usually it should work out of the box if you have `nvcc` installed. Try this one

```
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release
```

This does not pick up the correct gcc and nvcc compiler. Let's better specify it:

```
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.2/bin/nvcc
```

And we get the error message `nvcc fatal   : Unsupported gpu architecture 'compute_80'` since 10.2 only knows about CUDA Compute Capability up to 7.5 (Turing (2018)) and not beyond, like 8.6 Ampere (2020).



## Compile with gcc-9 compiler

First install gcc-9. Not sure if it work this way with gcc-8, since the compilation took 3 hours on the Jetson Nano.

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9 g++-9
```

Now compile with the new compiler:

```
cmake -B build -DCMAKE_C_COMPILER=/usr/bin/gcc-9 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9
cmake --build build --config Release
```

It actually worked and was reasonably fast!
