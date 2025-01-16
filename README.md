# `bitsandbytes`

[![Downloads](https://static.pepy.tech/badge/bitsandbytes)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/month)](https://pepy.tech/project/bitsandbytes) [![Downloads](https://static.pepy.tech/badge/bitsandbytes/week)](https://pepy.tech/project/bitsandbytes)

The `bitsandbytes` library is a lightweight Python wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and 8 & 4-bit quantization functions.

The library includes quantization primitives for 8-bit & 4-bit operations, through `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` and 8-bit optimizers through `bitsandbytes.optim` module.

This fork is actively developed for ROCm and updates are being pushed into `multi-backend-refactor` branch of upstream bitsandbytes. Users can use either of these to run bitsandbytes on AMD GPUs.

**Note: The default branch of this fork is switched from `rocm_enabled` to `rocm_enabled_multi_backend`. This is synced periodically with `multi-backend-refactor` branch of upstream, and latest developments are pushed here until upstream branch is merged into `main`.**

**Installation for ROCm:**

For latest develop version:
```bash
git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled_multi_backend
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -S . #Use -DBNB_ROCM_ARCH="gfx90a;gfx942" to target specific gpu arch
make
pip install .
```

**(Experimental) Build on windows tips for ROCm :**

1, HIPSDK 6.2.4 
2, Ninja, Cmake,
3, add rocm/6.2/bin in the PATH
4, `git clone https://github.com/likelovewant/bitsandbytes` , `git checkout rocm_enabled_multi_backend` or edit those changes on this repo into other fork or upstream `multibackend`.
5, edit few lines in  csrc/ops.hip, csrc/ops_hip.cuh,include/Algo-Direct-Common.h (by comparint the difference on this repo and upstream) and grab CMakeLists.txt from this repo.

Build use .
```
cmake -G "Ninja" -DCOMPUTE_BACKEND=hip -S .

ninja

```
Change the arches in cmake file in line 210 as needed . eg, gfx1100 ,gfx1102..., 

Option `CMakeLists2.txt` , rename `CMakeLists2.txt` to `CMakeLists.txt`

Build use.

```
cmake -G "Ninja" -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1100" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -funroll-loops" -S .
ninja

```


Build wheel , place builded `libbitsandbytes_rocm_nohipblaslt.dll` into `bitsandbytes\bitsandbytes`

```
python setup.py bdist_wheel

```


if you can't install it by pip , then unzip the wheel and place into pip install directory .
Cureently , this is no torch for rocm available at windows. Not sure ,how to use it . 
Note: amd don't support NF4 OR 4 BITS currently ,if you want enable hipblaslt , by comment the line 248 in cmakelists, `add_definitions(-DLEGACY_HIPBLAS_DIRECT=0)` or delete it .fix the conflicts bug in `hipcomon and hipblas`. (bugs will show when you build .) or keep use the no hipblaslt version.

**For more details, please head to the official documentation page:**

**[https://huggingface.co/docs/bitsandbytes/main](https://huggingface.co/docs/bitsandbytes/main)**

## License

`bitsandbytes` is MIT licensed.

We thank Fabio Cannizzo for his work on [FastBinarySearch](https://github.com/fabiocannizzo/FastBinarySearch) which we use for CPU quantization.
