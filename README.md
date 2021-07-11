# CUDA-Benchmarking

## Motivation ##
This is a project in which I tried to implement comparisons between the different flavours of Matrix-Matrix products, as used in feed-forward and backpropagation.
`Cuda` and `C` - examples are compiled and then executed via `Ctypes` in Python - since that is my language of choice.
Cuda examples were borrowed and modified from [Nvidia's toolkit webpage](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) to suit my needs.<br>
The following techniques are compared against each other:

- numpy (calls optimized C library)
- c sequential (my sequential c implementation)
- Cuda Shared Memory approach (smaller blocks are stored in faster L1-cache and calculated one after the other)
- CudaBlas (used by modern ML libraries such as `Pytorch` and `Tensorflow`)

## Method ##
Two matrices `A` of incresing size (`m - 2^5 to 2^16`, `n - 2^4 to 2^15`) as well as `B`, the transposed matrix of same size as `A`are multiplied with one another.
If the calculation is > `0.05 seconds` the method is being disqualified.


## Results ##
<img src="https://i.ibb.co/FgyG522/cross-comparison.jpg" width="500"/>



###Todo###
- testing on Jetson Nano device
- can we improve the speed though increasing the thread-block size?
- what role play the VRAM sizes Jetson Nano (2GB) vs. 1080Ti(11GB)?
