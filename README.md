# CUDA-Benchmarking

## Motivation ##
This is a project in which I tried to implement comparisons between the different flavours of Matrix-Matrix products, as used in feed-forward and backpropagation.
`Cuda` and `C` - examples are compiled and then executed via `Ctypes` in Python - since that is my language of choice.
Cuda examples were borrowed and modified from [Nvidia's toolkit webpage](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) to suit my needs.<br>
The following techniques are compared against each other:

- numpy (calls optimized C library)
- c sequential (my sequential c implementation)
- Cuda Shared Memory approach (smaller blocks are stored in faster L1-cache and calculated one after the other)
- CudaBlas (used by modern ML frameworks such as `Pytorch` and `Tensorflow`)

## Method ##
Two matrices `A` of incresing size (`m - 2^5 to 2^16`, `n - 2^4 to 2^15`) as well as `B`, the transposed matrix of same size as `A`are multiplied with one another.
If the calculation is > `0.05 seconds` the method is being disqualified. Random floats (32bit) were used for testing.


## Results ##

Here the relative differences between Method and stepsize: <br>
As you can see my sequential `C` implementation disqualified pretty quickly as the matrices grow exponential in size. <br>
Numpy is extremely optimized and methods are called from `C` and it disqualifies at a `2^12` where it is 3853x slower than the fastes method - `CudaBlas`<br>
Interestingly, the `CUDA -shared mem` implementation holds up pretty well against `CUDA Blas` and is only about `6x` (results are rounded) slower.<br>
Also interesting - which needs further investigation is how slow `Cuda shared mem` and `Cuda blas` is at smaller matrices. My assumption is the introduced latency of loading the data onto the GPU or the thread-block size of `16*16 threads` causes some issues. This needs further investigation however. <br>
<img src="https://i.ibb.co/FgyG522/cross-comparison.jpg" width="500"/>



### Todo ###
- why is cuda lagging on lower matrix sizes?
- testing on Jetson Nano device
- can we improve the speed though increasing the thread-block size?
- what role play the VRAM sizes Jetson Nano (2GB) vs. 1080Ti(11GB)?
