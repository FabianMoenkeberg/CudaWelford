# CudaWelford

The goal is to compare different implementations to calculate the variance and the mean of a large vector. The direct approach is a two-pass algorithm, but when using the [Welford algorithm](https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/) we can do it in a single pass.
Three different approaches are used:
* normal variance calculation with CUP
* Welford algorithm with CUP
* Custom Welford algorithm with CUDA.

In the [unit tests](CudaWelford/test/test_cuda.cpp) the different approaches  can be tested. The following resuls can be found on a Nvidia GeForce RTX 2080 SUPER:
* normal variance calculation with CUP (CubVarianceMultistep):		~0.24ms
* Welford algorithm with CUP (CubWelford):							~0.18ms
* Custom Welford algorithm with CUDA (CustomCudaWelford_optimization2):			~0.056ms
* Custom Welford algorithm with CUDA (CustomCudaWelford_optimization1):			~0.063ms
* Custom Welford algorithm with CUDA (CustomCudaWelford_optimization0):			~0.085ms

The custom Welford CUDA algorithm is still 3.2 times faster than the CUP algorithm.
Howerver, the CUP algorithms are way simpler to derive.