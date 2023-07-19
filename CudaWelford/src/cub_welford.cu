#include "test_util.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include <iostream>
#include <vector>

using namespace cub;

struct CustomMin
    {
        template <typename T>
        __device__ __forceinline__
        T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct Pow2
{
    __host__ __device__ __forceinline__
    float operator()(const float &a) const {
        return float(a * a);
    }
};

struct SubstractPow
{
    float constant = 0.0f;

    __host__ __device__ __forceinline__
    SubstractPow(float constant) : constant(constant) {}

    __host__ __device__ __forceinline__
    float operator()(const float &a) const {
        return (a-constant)*(a-constant);
    }
};

void cubBaseAlgorithm(const std::vector<float>& input, float& mean, float& var){
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    int N = input.size();

    // Allocate device memory for input
    float* d_input;
    cudaMalloc((void**)&d_input, sizeof(float) * N);

    // Copy input data from host to device
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    // Compute the mean
    
    float* dmean ;
    cudaMalloc((void**)&dmean, sizeof(float) );
    // Copy input data from host to device

    // output on host side
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    float* dout ;
    cudaMalloc((void**)&dout, sizeof(float)*1 );

    gpu_timer.Start();
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, dmean, N));

    cudaDeviceSynchronize();

    // CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);

    cudaDeviceSynchronize();

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, dmean, N);

    cudaDeviceSynchronize();
    cudaMemcpy(&mean, dmean, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    mean /= N;
    SubstractPow subPow_op(mean);

    CacheModifiedInputIterator<LOAD_LDG,float> cached_iter(d_input);
    TransformInputIterator<float, SubstractPow, CacheModifiedInputIterator<LOAD_LDG, float> > input_iter(cached_iter, subPow_op);
    // cub::TransformInputIterator<float, SubstractPow, float*> input_iter(d_input, subPow_op);

    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, input_iter, dout, N, Sum(), 0);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, input_iter, dout, N, Sum(), 0);

    cudaDeviceSynchronize();
    gpu_timer.Stop();
    elapsed_millis = gpu_timer.ElapsedMillis();

    cudaDeviceSynchronize();
    printf("Run time: %f\n", elapsed_millis);

    var = 0;
    cudaMemcpy(&var, dout, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    var/=(N-1);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_temp_storage);
    cudaFree(dout);
    cudaFree(dmean);
}

int cubWelford() {
    
    const int N = 1024*1024*2;
    std::vector<float> input(N);// = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float sum = 0;
    for (int i = 0; i < N ; i++) {
        input[i] = 1.0f + 100*(float)rand()/(float)RAND_MAX;
        input[i] = static_cast<float>(i%2);
        sum+= input[i];
    }

    float mean = 0;
    float var = 0;

    cubBaseAlgorithm(input, mean, var);
    

    printf("Mean: %f, Var: %f\n", mean, var);
    std::cout << "Variance: " << mean << std::endl;

    return 0;
}
