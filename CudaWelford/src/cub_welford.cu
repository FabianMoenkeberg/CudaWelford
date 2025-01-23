#include "../include/cub_welford.h"
#include "../include/cub_sum.h"

#include "../include/test_util.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include "cub_sum.h"

#include <iostream>
#include <vector>

using namespace cub;
using namespace CubSum;

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

/// @brief Fast way to calculate (a-const)^2 on a single cub transformation call.
/// constant is set in the constructor.
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

/// @brief Classical Multicall Method to calculate the Variance with CUB. 
/// @param input vector from which we calculate variance.
/// @param mean     Resulting mean
/// @param var      Resulting variance
void cubBaseAlgorithm(const std::vector<float>& input, float& mean, float& var){
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    int N = input.size();

    // Allocate device memory for input
    float* d_input;
    cudaMalloc((void**)&d_input, sizeof(float) * N);

    // Copy input data from host to device
    cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    // Initialize mean on device
    
    float* dmean ;
    cudaMalloc((void**)&dmean, sizeof(float) );
    // Copy input data from host to device

    // output on host side
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    float* dout ;
    cudaMalloc((void**)&dout, sizeof(float)*1 );

    gpu_timer.Start();
    // Determine temporary storage size with nullptr
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, dmean, N));

    cudaDeviceSynchronize();

    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);

    cudaDeviceSynchronize();

    // 1. Run Reduction to Calculate Sum to calculate Mean value after.
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, dmean, N);

    cudaDeviceSynchronize();
    // Copy sum from device to host
    cudaMemcpy(&mean, dmean, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    mean /= N;
    // Initialize Subtract and power operator
    SubstractPow subPow_op(mean);

    // 2. Apply Subtraction of mean to each value and take the value to the power of 2.
    CacheModifiedInputIterator<LOAD_LDG,float> cached_iter(d_input);
    TransformInputIterator<float, SubstractPow, CacheModifiedInputIterator<LOAD_LDG, float> > input_iter(cached_iter, subPow_op);
    // cub::TransformInputIterator<float, SubstractPow, float*> input_iter(d_input, subPow_op);

    // Determine temporary storage size with nullptr
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, input_iter, dout, N, Sum(), 0);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // 3. Run Reduction to Calculate Sum of the vector and afterwards mean.
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, input_iter, dout, N, Sum(), 0);

    cudaDeviceSynchronize();
    gpu_timer.Stop();
    elapsed_millis = gpu_timer.ElapsedMillis();

    cudaDeviceSynchronize();
    printf("Run time: %f\n", elapsed_millis);

    var = 0;
    // Copy result from device to host
    cudaMemcpy(&var, dout, sizeof(float) * 1, cudaMemcpyDeviceToHost);

    var/=(N-1);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_temp_storage);
    cudaFree(dout);
    cudaFree(dmean);
}

/// @brief Reduction operator to calculate the variance in a single pass using the Welford algorithm.
/// Similar to CustomSum operator in cub_sum.cu
struct WelfordOp
{
    __device__ __forceinline__
    point operator()(const point &a, const point &b) const {
        if (a.N == 0){
            return b;
        }
        float diff = (a.N/b.N*b.T-a.T);
        point res{b.M + a.M + b.N*diff*diff/((b.N+a.N)*a.N), b.T + a.T, a.N + b.N};
        return res;
    }
};


void cubReduceAlgorithm(const std::vector<float>& input0, float& sumOut, float& Nout, float& varNOut){
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    int N = input0.size();
    std::vector<point> input(N);
    for (int i = 0; i < N;++i){
        input.at(i) = point{0, input0.at(i), 1};
    }
    printf("N is %d.\n", N);
    // Allocate device memory for input
    point* d_input;
    cudaMalloc((void**)&d_input, sizeof(point) * N);

    // Copy input data from host to device
    cudaMemcpy(d_input, input.data(), sizeof(point) * N,cudaMemcpyHostToDevice);

    // Compute the mean
    WelfordOp wel_op;
    point init{0, 0, 0}; 
    point sum;
    point dinit;
    
    point* dsum ;
    cudaMalloc((void**)&dsum, sizeof(point) );
    cudaMalloc((void**)&dinit, sizeof(point) );
    // Copy input data from host to device
    cudaMemcpy(&dinit, &init, sizeof(point) * 1, cudaMemcpyHostToDevice);

    // output on host side
    void            *d_temp_storage = nullptr;
    size_t          temp_storage_bytes = 0;

    float* dout ;
    cudaMalloc((void**)&dout, sizeof(float)*1 );

    gpu_timer.Start();
    // Determine temporary storage size with nullptr and allocate it
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, wel_op, init));

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run Welford Reduction
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, wel_op, init));

    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(&sum, dsum, sizeof(point) * 1, cudaMemcpyDeviceToHost);
    printf(" %f, %f, %f. \n", sum.M, sum.T,sum.N);
    sumOut = sum.T;
    Nout = sum.N;
    varNOut = sum.M;

    // Measure time
    cudaDeviceSynchronize();
    gpu_timer.Stop();
    elapsed_millis = gpu_timer.ElapsedMillis();

    cudaDeviceSynchronize();
    printf("Run time: %f\n", elapsed_millis);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_temp_storage);
    cudaFree(dout);
    cudaFree(dsum);
}

int cubVarianceReduceMultiCall() {
    
    const int N = 1024*1024*2;
    // const int N = 8;
    std::vector<float> input(N);// = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float sum = 0;
    for (int i = 0; i < N ; i++) {
        input[i] = 1.0f + 100*(float)rand()/(float)RAND_MAX;
        input[i] = static_cast<float>(i%2);
        sum+= input[i];
    }

    float mean = 0;
    float var = 0;
    float nOut = 0;

    cubBaseAlgorithm(input, mean, var);

    printf("Mean: %f, Var: %f\n", mean, var);
    std::cout << "Variance: " << mean << std::endl;

    return 0;
}

int cubWelfordReduceSingle() {
    
    const int N = 1024*1024*2;
    // const int N = 8;
    std::vector<float> input(N);// = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float sum = 0;
    for (int i = 0; i < N ; i++) {
        input[i] = 1.0f + 100*(float)rand()/(float)RAND_MAX;
        input[i] = static_cast<float>(i%2);
        sum+= input[i];
    }

    float mean = 0;
    float var = 0;
    float nOut = 0;

    cubReduceAlgorithm(input, mean, nOut, var);


    printf("Mean: %f, Var: %f, pure: %f, %f, %d\n", mean/N, var/N, mean, var, N);
    std::cout << "Variance: " << mean << std::endl;

    return 0;
}
