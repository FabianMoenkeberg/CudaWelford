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

struct WelfordOp
{
    __device__ __forceinline__
    point operator()(const point &a, const point &b) const {
        if (a.N == 0){
            return b;
        }
        float diff = (a.N/b.N*b.T-a.T);
        // printf(" %f, %f, %f. \n", a.M, a.T, a.N);
        // printf(" %f, %f, %f. => %f \n", b.M, b.T, b.N, diff);
        // point res{a.M + b.M + a.N*diff*diff/((a.N+b.N)*b.N), a.T + b.T, a.N + b.N};
        point res{b.M + a.M + b.N*diff*diff/((b.N+a.N)*a.N), b.T + a.T, a.N + b.N};
        // printf(" %f, %f, %f. end\n", res.M, res.T, res.N);
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
    // Determine temporary storage size with nullptr
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, wel_op, init));

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run Reduction
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, wel_op, init));

    cudaDeviceSynchronize();
    cudaMemcpy(&sum, dsum, sizeof(point) * 1, cudaMemcpyDeviceToHost);
    printf(" %f, %f, %f. \n", sum.M, sum.T,sum.N);
    sumOut = sum.T;
    Nout = sum.N;
    varNOut = sum.M;

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

int cubWelfordReduceMultiCall() {
    
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
