#include "test_util.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include <stdio.h>
#include <limits.h>
#include <iostream>

bool checkResultsCUB(float& mean, float &var, float& mean2, float& var2, float rel_tol) {
    printf("mean = %f, var = %f and mean_ref = %f and var_ref = %f \n", mean, var, mean2, var2);

    if (abs(mean-mean2)/mean > rel_tol|| abs(var-var2)/var > rel_tol) {
    printf("Error solutions don't match \n");
    return false;
    }
  
  return true;
}


struct point {
    float M;
    float T;
    float N;
};

struct CustomSum
{
    __device__ __forceinline__
    point operator()(const point &a, const point &b) const {
        point res{0, a.T + b.T, a.N + b.N};
        return res;
    }
};


void cubSumAlgorithm(const std::vector<float>& input0, float& sumOut){
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    int N = input0.size();
    std::vector<point> input(N);
    for (int i = 0; i < N;++i){
        input.at(i) = point{0, input0.at(i), 1};
    }
    // Allocate device memory for input
    point* d_input;
    cudaMalloc((void**)&d_input, sizeof(point) * N);

    // Copy input data from host to device
    cudaMemcpy(d_input, input.data(), sizeof(point) * N,cudaMemcpyHostToDevice);

    // Compute the mean
    CustomSum sum_op;
    point init{0, 0, 0}; 
    point sum;
    point dinit;
    
    point* dsum ;
    cudaMalloc((void**)&dsum, sizeof(point) );
    cudaMalloc((void**)&dinit, sizeof(point) );
    // Copy input data from host to device
    cudaMemcpy(&dinit, &init, sizeof(point) * 1, cudaMemcpyHostToDevice);

    // output on host side
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    float* dout ;
    cudaMalloc((void**)&dout, sizeof(float)*1 );

    gpu_timer.Start();
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, sum_op, init));

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, sum_op, init));

    cudaDeviceSynchronize();
    cudaMemcpy(&sum, dsum, sizeof(point) * 1, cudaMemcpyDeviceToHost);
    sumOut = sum.T;

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


int cubCustomSum() {
    
    const int N = 1024*1024*2;
    std::vector<float> input(N);// = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float sum = 0;
    for (int i = 0; i < N ; i++) {
        input[i] = 1.0f + 100*(float)rand()/(float)RAND_MAX;
        input[i] = static_cast<float>(i%2);
        sum+= input[i];
    }

    float sumRes = 0;

    cubSumAlgorithm(input, sumRes);
    

    printf("Mean: %f\n", sumRes);
    std::cout << "Sum: " << sum << std::endl;

    return 0;
}