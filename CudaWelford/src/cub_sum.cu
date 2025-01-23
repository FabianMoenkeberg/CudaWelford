#include "cub_sum.h"
#include "../include/test_util.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <stdio.h>
#include <limits.h>
#include <iostream>

namespace CubSum{
struct CustomSum
    {
        __device__ __forceinline__
        point operator()(const point &a, const point &b) const {
            point res{0, a.T + b.T, a.N + b.N};

            return res;
        }
    };

bool checkResultsCUB(float& mean, float &var, float& mean2, float& var2, float rel_tol) {
    printf("mean = %f, var = %f and mean_ref = %f and var_ref = %f \n", mean, var, mean2, var2);

    if (abs(mean-mean2)/mean > rel_tol|| abs(var-var2)/var > rel_tol) {
    printf("Error solutions don't match \n");
    return false;
    }
  
  return true;
}

/// @brief Calculate the Sum of a vector using a Custom Summation operation that is similar to the Welford operator
/// @param input0 
/// @param sumOut 
void cubSumAlgorithm(const std::vector<float>& input0, float& sumOut){
    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    int N = input0.size();

    // Assign values to a vector that can be processed by CUB
    std::vector<point> input(N);
    for (int i = 0; i < N;++i){
        input.at(i) = point{0, input0.at(i), 1};
    }

    // Allocate device memory for input
    point* d_input;
    cudaMalloc((void**)&d_input, sizeof(point) * (N));

    // Copy input data from host to device
    cudaMemcpy(d_input, input.data(), sizeof(point) * (N),cudaMemcpyHostToDevice);

    // Initialize values and allocate memory on device
    CustomSum sum_op;
    point init{0, 0, 0}; 
    point sum;
    point dinit;
    
    point* dsum ;
    cudaMalloc((void**)&dsum, sizeof(point) );
    cudaMalloc((void**)&dinit, sizeof(point) );

    // Copy input data from host to device
    cudaMemcpy(&dinit, &init, sizeof(point) * 1, cudaMemcpyHostToDevice);

    // Initialize output on host side and allocate memory on device
    void            *d_temp_storage = nullptr;
    size_t          temp_storage_bytes = 0;

    float* dout ;
    cudaMalloc((void**)&dout, sizeof(float)*1 );

    gpu_timer.Start();
    // Determine temporary storage size with nullptr
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, sum_op, init));

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform reduction with sum_op
    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_input, dsum, N, sum_op, init));

    cudaDeviceSynchronize();

    // Copy result from device to host
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


/// @brief Test function of the custom sum
/// @return 
int cubCustomSum() {
    
    const int N = 1024*1024*2;
    std::vector<float> input(N);// = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // initialize some float values around 1.0
    float sum = 0;
    for (int i = 0; i < N ; i++) {
        input[i] = 1.0f + 100*(float)rand()/(float)RAND_MAX;
        input[i] = static_cast<float>(i%2);
        input[i] = i;
        sum+= input[i];
    }

    float sumRes = 0;

    // Execute the custom cub sum
    cubSumAlgorithm(input, sumRes);
    

    printf("Sum: %f\n", sumRes);
    std::cout << "Sum calculated manually: " << sum << " Difference between sum and cub-sum: " << sum - sumRes << std::endl;

    return 0;
}

}