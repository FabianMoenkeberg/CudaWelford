#include "../include/cuda_welford.h"
#include "../include/test_util.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <stdio.h>
#include <limits.h>
#include <iostream>

/// @brief Compare the given results of mean and variance towards a second solution relative to a tolerance rel_tol.
/// @param mean 
/// @param var 
/// @param mean2 
/// @param var2 
/// @param rel_tol 
/// @return 
bool checkResults(float& mean, float &var, float& mean2, float& var2, float rel_tol) {
    printf("mean = %f, var = %f and mean_ref = %f and var_ref = %f \n", mean, var, mean2, var2);

    if (abs(mean-mean2)/mean > rel_tol|| abs(var-var2)/var > rel_tol) {
    printf("Error solutions don't match \n");
    return false;
    }
  
  return true;
}

/// @brief Calculate the statistical values variance and mean in a single pass using Welford algorithm with CPU.
/// @param g_data 
/// @param mean 
/// @param var 
/// @param iStart 
/// @param iEnd 
void computeCpuStatistics(float *g_data, float& mean, float& var, int iStart, int iEnd) {

  float T = 0;
  float T0 = 0;
  float M = 0;
  long N = 0;
  long N0 = 1;
  for (int ix = iStart; ix < iEnd; ++ix) {
    int idx = ix;

    float value = g_data[idx];
    T0 = T;
    T += value;
    N++;
    M += (value - T/N)*(value - T0/N0);
    N0 = N;
  }
  printf("Result %f, %d \n", M, N);
  mean = T/N;
  var = M/(N-1);
}


/// @brief Compute the statistical values variance and mean in the classical two pass method with CPU.
/// @param g_data 
/// @param dimx 
/// @param mean 
/// @param var 
void computeCpuStatisticsClassic(float *g_data, int dimx, float& mean, float& var) {

  float T = 0;
  float T0 = 0;
  float M = 0;
  long N = 0;
  // First pass to calculate the sum and later the mean.
  for (int ix = 0; ix < dimx; ++ix) {
    int idx = ix;

    float value = g_data[idx];
    T += value;
    N++;
  }

  mean = T/N;
  float diff;
  // Second pass to calculate the variance given the mean.
  for (int ix = 0; ix < dimx; ++ix) {
    int idx = ix;

    float value = g_data[idx];
    diff = value - mean;
    M += diff*diff;
  }

  printf("Result Test %f, %d, %f \n", M, N, mean);
  var = M/(N-1);
}

/// @brief Introduce warp reduce block on lowest level with __shfl_down_sync.
/// @param T0in 
/// @param T2in 
/// @param Min 
/// @param M2in 
/// @param warpSize 
/// @param n 
/// @return 
__inline__ __device__ void warpReduceWelford(float& T0in, float& T2in, float& Min, float& M2in, int warpSize, int n) {
  float T0 = T0in;
  float T2 = T2in;
  float M = Min;
  float M2 = M2in;
  float diff;
  // if (n>1024) printf("Warp00 %d: M2 %f, T2 %f, M %f, T %f\n", threadIdx.x, M2, T2, M, T0);
  for (int offset = warpSize/2; offset > 0; offset /= 2){ 
    diff = (T0 - T2);
    M += M2 + diff*diff/(2*n);
    T0 += T2;
    // __syncthreads();
    T2 = __shfl_down_sync(0xffffffff, T0, static_cast<unsigned int>(offset), warpSize);
    M2 = __shfl_down_sync(0xffffffff, M, static_cast<unsigned int>(offset), warpSize);
    n*=2;
  }

  diff = (T0 - T2);
  M += M2 + diff*diff/(2*n);
  T0 += T2;

  T0in = T0;
  Min = M;
}


/// @brief Welford Cuda algorithm using some warp reduce to speed up.
/// Note: try to run her again with just smaller size and different structure. See output and take it as input.
/// @param g_data 
/// @param g_out 
/// @param n0 
/// @param firstRun 
/// @return 
__global__ void kernelWelfordWarp(float *g_data, float *g_out, int n0, bool firstRun) {

    extern __shared__ float sdata[];

    int Nhalf = blockDim.x;
    // int Nhalf0 = Nhalf;
    unsigned int tid = threadIdx.x;

    int idx =  blockIdx.x * (blockDim.x) + tid;
    int idx2 = tid + Nhalf;

    int dT = Nhalf;
    float T, T2;
    float M = 0.0f;
    float M2 = 0.0f;
    
    if (!firstRun){
      M = g_out[idx]; // e.g. M
      M2 = g_out[idx2];
    }

    T = g_data[idx]; // e.g. M
    T2 = g_data[idx2];
    
    int lane = tid % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceWelford(T, T2, M, M2, warpSize, n0);

    if (lane==0) {
      sdata[wid] = M;
      sdata[wid + dT] = T;
    }
    n0 = 2*warpSize;
    
    Nhalf/=n0;
    
    __syncthreads();
    if (tid < Nhalf)
    {
      idx2 = tid + Nhalf;
      M = sdata[tid];
      M2 = sdata[idx2];
      T = sdata[tid+dT];
      T2 = sdata[idx2+dT];
      
      warpReduceWelford(T, T2, M, M2, min(Nhalf, warpSize), n0);
    }

    if (tid == 0){
      g_data[blockIdx.x] = T;//(blockDim.x*2);
      g_out[blockIdx.x] = M;//(blockDim.x*2 - 1);
    }
}


/// @brief Second version Welford Cuda algorithm using shared memory and handling multiple runs.
/// Note: try to run her again with just smaller size and different structure. See output and take it as input
/// @param g_data 
/// @param g_out 
/// @param n0 
/// @param firstRun 
/// @return 
__global__ void kernelWelford_version2(float *g_data, float *g_out, int n0, bool firstRun) {
    extern __shared__ float sdata[];
    int diff = n0;
    int N = 2*blockDim.x;
    
    int Nhalf = N/2;
    int Nhalf0 = Nhalf;
    unsigned int tid = threadIdx.x;

    int idx =  blockIdx.x * (blockDim.x) + tid;
    int idx2 = tid + Nhalf;
    int dT = Nhalf;
    float M, M2, T, T2, T0;
    M = 0;
    M2 = 0;
    if (firstRun){
      sdata[tid] = g_data[idx2]; // e.g. M
      dT = 0;
    }else{
      sdata[tid] = g_out[idx]; // e.g. M
    }
    
    T = g_data[idx]; // e.g. T
    sdata[tid + Nhalf] = T;

    __syncthreads();
    if (!firstRun){
      M = sdata[tid];
      N/=2;
      Nhalf/=2;
      M2 = sdata[tid + Nhalf];
    }
    
    
    while (Nhalf>0){
        idx2 = tid + Nhalf;
        
        if (tid < Nhalf)
        {
          T = sdata[tid+dT];
          T2 = sdata[idx2+dT];
          T0 = (T - T2);
          
          M = M + M2 + T0*T0/(2*diff);
          sdata[tid] = M;
          T += T2;
          dT = Nhalf0;
          sdata[tid+dT] = T;
        }
        diff*=2;
        N /= 2;
        Nhalf = N/2;
        __syncthreads();
        M2 = sdata[tid + Nhalf];
    }
    if (tid == 0){
      g_data[blockIdx.x] = T;//(blockDim.x*2);
      g_out[blockIdx.x] = M;//(blockDim.x*2 - 1);
    }
}

/// @brief Basic Welford Cuda algorithm without optimization.
/// @param g_data 
/// @param g_out 
/// @param dimx 
/// @return 
__global__ void kernelWelford_version1(float *g_data, float *g_out, int dimx) {
    int diff = 1;
    int N = 2*blockDim.x;
    
    int Nhalf = N/2;
    int Nhalf0 = Nhalf;
    int x = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    int idx = (x);
    int idx2 = idx + Nhalf;
    int dT = 0;
    float M, M2, T, T2, T0;
    M = 0;
    M2 = 0;
    
    while (N>0){
        idx2 = idx + Nhalf;
        
        if (threadIdx.x < Nhalf)
        {
          T = g_data[idx+dT];
          T2 = g_data[idx2+dT];
          T0 = (T - T2);       
          
          M = M + M2 + T0*T0/(2*diff);
          g_data[idx] = M;
          T += T2;
          dT = Nhalf0;
          g_data[idx+dT] = T;
        }
        diff*=2;
        N /= 2;
        Nhalf = N/2;
        __syncthreads();
        M2 = g_data[idx+Nhalf];
    }
    if (threadIdx.x == 0){
      g_out[blockIdx.x+gridDim.x] = T;//(blockDim.x*2);
      g_out[blockIdx.x] = M;//(blockDim.x*2 - 1);
    }
}

/// @brief Launch the cuda kernel kernelWelfordWarp until it is finished.
/// @param d_data 
/// @param d_out 
/// @param dimx 
/// @param nBlocks 
/// @param optimization 
void launchKernelWelford(float * d_data, float *d_out, int dimx, int& nBlocks, int optimization) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // int num_sms = prop.multiProcessorCount;
  int blockSize = min(1024, dimx/2);
  nBlocks = dimx/2/blockSize;
  dim3 block(blockSize, 1);
  dim3 grid(nBlocks, 1);

  if (optimization == 2){
    kernelWelfordWarp<<<grid, block, blockSize*2*sizeof(float)>>>(d_data, d_out, 1, true);

    if (nBlocks > 1){
      dimx = nBlocks;
      blockSize = min(1024, dimx/2);

      nBlocks = dimx/2/blockSize;
      block.x = blockSize;
      grid.x = nBlocks;

      kernelWelfordWarp<<<grid, block, blockSize*2*sizeof(float)>>>(d_data, d_out, 2*1024, false);
    }
  }else if(optimization == 1){
    kernelWelford_version2<<<grid, block, blockSize*2*sizeof(float)>>>(d_data, d_out, 1024, false);
  }else{
    kernelWelford_version1<<<grid, block>>>(d_data, d_out, dimx);
  }
}


/// @brief Run Cuda algorithm and measure time for the different optimization versions.
/// @param d_data 
/// @param d_out 
/// @param dimx 
/// @param nBlocks 
/// @param optimization 
/// @return 
float algorithmWelford(float *d_data, float *d_out, int dimx, int& nBlocks, int optimization) {
  float elapsed_time_ms = 0.0f;
  // GpuTimer gpu_timer;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // gpu_timer.Start();

  launchKernelWelford(d_data, d_out, dimx, nBlocks, optimization);
  
  // printf("number of blocks: %d \n", nBlocks);
  cudaEventRecord(stop, 0);
  
  cudaDeviceSynchronize();
  // gpu_timer.Stop();

  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  // elapsed_time_ms =  gpu_timer.ElapsedMillis();
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_time_ms;
}

void calcRemainingVar(float* h_data, float* h_out, int dimx, int nBlocks, float* totvar, float* totmean){
  *totvar = h_out[0];
  *totmean = h_data[0];
  float diff;
  int n = dimx/nBlocks; 
  float m = static_cast<float>(dimx/nBlocks);

  for(int i = 0; i < nBlocks; ++i){
    printf("Results kernel %d: M %f,  T %f, N %d \n", i, h_out[i], h_data[i], dimx/nBlocks);
  }
  for(int i = 1; i < nBlocks; ++i){
    n+= dimx/nBlocks;
    diff = (*totmean - (n-m)*h_data[i]/m);
    // printf("diff: %f, %f, %f, %f\n", diff, h_data[i], totvar, h_data[i] +m*diff*diff/(n-m)/n);
    *totvar += h_out[i] +m*diff*diff/(n-m)/n;
    *totmean+=h_data[i];
  }
}

/// @brief Run the welford algorithm for cuda kernels with different optimization = 0, 1, 2.
/// @param optimization 
/// @return 
int run_welford(int optimization) {
  int dimx = 1024*1024*2;

  int nbytes = dimx * sizeof(float);

  float *d_data = 0, *h_data = 0, *h_out, *h_gold = 0, *d_out = 0;
  cudaMalloc((void **)&d_data, nbytes);
  cudaMalloc((void **)&d_out, nbytes);
  if (0 == d_data) {
    printf("couldn't allocate GPU memory\n");
    return -1;
  }
  printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));
  h_data = (float *)malloc(nbytes);
  h_out = (float *)malloc(nbytes);
  h_gold = (float *)malloc(nbytes);
  if (0 == h_data || 0 == h_gold) {
    printf("couldn't allocate CPU memory\n");
    return -2;
  }
  printf("allocated %.2f MB on CPU\n", 2.0f * nbytes / (1024.f * 1024.f));
  float sum = 0;
  for (int i = 0; i < dimx ; i++) {
    h_gold[i] = 1.0f + 100*(float)rand()/(float)RAND_MAX;
    h_gold[i] = 1.0f + static_cast<float>(i%2);
    sum+= h_gold[i];
  }
  printf("sum vector: %f\n", sum);

  cudaMemcpy(d_data, h_gold, nbytes, cudaMemcpyHostToDevice);
  
  int nBlocks = 0;
  float time = algorithmWelford(d_data, d_out, dimx, nBlocks, optimization);

  cudaMemcpy(h_data, d_data, nbytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost);
  float var0 = 0;
  float mean0 = 0;
  float* totvar = &var0;
  float* totmean = &mean0;
  calcRemainingVar(h_data, h_out, dimx, nBlocks, totvar, totmean);

  float meanCuda = *totmean/dimx;
  float varCuda = *totvar/(dimx-1);
  printf("Runtime Cuda: %f\n", time);
  printf("Mean %f, and Var %f \n", meanCuda, varCuda);
  printf("Verifying solution\n");

//   cudaMemcpy(h_data, d_data, nbytes, cudaMemcpyDeviceToHost);

  float rel_tol = .001;
  float mean = 0;
  float var = 0;
  computeCpuStatistics(h_gold, mean, var, 0, dimx);
  float mean2 = 0;
  float var2 = 0;
  computeCpuStatisticsClassic(h_gold, dimx, mean2, var2);
  bool passCuda = checkResults(meanCuda, varCuda, mean2, var2, rel_tol);
  bool pass = checkResults(mean, var, mean2, var2, rel_tol);

  if (pass && passCuda) {
    printf("Results are correct\n");
  } else {
    printf("FAIL:  results are incorrect\n");
  }  

  // float elapsed_time_ms = 0.0f;
 
//   elapsed_time_ms = timing_experiment(d_data, dimx, dimy, niterations, nreps);
  // printf("A:  %8.2f ms\n", elapsed_time_ms);

  printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

  if (d_data) cudaFree(d_data);
  if (h_data) free(h_data);

  cudaDeviceReset();

  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
