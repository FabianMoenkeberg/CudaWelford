#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <stdio.h>
#include <limits.h>
#include <iostream>

bool checkResults(float& mean, float &var, float& mean2, float& var2, float rel_tol) {
    printf("mean = %f, var = %f and mean_ref = %f and var_ref = %f \n", mean, var, mean2, var2);

    if (abs(mean-mean2)/mean > rel_tol|| abs(var-var2)/var > rel_tol) {
    printf("Error solutions don't match \n");
    return false;
    }
  
  return true;
}

void computeCpuStatistics(float *g_data, int dimx, float& mean, float& var, int iStart, int iEnd) {

float T = 0;
float T0 = 0;
float M = 0;
long N = 0;
long N0 = 1;
// #pragma omp parallel for
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


void computeCpuStatisticsTest(float *g_data, int dimx, float& mean, float& var) {

float T = 0;
float T0 = 0;
float M = 0;
long N = 0;
// #pragma omp parallel for
    for (int ix = 0; ix < dimx; ++ix) {
      int idx = ix;

      float value = g_data[idx];
      T += value;
      N++;
    }

    mean = T/N;
    float diff;
    for (int ix = 0; ix < dimx; ++ix) {
      int idx = ix;

      float value = g_data[idx];
      // printf("Result %f.\n", value);
      diff = value - mean;
      M += diff*diff;
    }
    
    printf("Result Test %f, %d, %f \n", M, N, mean);
    var = M/(N-1);
}

__inline__ __device__ void warpReduceWelford(float& T0in, float& T2in, float& Min, float& M2in, int warpSize, int n) {
  float T0 = T0in;
  float T2 = T2in;
  float M = Min;
  float M2 = M2in;
  float diff;
  // if (n>1024) printf("Warp00 %d: M2 %f, T2 %f, M %f, T %f\n", threadIdx.x, M2, T2, M, T0);
  for (int offset = warpSize/2; offset > 0; offset /= 2){ 
    diff = (T0 - T2);
    // printf("Diff %d, diff %f, M %f, M2 %f, n %d, %f\n", threadIdx.x, diff, M, M2, n);
    M += M2 + diff*diff/(2*n);
    T0 += T2;
    // printf("Warp0 %d: M2 %f, T2 %f, M %f, T %f\n", threadIdx.x, M2, T2, M, T0);
    // __syncthreads();
    T2 = __shfl_down_sync(0xffffffff, T0, static_cast<unsigned int>(offset), warpSize);
    M2 = __shfl_down_sync(0xffffffff, M, static_cast<unsigned int>(offset), warpSize);
    // printf("Warp %d: M2 %f, T2 %f, M %f, T %f\n", threadIdx.x, M2, T2, M, T0);
    n*=2;
  }

  diff = (T0 - T2);
  M += M2 + diff*diff/(2*n);
  T0 += T2;

  T0in = T0;
  Min = M;
}

// Note: try to run her again with just smaller size and different structure. See output and take it as input
__global__ void kernelWelfordWarp(float *g_data, float *g_out, int n0, bool firstRun) {
 
    extern __shared__ float sdata[];

    int Nhalf = blockDim.x;
    int Nhalf0 = Nhalf;
    unsigned int tid = threadIdx.x;

    int idx =  blockIdx.x * (blockDim.x) + tid;
    int idx2 = tid + Nhalf;
    int dT = Nhalf;
    float T, T2, diffT;
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
    
    warpReduceWelford(T, T2, M, M2, warpSize, 1);
    
    if (lane==0) {
      sdata[wid] = M;
      sdata[wid + dT] = T;
    }
    n0 = 2*warpSize;
    
    Nhalf/=n0;
    
    __syncthreads();
    if (!firstRun){
      M = sdata[tid];  
      M2 = sdata[tid + Nhalf];
    }else{
      M = sdata[tid];
      M2 = sdata[tid + Nhalf];
    }
    
    while (Nhalf>0){
        idx2 = tid + Nhalf;
        
        if (tid < Nhalf)
        {
          // calculateOneWelfordStep(sdata, T, T2, diffT, M, M2);
          T = sdata[tid+dT];
          T2 = sdata[idx2+dT];

          diffT = (T - T2);
          
          M += M2 + diffT*diffT/(2*n0);
          sdata[tid] = M;
          T += T2;
          
          sdata[tid+dT] = T;
        }
        n0*=2;
        Nhalf/=2;
        __syncthreads();
        M2 = sdata[tid + Nhalf];
    }
    if (tid == 0){
      g_data[blockIdx.x] = T;//(blockDim.x*2);
      // printf("Results kernel T: %f, %d, %f, %d \n", T, blockDim.x*2, g_out[blockIdx.x+gridDim.x], blockIdx.x);
      g_out[blockIdx.x] = M;//(blockDim.x*2 - 1);
      // printf("Results kernel M: %f, %d, %f %d \n", M, gridDim.x, g_out[blockIdx.x], blockIdx.x);
    }
}

// Note: try to run her again with just smaller size and different structure. See output and take it as input
__global__ void kernelWelford2B(float *g_data, float *g_out, int n0, bool firstRun) {
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
    // printf("Idx %d, %d, %d, %d\n", idx, dimx, dT, Nhalf);
    // printf("Init %f, %f\n", sdata[tid], sdata[tid + Nhalf]);
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
          // printf("Values %f, %f, %f, %f\n", M, M2, T, T2);
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
      // printf("Results kernel T: %f, %d, %f, %d \n", T, blockDim.x*2, g_out[blockIdx.x+gridDim.x], blockIdx.x);
      g_out[blockIdx.x] = M;//(blockDim.x*2 - 1);
      // printf("Results kernel M: %f, %d, %f %d \n", M, gridDim.x, g_out[blockIdx.x], blockIdx.x);
    }
}

__global__ void kernelWelford(float *g_data, float *g_out, int dimx) {
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
    
    // printf("Block: %d, %d, %d \n", blockIdx.x, idx, idx2);
    while (N>0){
        idx2 = idx + Nhalf;
        
        if (threadIdx.x < Nhalf)
        {
          T = g_data[idx+dT];
          T2 = g_data[idx2+dT];
          T0 = (T - T2);

          // if (blockIdx.x==1){
          //   printf("IDX %f, %f, %d, %d: %f, %f, %f: %f, %f : %f \n", g_data[idx+dT], g_data[idx2+dT], idx+dT, idx2+dT, T, T2, T0, M, M2, M + M2 + T0*T0/(2*diff));
          // }           
          
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
      // printf("Results kernel T: %f, %d, %f, %d \n", T, blockDim.x*2, g_out[blockIdx.x+dimx/2], blockIdx.x);
      g_out[blockIdx.x] = M;//(blockDim.x*2 - 1);
      // printf("Results kernel M: %f, %d, %f %d \n", M, dimx, g_out[blockIdx.x], blockIdx.x);
    }
}


void launchKernelWelford(float * d_data, float *d_out, int dimx, int& nBlocks) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sms = prop.multiProcessorCount;
  int blockSize = min(1024, dimx/2);
  nBlocks = dimx/2/blockSize;
  dim3 block(blockSize, 1);
  dim3 grid(nBlocks, 1);
  // printf("test00 %d", blockSize);
  kernelWelfordWarp<<<grid, block, blockSize*2*sizeof(float)>>>(d_data, d_out, 1, true);
  // printf("test000 %d", nBlocks); 
  dimx = nBlocks;
  blockSize = min(1024, dimx/2);
  // printf("Blocksize: %d\n", blockSize);
  nBlocks = dimx/blockSize/2;
  block.x = blockSize;
  grid.x = nBlocks;

  kernelWelfordWarp<<<grid, block, blockSize*2*sizeof(float)>>>(d_data, d_out, 2*1024, false);
  // kernelWelford2B<<<grid, block, blockSize*2*sizeof(float)>>>(d_data, d_out, 1024, false);


  // kernelWelford<<<grid, block>>>(d_data, d_out, dimx);
}



float algorithmWelford(float *d_data, float *d_out, int dimx, int& nBlocks) {
  float elapsed_time_ms = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // int nBlocks = 0;
  launchKernelWelford(d_data, d_out, dimx, nBlocks);
  
  // printf("number of blocks: %d \n", nBlocks);
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

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

int run_welford() {
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
  float time = algorithmWelford(d_data, d_out, dimx, nBlocks);

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
  computeCpuStatistics(h_gold, dimx, mean, var, 0, dimx);
  float mean2 = 0;
  float var2 = 0;
  computeCpuStatisticsTest(h_gold, dimx, mean2, var2);
  bool passCuda = checkResults(meanCuda, varCuda, mean2, var2, rel_tol);
  bool pass = checkResults(mean, var, mean2, var2, rel_tol);

  if (pass && passCuda) {
    printf("Results are correct\n");
  } else {
    printf("FAIL:  results are incorrect\n");
  }  

  float elapsed_time_ms = 0.0f;
 
//   elapsed_time_ms = timing_experiment(d_data, dimx, dimy, niterations, nreps);
  printf("A:  %8.2f ms\n", elapsed_time_ms);

  printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

  if (d_data) cudaFree(d_data);
  if (h_data) free(h_data);

  cudaDeviceReset();

  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
