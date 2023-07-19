#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <stdio.h>
#include <limits.h>
#include <iostream>

bool checkResults(float *gold, float *d_data, int dimx, int dimy, float rel_tol) {
  for (int iy = 0; iy < dimy; ++iy) {
    for (int ix = 0; ix < dimx; ++ix) {
      int idx = iy * dimx + ix;

      float gdata = gold[idx];
      float ddata = d_data[idx];

      if (isnan(gdata) || isnan(ddata)) {
        printf("Nan detected: gold %f, device %f\n", gdata, ddata);
        return false;
      }

      float rdiff;
      if (fabs(gdata) == 0.f)
        rdiff = fabs(ddata);
      else
        rdiff = fabs(gdata - ddata) / fabs(gdata);

      if (rdiff > rel_tol) {
        printf("Error solutions don't match at iy=%d, ix=%d.\n", iy, ix);
        printf("gold: %f, device: %f\n", gdata, ddata);
        printf("rdiff: %f\n", rdiff);
        return false;
      }
    }
  }
  return true;
}

void computeCpuResults(float *g_data, int dimx, int dimy, int niterations,
                       int nreps) {
  for (int r = 0; r < nreps; r++) {
    printf("Rep: %d\n", r);
#pragma omp parallel for
    for (int iy = 0; iy < dimy; ++iy) {
      for (int ix = 0; ix < dimx; ++ix) {
        int idx = iy * dimx + ix;

        float value = g_data[idx];

        for (int i = 0; i < niterations; i++) {
          if (ix % 4 == 0) {
            value += sqrtf(logf(value) + 1.f);
          } else if (ix % 4 == 1) {
            value += sqrtf(cosf(value) + 1.f);
          } else if (ix % 4 == 2) {
            value += sqrtf(sinf(value) + 1.f);
          } else if (ix % 4 == 3) {
            value += sqrtf(tanf(value) + 1.f);
          }
        }
        g_data[idx] = value;
      }
    }
  }
}

__global__ void kernel_A(float *g_data, int dimx, int dimy, int niterations) {
  for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy;
       iy += blockDim.y * gridDim.y) {
    for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < dimx;
         ix += blockDim.x * gridDim.x) {
      int idx = iy * dimx + ix;

      float value = g_data[idx];

      for (int i = 0; i < niterations; i++) {
        if (ix % 4 == 0) {
          value += sqrtf(logf(value) + 1.f);
        } else if (ix % 4 == 1) {
          value += sqrtf(cosf(value) + 1.f);
        } else if (ix % 4 == 2) {
          value += sqrtf(sinf(value) + 1.f);
        } else if (ix % 4 == 3) {
          value += sqrtf(tanf(value) + 1.f);
        }
      }
      g_data[idx] = value;
    }
  }
}

__global__ void kernel_B(float *g_data, int dimx, int dimy, int niterations) {
  int ix = (blockIdx.x * blockDim.x + threadIdx.x)*4;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix < dimx-3 && iy <dimy){
    int idx = iy * dimx + ix;
    
    float value = g_data[idx];
    float value1 = g_data[idx+1];
    float value2 = g_data[idx+2];
    float value3 = g_data[idx+3];
    for (int i = 0; i < niterations; i++) {
      value += sqrtf(logf(value) + 1.f);
      value1 += sqrtf(cosf(value1) + 1.f);
      value2 += sqrtf(sinf(value2) + 1.f);
      value3 += sqrtf(tanf(value3) + 1.f);
    }
    g_data[idx] = value;
    g_data[idx+1] = value1;
    g_data[idx+2] = value2;
    g_data[idx+3] = value3;
  }else if(iy < dimy){
    for (int ix0 = ix; ix0 < dimx; ++ix0){
      int idx = iy * dimx + ix0;
      float value = g_data[idx];

      if (ix0 % 4 == 0) {
        for (int i = 0; i < niterations; i++) {
          value += sqrtf(logf(value) + 1.f);
        }
      }else if (ix0 % 4 == 1) {
          for (int i = 0; i < niterations; i++) {
            value += sqrtf(cosf(value) + 1.f);
          }
      }else if (ix0 % 4 == 2) {
          for (int i = 0; i < niterations; i++) {
            value += sqrtf(sinf(value) + 1.f);
          }
      }else{
          for (int i = 0; i < niterations; i++) {
            value += sqrtf(tanf(value) + 1.f);
          }
      }
      g_data[idx] = value;
    }
  }
}

// Note that this is not working if dimx != 4*k, because the cast messes up the indices after the first line!
__global__ void kernel_B2(float *g_data, int dimx, int dimy, int niterations) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix*4 < dimx-3 && iy <dimy){
    int idx = iy * dimx/4 + ix;
    
    float4 values = reinterpret_cast<float4*>(g_data)[idx];

    for (int i = 0; i < niterations; i++) {
      values.x += sqrtf(logf(values.x) + 1.f);
      values.y += sqrtf(cosf(values.y) + 1.f);
      values.z += sqrtf(sinf(values.z) + 1.f);
      values.w += sqrtf(tanf(values.w) + 1.f);
    }

    reinterpret_cast<float4*>(g_data)[idx] = values;
  }
}

__global__ void kernel_test(float *g_data, int dimx, int dimy, int niterations) {
  // Note I just apply a element-wise operation on each element, NOT DIFFERENT on x-axis values.
  int ix = (blockIdx.x * blockDim.x + threadIdx.x)*4 ;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix < dimx && iy <dimy){
    int idx = iy * dimx + ix;
    float value = g_data[idx];
    for (int i = 0; i < niterations; i++) {
      value += sqrtf(logf(value) + 1.f);
    }
    g_data[idx] = value;
  }
}

// 1.26ms
void launchKernel_test(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  int maxThreadsPerBlock;
  cudaGetDeviceProperties(&prop, 0);
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  int nBlockX = 16;
  int nBlockY = 16;        
  dim3 block(nBlockX, nBlockY);
  dim3 grid((dimx + nBlockX - 1)/nBlockX, (dimy + nBlockY - 1)/nBlockY);

  kernel_test<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

void launchKernel(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int num_sms = prop.multiProcessorCount;

  dim3 block(1, 32);
  dim3 grid(1, num_sms);
  kernel_A<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

// 2.07/2.09 ms
void launchKernelB(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  int maxThreadsPerBlock;
  cudaGetDeviceProperties(&prop, 0);
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  int nBlockX = 16;
  int nBlockY = 16;        
  dim3 block(nBlockX, nBlockY);
  dim3 grid(((dimx+4-1)/4 + nBlockX - 1)/nBlockX, (dimy + nBlockY - 1)/nBlockY);
  kernel_B2<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

float timing_experiment(float *d_data,
                        int dimx, int dimy, int niterations, int nreps) {
  float elapsed_time_ms = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  for (int i = 0; i < nreps; i++) {
    launchKernelB(d_data, dimx, dimy, niterations);
  }
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  elapsed_time_ms /= nreps;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_time_ms;
}

int cuda_prog_final() {
  int dimx = 8 * 1024;
  int dimy = 8 * 1024;

  int nreps = 10;
  int niterations = 5;

  int nbytes = dimx * dimy * sizeof(float);

  float *d_data = 0, *h_data = 0, *h_gold = 0;
  cudaMalloc((void **)&d_data, nbytes);
  if (0 == d_data) {
    printf("couldn't allocate GPU memory\n");
    return -1;
  }
  printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));
  h_data = (float *)malloc(nbytes);
  h_gold = (float *)malloc(nbytes);
  if (0 == h_data || 0 == h_gold) {
    printf("couldn't allocate CPU memory\n");
    return -2;
  }
  printf("allocated %.2f MB on CPU\n", 2.0f * nbytes / (1024.f * 1024.f));
  for (int i = 0; i < dimx * dimy; i++) h_gold[i] = 1.0f + 0.01*(float)rand()/(float)RAND_MAX;
  cudaMemcpy(d_data, h_gold, nbytes, cudaMemcpyHostToDevice);

  timing_experiment(d_data, dimx, dimy, niterations, 1);
  printf("Verifying solution\n");

  cudaMemcpy(h_data, d_data, nbytes, cudaMemcpyDeviceToHost);

  float rel_tol = .001;
  computeCpuResults(h_gold, dimx, dimy, niterations, 1);
  bool pass = checkResults(h_gold, h_data, dimx, dimy, rel_tol);

  if (pass) {
    printf("Results are correct\n");
  } else {
    printf("FAIL:  results are incorrect\n");
  }  

  float elapsed_time_ms = 0.0f;
 
  elapsed_time_ms = timing_experiment(d_data, dimx, dimy, niterations,
                                      nreps);
  printf("A:  %8.2f ms\n", elapsed_time_ms);

  printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

  if (d_data) cudaFree(d_data);
  if (h_data) free(h_data);

  cudaDeviceReset();

  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
