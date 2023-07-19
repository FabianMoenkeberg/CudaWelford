#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
namespace Test{
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

__global__ void kernel_test(float *g_data, int dimx, int dimy, int niterations) {
  // Note I just apply a element-wise operation on each element, NOT DIFFERENT on x-axis values.
  int ix = (blockIdx.x * blockDim.x + threadIdx.x)*4 ;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("idx %d\n",ix);
  if (ix < dimx && iy <dimy){
    int idx = iy * dimx + ix;
    // s_g[tx][ty] = g_data[idx];
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

  // note     , 32 - threads in a thread block (should be multiple of 32)    
  int nBlockX = 16;
  int nBlockY = 16;        
  dim3 block(nBlockX, nBlockY);
  dim3 grid((dimx + nBlockX - 1)/nBlockX, (dimy + nBlockY - 1)/nBlockY);

  kernel_test<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

__global__ void kernel_A(float *g_data, int dimx, int dimy, int niterations) {

  for (int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < dimy; iy += blockDim.y * gridDim.y) {
    for (int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < dimx; ix += blockDim.x * gridDim.x) {
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

// 5.3 ms
void launchKernelA(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  int maxThreadsPerBlock;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  dim3 block(1, maxThreadsPerBlock/4);
  dim3 grid(dimx,dimy/block.y);
  // int num_sms = prop.multiProcessorCount;

  // dim3 block(1, 32);
  // dim3 grid(1, num_sms);
  kernel_A<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

__global__ void kernel_H(float *g_data, int dimx, int dimy, int niterations) {

  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix < dimx && iy <dimy){
    int idx = iy * dimx + ix;
    float value = g_data[idx];
    if (ix % 4 == 0) {
        for (int i = 0; i < niterations; i++) {
          value += sqrtf(logf(value) + 1.f);
        }
    }else if (ix % 4 == 1) {
        for (int i = 0; i < niterations; i++) {
          value += sqrtf(cosf(value) + 1.f);
        }
    }else if (ix % 4 == 2) {
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

__global__ void kernel_F(float *g_data, int dimx, int dimy, int niterations) {
  // printf("blockIdx: %d, with bloxDim.x: %d, and gridDim.x: %d \n", blockIdx.x, blockDim.x, gridDim.x);
  // printf("blockIdy: %d, with bloxDim.y: %d, and gridDim.y: %d \n", blockIdx.y, blockDim.y, gridDim.y);
  // __shared__ float s_g[16][16];
  int ix = (blockIdx.x * blockDim.x + threadIdx.x)*4;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  __half one = __float2half(1.0f);
  // float tan;
  if (ix < dimx && iy <dimy){
    int idx = iy * dimx + ix;
    // s_g[tx][ty] = g_data[idx];
    // __half  value = __float2half(g_data[idx]);
    // __half  value1 = __float2half(g_data[idx+1]);
    // __half  value2 = __float2half(g_data[idx+2]);
    // __half  value3 = __float2half(g_data[idx+3]);
    float value = g_data[idx];
    float value1 = g_data[idx+1];
    float value2 = g_data[idx+2];
    float value3 = g_data[idx+3];
    for (int i = 0; i < niterations; i++) {
      value += __half2float(hsqrt(__float2half(logf(value)) + one));
      // value1 += sqrtf(__half2float(hcos(__float2half(value1)) + one));
      // value2 += hsqrt(hsin(value2) + 1.0f);
      value1 += sqrtf(cosf(value1) + 1.0f);
      // value2 += sqrtf(sinf(value2) + 1.0f);
      value2 += __half2float(hsqrt(__float2half(sinf(value2)) + one));
      // tan = tanf(__half2float(value3));
      // value3 += sqrtf((tan) + one);
      // value3 += sqrtf(tanf(value3) + 1.0f);
      value3 += __half2float(hsqrt(__float2half(tanf(value3)) + one));
    }
    g_data[idx] = value;
    g_data[idx+1] = value1;
    g_data[idx+2] = value2;
    g_data[idx+3] = value3;
  }
}

// Note that here we need that dimx = gridDim.x * blockDim.x * 4 * k.
__global__ void kernel_G(float *g_data, int dimx, int dimy, int niterations) {
  
  int widthBlocky = (dimy+gridDim.y-1)/gridDim.y;
  int widthThready = (widthBlocky+blockDim.y-1)/blockDim.y;
  widthBlocky = widthThready * blockDim.y;
  int widthBlockx = (dimx+gridDim.x-1)/gridDim.x;
  int widthThreadx = (widthBlockx+blockDim.x-1)/blockDim.x;
  widthBlockx = widthThreadx*blockDim.x;

  int iStartY = blockIdx.y * widthBlocky + threadIdx.y*widthThready;
  int iEndY = blockIdx.y *widthBlocky + (threadIdx.y+1)*widthThready;
  int iStartX = blockIdx.x *widthBlockx + threadIdx.x*widthThreadx;
  int iEndX = blockIdx.x *widthBlockx + (threadIdx.x+1)*widthThreadx;
  iEndX = min(iEndX, dimx);
  iEndY = min(iEndY, dimy);
  float value, value1, value2, value3;
  
  for (int iy = iStartY; iy < iEndY; iy+=1) {
    for (int ix = iStartX; ix < iEndX; ix+=4) {
      int idx = iy * dimx + ix;

      value = g_data[idx];
      value1 = g_data[idx+1];
      value2 = g_data[idx+2];
      value3 = g_data[idx+3];
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
    }
  }
}

// Note that here we need that dimx = gridDim.x * blockDim.x * 4 * k.
__global__ void kernel_G2(float *g_data, int dimx, int dimy, int niterations) {
  
  int widthBlocky = (dimy+gridDim.y-1)/gridDim.y;
  int widthThready = (widthBlocky+blockDim.y-1)/blockDim.y;
  widthBlocky = widthThready * blockDim.y;
  int widthBlockx = (dimx+gridDim.x-1)/gridDim.x;
  int widthThreadx = (widthBlockx+blockDim.x-1)/blockDim.x;
  widthBlockx = widthThreadx*blockDim.x;

  int iStartY = blockIdx.y * widthBlocky + threadIdx.y*widthThready;
  int iEndY = blockIdx.y *widthBlocky + (threadIdx.y+1)*widthThready;
  int iStartX = blockIdx.x *widthBlockx + threadIdx.x*widthThreadx;
  int iEndX = blockIdx.x *widthBlockx + (threadIdx.x+1)*widthThreadx;
  iEndX = min(iEndX, dimx);
  iEndY = min(iEndY, dimy);
  float4 values;
  
  for (int iy = iStartY; iy < iEndY; iy+=1) {
    for (int ix = iStartX; ix < iEndX; ix+=4) {
      int idx = iy * dimx + ix/4;
      values = reinterpret_cast<float4*>(g_data)[idx];

      for (int i = 0; i < niterations; i++) {
        values.x += sqrtf(logf(values.x) + 1.f);
        values.y += sqrtf(cosf(values.y) + 1.f);
        values.z += sqrtf(sinf(values.z) + 1.f);
        values.w += sqrtf(tanf(values.w) + 1.f);
      }

      reinterpret_cast<float4*>(g_data)[idx] = values;
    }
  }
}

// 2.07 ms
// Note kernel_D needs around 2.15 ms.
void launchKernel(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  int maxThreadsPerBlock;
  cudaGetDeviceProperties(&prop, 0);
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  // note     , 32 - threads in a thread block (should be multiple of 32)    
  int nBlockX = 16;//maxThreadsPerBlock/2;
  int nBlockY = 16;        
  dim3 block(nBlockX, nBlockY);
  dim3 grid(((dimx+4-1)/4 + nBlockX - 1)/nBlockX, (dimy + nBlockY - 1)/nBlockY);
  // dim3 grid(1,1);
  kernel_B<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

// 2.15 ms
void launchKernelG(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  int maxThreadsPerBlock;
  cudaGetDeviceProperties(&prop, 0);
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  // note     , 32 - threads in a thread block (should be multiple of 32)    
  int nBlockX = 32;
  int nBlockY = maxThreadsPerBlock/nBlockX;        
  dim3 block(nBlockX, nBlockY);
  dim3 grid(32, 32);

  kernel_G<<<grid, block>>>(d_data, dimx, dimy, niterations);
}


// 4.7 ms
void launchKernelH(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  int maxThreadsPerBlock;
  cudaGetDeviceProperties(&prop, 0);
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  // note     , 32 - threads in a thread block (should be multiple of 32)    
  int nBlockX = 2;
  int nBlockY = maxThreadsPerBlock/nBlockX/4;      

  dim3 block(nBlockX, nBlockY);
  dim3 grid((dimx + nBlockX - 1)/nBlockX, (dimy + nBlockY - 1)/nBlockY);

  kernel_B<<<grid, block>>>(d_data, dimx, dimy, niterations);
}

float timing_experiment(float *d_data,
                        int dimx, int dimy, int niterations, int nreps) {
  float elapsed_time_ms = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  for (int i = 0; i < nreps; i++) {
    launchKernel(d_data, dimx, dimy, niterations);
  }
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  elapsed_time_ms /= nreps;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_time_ms;
}

int cuda_prog() {
  int dimx = 8 * 1024;
  int dimy = 8 * 1024;

  int dev = 0;
  cudaGetDevice(&dev);
  std::cout << "device found: " << dev << std::endl;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  
  int nreps = 10;
  int niterations = 5;

  int nbytes = dimx * dimy * sizeof(float);

  // allocate memory on device of size dimx x dimy in float
  float *d_data = 0, *h_data = 0, *h_gold = 0;
  cudaMalloc((void **)&d_data, nbytes);
  if (0 == d_data) {
    printf("couldn't allocate GPU memory\n");
    return -1;
  }
  printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));
  // allocate memory on host
  h_data = (float *)malloc(nbytes);
  h_gold = (float *)malloc(nbytes);
  if (0 == h_data || 0 == h_gold) {
    printf("couldn't allocate CPU memory\n");
    return -2;
  }
  printf("allocated %.2f MB on CPU\n", 2.0f * nbytes / (1024.f * 1024.f));
  // create random data h_gold  = 1.0 + 0.01*rand, rand in [0,1]
  for (int i = 0; i < dimx * dimy; i++) h_gold[i] = 1.0f + 0.01*(float)rand()/(float)RAND_MAX;
  // copy h_gold from host to device.
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
}