#pragma once
 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

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
        // printf(" %f, %f, %f. ands %f \n", a.M, a.T, a.N, b.T);
        return res;
    }
};

int cubCustomSum();