#pragma once
 
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <cuda_fp16.h>
// #include <cub/cub.cuh>

namespace CubSum{

    struct point {
        float M;
        float T;
        float N;
    };

    int cubCustomSum();
}