#include <gtest/gtest.h>
#include <random>
#include <fstream>
#include "../include/cuda_welford.h"
#include "../include/cub_welford.h"
#include "../include/cub_sum.h"

TEST(Cuda, CubSum){

   CubSum::cubCustomSum();
}

TEST(Cuda, CustomCudaWelford_optimized2){

   run_welford(2);
}

TEST(Cuda, CustomCudaWelford_optimized1){

   run_welford(1);
}

TEST(Cuda, CustomCudaWelford_optimized0){

   run_welford(0);
}

TEST(Cuda, CubVarianceMultistep){

   cubVarianceReduceMultiCall();
}

TEST(Cuda, CubWelford){

   cubWelfordReduceSingle();
}



