#include <gtest/gtest.h>
#include <random>
#include <fstream>
#include "../include/cuda_welford.h"
#include "../include/cub_welford.h"
#include "../include/cub_sum.h"

TEST(Cuda, CubSum){

   CubSum::cubCustomSum();
}

TEST(Cuda, CustomCudaWelford){

   run_welford();
}

TEST(Cuda, CubVarianceMultistep){

   cubVarianceReduceMultiCall();
}

TEST(Cuda, CubWelford){

   cubWelfordReduceSingle();
}



