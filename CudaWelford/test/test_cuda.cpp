#include <gtest/gtest.h>
#include <random>
#include <fstream>
#include "../include/cuda_welford.h"
#include "../include/cub_welford.h"
#include "../include/cub_sum.h"

TEST(Cuda, cubSum){

   CubSum::cubCustomSum();
}

TEST(Cuda, customCuda){

   run_welford();
}

TEST(Cuda, CubWelfordMultistep){

   cubWelfordReduceMultiCall();
}

TEST(Cuda, CubCustomSum){

   cubWelfordReduceSingle();
}



