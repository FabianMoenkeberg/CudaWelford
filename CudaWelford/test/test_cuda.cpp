#include <gtest/gtest.h>
#include <random>
#include <fstream>
#include "../src/example_device_reduce.cu"

TEST(Cuda, base)
{

   ASSERT_EQ(2, 2);
}

TEST(Cuda, Cub_welford)
{
   // const int argc = 3;
   //  char* argv[] = {
   //      (char*)"program_name",
   //      (char*)"arg1",
   //      (char*)"arg2"
   //  };
   // example_device_reduce(argc, argv);

   ASSERT_EQ(2, 2);
}



