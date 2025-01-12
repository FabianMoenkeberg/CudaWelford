#include "CudaWelford/src/cuda_prog_all.cu"
#include "CudaWelford/src/cuda_prog.cu"
#include "CudaWelford/include/cuda_welford.h"
#include "CudaWelford/include/cub_welford.h"
#include "CudaWelford/include/example_device_reduce.h"
#include "CudaWelford/include/cub_sum.h"

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cassert>
#include <iostream>
#include <string>

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int main(int argc, char* argv[])
{
  char **string_retval = new char*[1];
  string_retval[0] = new char[100];
  std::cout << "Start New!" << std::endl;
  example_device_reduce(argc, argv);
  cubCustomSum();

  cubWelford();
  
  run_welford();
}

// #include <stdio.h>

// #include <cub/util_allocator.cuh>
// #include <cub/device/device_reduce.cuh>

// #include "CudaWelford/src/test_util.h"

// using namespace cub;

// bool                    g_verbose = false;  // Whether to display input/output to console
// CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

// /**
//  * Initialize problem
//  */
// void Initialize(
//     int   *h_in,
//     int     num_items)
// {
//     for (int i = 0; i < num_items; ++i)
//         h_in[i] = i;

//     if (g_verbose)
//     {
//         printf("Input:\n");
//         DisplayResults(h_in, num_items);
//         printf("\n\n");
//     }
// }


// /**
//  * Compute solution
//  */
// void Solve(
//     int           *h_in,
//     int           &h_reference,
//     int             num_items)
// {
//     for (int i = 0; i < num_items; ++i)
//     {
//         if (i == 0)
//             h_reference = h_in[0];
//         else
//             h_reference += h_in[i];
//     }
// }

// int main(int argc, char** argv)
// {
//     int num_items = 150;

//     // Initialize command line
//     CommandLineArgs args(argc, argv);
//     g_verbose = args.CheckCmdLineFlag("v");
//     args.GetCmdLineArgument("n", num_items);

//     // Print usage
//     if (args.CheckCmdLineFlag("help"))
//     {
//         printf("%s "
//             "[--n=<input items> "
//             "[--device=<device-id>] "
//             "[--v] "
//             "\n", argv[0]);
//         exit(0);
//     }

//     // Initialize device
//     CubDebugExit(args.DeviceInit());

//     printf("cub::DeviceReduce::Sum() %d items (%d-byte elements)\n",
//         num_items, (int) sizeof(int));
//     fflush(stdout);

//     // Allocate host arrays
//     int* h_in = new int[num_items];
//     int  h_reference;

//     // Initialize problem and solution
//     Initialize(h_in, num_items);
//     Solve(h_in, h_reference, num_items);

//     // Allocate problem device arrays
//     int *d_in = NULL;
//     CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(int) * num_items));

//     // Initialize device input
//     CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));

//     // Allocate device output array
//     int *d_out = NULL;
//     CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(int) * 1));

//     // Request and allocate temporary storage
//     void            *d_temp_storage = NULL;
//     size_t          temp_storage_bytes = 0;
//     CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
//     CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

//     // Run
//     CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

//     // Check for correctness (and display results, if specified)
//     int compare = CompareDeviceResults(&h_reference, d_out, 1, g_verbose, g_verbose);
//     printf("\t%s", compare ? "FAIL" : "PASS");
//     AssertEquals(0, compare);

//     // Cleanup
//     if (h_in) delete[] h_in;
//     if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
//     if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
//     if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

//     printf("\n\n");

// }