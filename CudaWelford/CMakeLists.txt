enable_language(CUDA)

add_library(CUDAWELFORD
    src/cuda_welford.cu
    src/cub_welford.cu
    src/example_device_reduce.cu
    src/cub_sum.cu
    src/test_util.cpp
    )
    
target_include_directories(CUDAWELFORD PUBLIC
    ${Boost_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/CudaWelford/include"
    ${CUDAToolkit_INCLUDE_DIRS}
	)

target_link_libraries(CUDAWELFORD PUBLIC  
    ${CUDAToolkit_LIBRARIES}
    CUDA::cudart 
    CUDA::cublas
	)

target_compile_options(CUDAWELFORD PRIVATE)