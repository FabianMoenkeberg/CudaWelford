cmake_minimum_required(VERSION 3.27)

set(CMAKE_VERBOSE_MAKEFILE ON)
enable_testing()
message(INFO " CMAKE version: ${CMAKE_VERSION}")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(Info ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=C++17")

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_GENERATOR "Visual Studio 16 2019")

set(CMAKE_GENERATOR_TOOLSET "v142")

find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
  message(INFO " versions: ${CUDAToolkit_VERSION}")
  message(INFO " dirs: ${CUDA_SDK_ROOT_DIR}")
endif()

enable_language(CUDA)
project(CudaCmake LANGUAGES CXX) # also need the C language because of imgproc.c, later to be changed to cpp

# IMPORTANT CUDA FLAG to improve performance, but it might reduce accuracy.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

enable_testing()

add_subdirectory(CudaWelford)

add_compile_options(-lineinfo)
# add_compile_options(-Xcompiler -fopenmp)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(CudaRun 
  main.cpp
  )

target_include_directories(CudaRun PUBLIC
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(CudaRun 
    CUDAWELFORD
    ${CUDAToolkit_LIBRARIES}
    CUDA::cudart 
    CUDA::cublas
    )

add_subdirectory(CudaWelford/test)

    