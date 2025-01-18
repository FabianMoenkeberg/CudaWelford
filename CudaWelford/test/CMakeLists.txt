enable_language(CUDA)
enable_testing()
message(INFO " Cuda_test running 1.")

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.13.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)

target_link_libraries(GTest::GTest INTERFACE gtest_main)
message(INFO " Cuda_test running 2.")
add_executable(cuda_test 
  test_cuda.cpp
  )

set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_test
 PUBLIC
  GTest::GTest
  CUDAWELFORD
  ${CUDAToolkit_LIBRARIES}
    CUDA::cudart 
    CUDA::cublas
  )

add_test(NAME cuda_gtests COMMAND cuda_test)

message(INFO " Cuda_test running 3.")

# Specify the source directory containing the data folder
set(DATA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/CudaWelford/test/data")

# Specify the destination directory in the build folder
set(DATA_DEST_DIR "${CMAKE_BINARY_DIR}")

# Copy the data folder from the source directory to the destination directory
file(COPY "${DATA_SOURCE_DIR}" DESTINATION "${DATA_DEST_DIR}")

add_dependencies(CudaRun cuda_test)