include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.13.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(cuda_test 
  test_cuda.cpp
  )

target_link_libraries(cuda_test
 PRIVATE
  GTest::GTest
  CUDAWELFORD)

add_test(cuda_gtests cuda_test)

# Specify the source directory containing the data folder
set(DATA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/CudaWelford/test/data")

# Specify the destination directory in the build folder
set(DATA_DEST_DIR "${CMAKE_BINARY_DIR}")

# Copy the data folder from the source directory to the destination directory
file(COPY "${DATA_SOURCE_DIR}" DESTINATION "${DATA_DEST_DIR}")