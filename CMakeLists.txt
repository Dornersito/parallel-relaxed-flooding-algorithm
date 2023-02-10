# forked from the template by j-levy
# How to use
# 1. mkdir build
# 2. cd build
# 3. cmake ..
# 4. make
# 5. <run your program>

cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME PRFA)
set(DEBUG false)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)    # RTX 3070
endif()

add_executable(${PROJECT_NAME} main.cpp
    bfa/bfaHost.cu
    pfa/pfaHost.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenMP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} ${OpenMP_CXX_LIBRARIES} ${OpenCV_LIBRARIES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

if (DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -g
        -G
        >)
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)