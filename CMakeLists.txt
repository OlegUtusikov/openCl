cmake_minimum_required(VERSION 3.1)
project(ParallelProgramming)

add_subdirectory(OpenCL)
add_subdirectory(OpenMP)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/OpenCL/function.txt)
file(COPY function.txt
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/OpenCL)