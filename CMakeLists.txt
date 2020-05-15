cmake_minimum_required(VERSION 3.1)
project(ParallelProgramming)

add_subdirectory(OpenCL)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/OpenCL/function.txt)
file(COPY function.txt
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/OpenCL)