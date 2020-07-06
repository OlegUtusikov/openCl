cmake_minimum_required(VERSION 3.1)
project(ParallelProgramming)

add_subdirectory(MatrixCL)
add_subdirectory(OpenMP)
add_subdirectory(PrefSumCL)
file(COPY MatrixCL/function_matrix.cl
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/MatrixCL)
file(COPY PrefSumCL/function_pref_sum.cl
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/PrefSumCL)