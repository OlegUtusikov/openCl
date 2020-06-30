cmake_minimum_required(VERSION 3.1)
project(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SRC main.cpp)

add_executable(${PROJECT_NAME} ${SRC})