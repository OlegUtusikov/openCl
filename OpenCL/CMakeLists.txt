cmake_minimum_required(VERSION 3.1)
project(OpenCL)

set(SOURCE_FILES main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

if (OPENCL_FOUND)
    message(STATUS "Found OpenCL!")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})
else ()
    message(STATUS "OpenCL didn't find!")
endif ()