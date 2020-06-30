cmake_minimum_required(VERSION 3.1)
project(OpenCL)

set(SRC main.cpp)

add_executable(${PROJECT_NAME} ${SRC})

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

if (OPENCL_FOUND)
    message(STATUS "OpenCL found.")
    message(STATUS "linking...")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})
else ()
    message(STATUS "Couldn't find OpenCL.")
endif ()