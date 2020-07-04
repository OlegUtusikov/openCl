#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include "../utils.h"

void init_rand_array(int* array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = 1 % 100;
    }
}

void print_array(const char* prefix, int* array, size_t size) {
    printf("%s\n", prefix);
    for (size_t i = 0; i < size; ++i) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    srand(time(nullptr));
    std::shared_ptr<Environment> params(new Environment());
    size_t n = 20;
    params->local_work_size = n;
    size_t cnt = get_nearest_up(n, params->local_work_size);
    printf("SIZE: %zu\n", cnt);
    int* array  = alloc_array<int>(cnt);
    init_rand_array(array, cnt);
    int* result_array = alloc_array<int>(cnt);

    init(params);
    create_context(params);
    read_and_build("function_pref_sum.cl", params,
            "-D LOCAL_GROUP_SIZE=" + std::to_string(params->local_work_size) +
                  " -D ELEMENTS=" + std::to_string(params->elements_one_thread) +
                  " -D MAX_ARRAY_SIZE=" + std::to_string(params->max_local_size_mem));

    params->kernel = clCreateKernel(params->program, "prefix_sum", nullptr);

    size_t size = cnt * sizeof(int);
    // set arguments
    cl_int result;
    cl_mem arrayBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, size, nullptr, &result);
    cl_mem resultBuffer = clCreateBuffer(params->context, CL_MEM_READ_WRITE, size, nullptr, &result);

    cl_mem sizeBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, sizeof(size_t), nullptr, &result);

    clEnqueueWriteBuffer(params->commandQueue, arrayBuffer, CL_TRUE, 0, size, array, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(params->commandQueue, sizeBuffer,  CL_TRUE, 0, sizeof(size_t), &size, 0, nullptr,
                         nullptr);

    clSetKernelArg(params->kernel, 0, sizeof(cl_mem), &arrayBuffer);
    clSetKernelArg(params->kernel, 1, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(params->kernel, 2, sizeof(cl_mem), &sizeBuffer);


    // execution
    constexpr size_t workDims = 1;
    size_t globalWorkSize[workDims] = { cnt };
    size_t localWorkSize[workDims] = { params->local_work_size };

    cl_event event;
    result = clEnqueueNDRangeKernel(params->commandQueue, params->kernel, workDims, nullptr, globalWorkSize,
                                    localWorkSize, 0, nullptr, &event);

    clWaitForEvents(1, &event);
    clFinish(params->commandQueue);

    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
    double time = time_end - time_start;

    result = clEnqueueReadBuffer(params->commandQueue, resultBuffer, CL_TRUE, 0, size, result_array, 0, nullptr,
                                 nullptr);

    printf("Time: %f seconds.\n", time / 1e9);
    printf("GFLOPS: %f.\n", cnt / time);

    print_array("array", array, n);
    print_array("result", result_array, n);
    clear_matrix(array);
    clear_matrix(result_array);
    return EXIT_SUCCESS;
}