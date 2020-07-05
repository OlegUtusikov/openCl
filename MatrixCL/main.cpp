#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include "../utils.h"

void init_random_matrix(float* matrix, size_t firstShape, size_t secondShape, size_t cols) {
    if (cols < secondShape) return;
    for (size_t i = 0; i < firstShape; ++i) {
        for (size_t j = 0; j < secondShape; ++j) {
            matrix[i * cols + j] = rand() % 10;
        }
    }
}

void transpose(const float* src, float* dst, size_t shapeX, size_t shapeY) {
    for(size_t i = 0; i < shapeX; ++i) {
        for(size_t j = 0; j < shapeY; ++j) {
            dst[j * shapeX + i] = src[i * shapeY + j];
        }
    }
}

void printMatrix(const char* name, const float* matrix, size_t shapeX, size_t shapeY, size_t cols) {
    if (cols < shapeY) {
        printf("Bad dims. Allowed: %zu. Actual: %zu", cols, shapeY);
        return;
    }
    printf("Matrix: %s\n", name);
    for (size_t i = 0; i < shapeX; ++i) {
        for (size_t j = 0; j < shapeY; ++j) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

bool check(const float* first, const float* second, const float* result, size_t firstShape, size_t secondShape, size_t thirdShape) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < firstShape; ++i) {
        for (size_t j = 0; j < thirdShape; ++j) {
            float acc = 0;
            for (size_t k = 0; k < secondShape; ++k) {
                acc += first[i * secondShape + k] * second[(k * thirdShape) + j];
            }
            if (static_cast<int>(result[i * thirdShape + j]) != static_cast<int>(acc)) {
                printf("Expected: %f. Actual: %f. i = %zu, j = %zu\n", acc, result[i * thirdShape + j], i, j);
                return false;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    size_t elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Trivial time: %zu mc. (seconds: %f)\n", elapsed_seconds, elapsed_seconds / 1000000.0);
    return true;
}

int main() {
    srand(time(nullptr));
    size_t A = 2048;
    size_t B = 512;
    size_t C = 1024;
    std::shared_ptr<Environment> params(new Environment());
    params->elementsOneThread = 16;
    params->localWorkSize = get_nearest_up(32, params->elementsOneThread);
    size_t firstShape  = get_nearest_up(A, params->localWorkSize);
    size_t secondShape = get_nearest_up(B, params->localWorkSize);
    size_t thirdShape  = get_nearest_up(C, params->localWorkSize);

    auto* firstMatrix   = alloc_array<float>(firstShape  * secondShape);
    auto* secondMatrix  = alloc_array<float>(secondShape * thirdShape);
    auto* secondMatrixT = alloc_array<float>(secondShape * thirdShape);
    auto* resultMatrix  = alloc_array<float>(firstShape  * thirdShape);

    init_random_matrix(firstMatrix, A, B, secondShape);
    init_random_matrix(secondMatrix, B, C, thirdShape);
    transpose(secondMatrix, secondMatrixT, secondShape, thirdShape);

    init(params);
    create_context(params);
    read_and_build("function_matrix.cl", params,
            "-D LOCAL_GROUP_SIZE=" + std::to_string(params->localWorkSize) +
                   " -D ELEMENTS=" + std::to_string(params->elementsOneThread));

    params->kernels = static_cast<cl_kernel*>(malloc(sizeof(cl_kernel)));
    params->kernels[0] = clCreateKernel(params->program, "matrix_mul", nullptr);

    // set arguments
    cl_int result;
    size_t firstSize  = firstShape  * secondShape * sizeof(float);
    size_t secondSize = secondShape * thirdShape  * sizeof(float);
    size_t resultSize = firstShape  * thirdShape  * sizeof(float);
    cl_mem firstBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, firstSize, nullptr, &result);
    cl_mem secondBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, secondSize, nullptr, &result);
    cl_mem resultBuffer = clCreateBuffer(params->context, CL_MEM_READ_WRITE, resultSize, nullptr, &result);

    cl_mem firstSizeBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, sizeof(size_t), nullptr, &result);
    cl_mem secondSizeBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, sizeof(size_t), nullptr, &result);
    cl_mem thirdSizeBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, sizeof(size_t), nullptr, &result);

    clEnqueueWriteBuffer(params->commandQueue, firstBuffer, CL_TRUE, 0, firstSize, firstMatrix, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(params->commandQueue, secondBuffer, CL_TRUE, 0, secondSize, secondMatrixT, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(params->commandQueue, firstSizeBuffer, CL_TRUE, 0, sizeof(size_t), &firstShape, 0, nullptr,
                         nullptr);
    clEnqueueWriteBuffer(params->commandQueue, secondSizeBuffer, CL_TRUE, 0, sizeof(size_t), &secondShape, 0, nullptr,
                         nullptr);
    clEnqueueWriteBuffer(params->commandQueue, thirdSizeBuffer, CL_TRUE, 0, sizeof(size_t), &thirdShape, 0, nullptr,
                         nullptr);

    clSetKernelArg(params->kernels[0], 0, sizeof(cl_mem), &firstBuffer);
    clSetKernelArg(params->kernels[0], 1, sizeof(cl_mem), &secondBuffer);
    clSetKernelArg(params->kernels[0], 2, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(params->kernels[0], 3, sizeof(cl_mem), &firstSizeBuffer);
    clSetKernelArg(params->kernels[0], 4, sizeof(cl_mem), &secondSizeBuffer);
    clSetKernelArg(params->kernels[0], 5, sizeof(cl_mem), &thirdSizeBuffer);

    // execution
    constexpr size_t workDims = 2;
    size_t globalWorkSize[workDims] = { firstShape, thirdShape / params->elementsOneThread };
    size_t localWorkSize[workDims] =  { params->localWorkSize, params->localWorkSize / params->elementsOneThread };

    cl_event event;
    result = clEnqueueNDRangeKernel(params->commandQueue, params->kernels[0], workDims, nullptr, globalWorkSize,
                                    localWorkSize, 0, nullptr, &event);

    clWaitForEvents(1, &event);
    clFinish(params->commandQueue);

    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
    double time = time_end - time_start;

    result = clEnqueueReadBuffer(params->commandQueue, resultBuffer, CL_TRUE, 0, resultSize, resultMatrix, 0, nullptr,
                                 nullptr);
    if (check(firstMatrix, secondMatrix, resultMatrix, firstShape, secondShape, thirdShape)) {
        printf("Time: %f seconds.\n", time / 1e9);
        printf("GFLOPS: %f.\n", 2.0 * firstShape * thirdShape * secondShape / time);
    }
    clear_array(firstMatrix);
    clear_array(secondMatrix);
    clear_array(secondMatrixT);
    clear_array(resultMatrix);
    return EXIT_SUCCESS;
}