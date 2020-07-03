#include <cstdio>
#include <CL/opencl.h>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <memory>

struct Environment {
    Environment() = default;
    ~Environment() {
        if (devices != nullptr)      free(devices);
        if (context != nullptr)      free(context);
        if (commandQueue != nullptr) free(commandQueue);
        if (program != nullptr)      free(program);
        if (kernel != nullptr)       free(kernel);
    }

    cl_uint          devicesCount    { 0 };
    cl_device_id*    devices         { nullptr };
    cl_context       context         { nullptr };
    cl_device_id     device          { nullptr };
    cl_command_queue commandQueue    { nullptr };
    cl_program       program         { nullptr };
    cl_kernel        kernel          { nullptr };
    size_t           local_work_size { 1 };
};

void init(const std::shared_ptr<Environment> &params) {
    cl_int result;
    cl_uint platformsCount = 0;

    result = clGetPlatformIDs(0, nullptr, &platformsCount);
    if (result != CL_SUCCESS) {
        printf("Can't get platforms count! Error code: %d", result);
        return;
    }
    printf("Platforms count: %d\n", platformsCount);

    auto* platformIds = static_cast<cl_platform_id*>(malloc(platformsCount * sizeof(cl_platform_id)));

    result = clGetPlatformIDs(platformsCount, platformIds, nullptr);
    if (result != CL_SUCCESS) {
        printf("Can't get platforms! Error code: %d", result);
        return;
    }
    const auto chooseDevices = [platformsCount, platformIds](const std::shared_ptr<Environment> &params, int device_type) {
        cl_int result = CL_DEVICE_NOT_FOUND;
        for (size_t i = 0; i < platformsCount; ++i) {
            cl_uint devicesCount = 0;
            result = clGetDeviceIDs(platformIds[i], device_type, 0, nullptr, &devicesCount);
            if (result != CL_SUCCESS || devicesCount == 0) {
                printf("Can't get devices! Error code: %d\n", result);
                continue;
            }
            auto* devices = static_cast<cl_device_id*>(malloc(devicesCount * sizeof(cl_device_id)));
            result = clGetDeviceIDs(platformIds[i], device_type, devicesCount, devices, nullptr);
            if (result != CL_SUCCESS) {
                printf("Can't get devices list! Error code: %d\n", result);
                continue;
            }
            params->devices = devices;
            params->devicesCount = devicesCount;
            printf("Found devices!\n");
            break;
        }
        return result;
    };

    if (chooseDevices(params, CL_DEVICE_TYPE_GPU) != CL_SUCCESS) {
        chooseDevices(params, CL_DEVICE_TYPE_CPU);
        printf("CPU devices. Count = %d\n", params->devicesCount);
    } else {
        printf("GPU devices. Count = %d\n", params->devicesCount);
    }
    free(platformIds);
}

void create_context(const std::shared_ptr<Environment> &params) {
    if (params->devicesCount == 0) {
        return;
    }

    cl_int result;
    params->context = clCreateContext(nullptr, params->devicesCount, params->devices, nullptr, nullptr, &result);
    if (params->context == nullptr || result != CL_SUCCESS) {
        printf("Can't create context! Error code: %d\n", result);
        return;
    }

    params->commandQueue = clCreateCommandQueueWithProperties(params->context, params->devices[0], nullptr, &result);
    if (result == CL_SUCCESS && params->commandQueue != nullptr) {
        params->device = params->devices[0];
        printf("Command queue created!\n");
    }
}

void read_and_build(const char* filename, const std::shared_ptr<Environment> &params) {
    cl_int result;
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    size_t fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* code = static_cast<char*>(malloc(fileSize * sizeof(char)));
    fread(code, 1, fileSize, f);

    params->program = clCreateProgramWithSource(params->context, 1, const_cast<const char**>(&code), &fileSize, &result);
    if (result != CL_SUCCESS) {
        printf("Can't create program from source. Error: %d\n", result);
        return;
    }

    fclose(f);
    free(code);

    result = clBuildProgram(params->program, params->devicesCount, params->devices,
                            ("-D LOCAL_GROUP_SIZE=" + std::to_string(params->local_work_size)).c_str(), nullptr,
                            nullptr);
    if (result != CL_SUCCESS) {
        printf("Can't build program. Error: %d\n", result);
        char* errorBuf = static_cast<char*>(malloc(1024 * sizeof(char)));
        clGetProgramBuildInfo(params->program, params->device, CL_PROGRAM_BUILD_LOG, 1024, errorBuf, nullptr);
        printf("%s", errorBuf);
        free(errorBuf);
        return;
    }
    printf("Program was built!\n");
}

inline size_t get_nearest_up(size_t current, size_t mode) {
    return current % mode != 0 ? mode * (1 + current / mode) : current;
}

float* alloc_matrix(size_t size) {
    auto* res = static_cast<float*>(malloc(size));
    std::memset(res, 0, size);
    return res;
}

void clear_matrix(float* matrix) {
    free(matrix);
}

void init_random_matrix(float* matrix, size_t firstShape, size_t secondShape, size_t cols) {
    if (cols < secondShape) return;
    for (size_t i = 0; i < firstShape; ++i) {
        for (size_t j = 0; j < secondShape; ++j) {
            matrix[i * cols + j] = i * cols + j;
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

size_t gcd (size_t first, size_t second) {
    return second > 0 ? gcd(second, first % second) : first;
}

size_t lcm(size_t first, size_t second) {
    return (first * second) / gcd (first, second);
}

int main() {
    size_t A = 3;
    size_t B = 5;
    size_t C = 6;
    std::shared_ptr<Environment> params(new Environment());
    params->local_work_size = 4;
    size_t firstShape  = get_nearest_up(A, params->local_work_size);
    size_t secondShape = get_nearest_up(B, params->local_work_size);
    size_t thirdShape  = get_nearest_up(C, params->local_work_size);

    size_t firstSize  = firstShape  * secondShape * sizeof(float);
    size_t secondSize = secondShape * thirdShape  * sizeof(float);
    size_t resultSize = firstShape  * thirdShape  * sizeof(float);

    auto* firstMatrix   = alloc_matrix(firstSize);
    auto* secondMatrix  = alloc_matrix(secondSize);
    auto* secondMatrixT = alloc_matrix(secondSize);
    auto* resultMatrix  = alloc_matrix(resultSize);

    init_random_matrix(firstMatrix, A, B, secondShape);
    init_random_matrix(secondMatrix, B, C, thirdShape);
    transpose(secondMatrix, secondMatrixT, secondShape, thirdShape);

    init(params);
    create_context(params);
    read_and_build("function.txt", params);

    params->kernel = clCreateKernel(params->program, "matrix_mul", nullptr);

    // set arguments
    cl_int result;
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

    clSetKernelArg(params->kernel, 0, sizeof(cl_mem), &firstBuffer);
    clSetKernelArg(params->kernel, 1, sizeof(cl_mem), &secondBuffer);
    clSetKernelArg(params->kernel, 2, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(params->kernel, 3, sizeof(cl_mem), &firstSizeBuffer);
    clSetKernelArg(params->kernel, 4, sizeof(cl_mem), &secondSizeBuffer);
    clSetKernelArg(params->kernel, 5, sizeof(cl_mem), &thirdSizeBuffer);

    // execution
    constexpr size_t workDims = 2;
    size_t globalWorkSize[workDims] = {firstShape, thirdShape};
    size_t localWorkSize[workDims] = {params->local_work_size, params->local_work_size};

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    result = clEnqueueNDRangeKernel(params->commandQueue, params->kernel, workDims, nullptr, globalWorkSize,
                                    localWorkSize, 0, nullptr, nullptr);

    clFinish(params->commandQueue);
    end = std::chrono::high_resolution_clock::now();

    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    result = clEnqueueReadBuffer(params->commandQueue, resultBuffer, CL_TRUE, 0, resultSize, resultMatrix, 0, nullptr,
                                 nullptr);

    printf("Time: %d mc seconds (%f seconds)\n", elapsed_seconds, elapsed_seconds / 1000000.0);
    printMatrix("first", firstMatrix, A, B, secondShape);
    printMatrix("second", secondMatrix, B, C, thirdShape);
    printMatrix("secondT", secondMatrixT, C, B, secondShape);
    printMatrix("result", resultMatrix, A, C, thirdShape);
    clear_matrix(firstMatrix);
    clear_matrix(secondMatrix);
    clear_matrix(secondMatrixT);
    clear_matrix(resultMatrix);
    return EXIT_SUCCESS;
}