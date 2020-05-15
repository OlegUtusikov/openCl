#include <cstdio>
#include <CL/opencl.h>
#include <cstdlib>
#include <memory>

struct WorkingParams {
    cl_uint devicesCount{0};
    cl_device_id *devices{nullptr};
    cl_context context{nullptr};
    cl_device_id device{nullptr};
    cl_command_queue commandQueue{nullptr};
    cl_program program{nullptr};
    cl_kernel kernel{nullptr};
    size_t local_work_size{1};
};

void init(const std::shared_ptr<WorkingParams> &params) {
    cl_int result;
    cl_platform_id *platformIds;
    cl_uint platformsCount = 0;

    result = clGetPlatformIDs(0, nullptr, &platformsCount);
    if (result != CL_SUCCESS) {
        printf("Obtain platforms count error! Error code: %d", result);
        return;
    }

    printf("Platforms count: %d\n", platformsCount);
    platformIds = (cl_platform_id *) malloc(platformsCount * sizeof(cl_platform_id));

    result = clGetPlatformIDs(platformsCount, platformIds, nullptr);
    if (result != CL_SUCCESS) {
        printf("Obtain platforms error! Error code: %d", result);
        return;
    }

    for (int i = 0; i < platformsCount; ++i) {
        cl_platform_id currentPlatformId = platformIds[i];

        cl_uint devicesCount = 0;
        result = clGetDeviceIDs(currentPlatformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &devicesCount);
        if (result != CL_SUCCESS || devicesCount == 0) {
            printf("Get devices count! Error code: %d\n", result);
            continue;
        }

        printf("Platform %d. GPU devices count: %d\n", (i + 1), devicesCount);
        auto *devices = static_cast<cl_device_id *>(malloc(devicesCount * sizeof(cl_device_id)));

        result = clGetDeviceIDs(currentPlatformId, CL_DEVICE_TYPE_GPU, devicesCount, devices, nullptr);
        if (result != CL_SUCCESS) {
            printf("Get devices error! Error code: %d\n", result);
            continue;
        }
        params->devices = devices;
        params->devicesCount = devicesCount;
        printf("Found suitable devices!\n");
        break;
    }
    free(platformIds);
}

void create_context(const std::shared_ptr<WorkingParams> &params) {
    if (params->devicesCount == 0) {
        return;
    }

    cl_int result;
    params->context = clCreateContext(nullptr, params->devicesCount, params->devices, nullptr, nullptr, &result);
    if (params->context == nullptr || result != CL_SUCCESS) {
        printf("Creating context error! Error code: %d\n", result);
        return;
    }

    params->commandQueue = clCreateCommandQueueWithProperties(params->context, params->devices[0], nullptr, &result);
    if (result == CL_SUCCESS && params->commandQueue != nullptr) {
        params->device = params->devices[0];
        printf("Create working command queue!\n");
    }
}

void read_and_build(const std::shared_ptr<WorkingParams> &params) {
    cl_int result;
    FILE *f = fopen("function.txt", "rb");
    fseek(f, 0, SEEK_END);
    size_t fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *code = static_cast<char *>(malloc(fileSize * sizeof(char)));
    fread(code, 1, fileSize, f);

    params->program = clCreateProgramWithSource(params->context, 1, const_cast<const char **>(&code), &fileSize,
                                                &result);
    if (result != CL_SUCCESS) {
        printf("Create program from source error: %d\n", result);
        return;
    }

    fclose(f);
    free(code);

    printf("Program created!\n");

    result = clBuildProgram(params->program, params->devicesCount, params->devices, ("-D LOCAL_GROUP_SIZE=" + std::to_string(params->local_work_size)).c_str(), nullptr, nullptr);
    if (result != CL_SUCCESS) {
        printf("Build program error: %d\n", result);
        char *errorBuf = static_cast<char *>(malloc(2048 * sizeof(char)));
        clGetProgramBuildInfo(params->program, params->device, CL_PROGRAM_BUILD_LOG, 2048, errorBuf, nullptr);
        printf("%s", errorBuf);
        free(errorBuf);
        return;
    }
    printf("Program was built!\n");
}

int main() {
    std::shared_ptr<WorkingParams> params(new WorkingParams());
    size_t firstShape = 4;
    size_t secondShape = 5;
    size_t thirdShape = 4;
    params->local_work_size = 2;
    if (firstShape % params->local_work_size != 0 ||
        thirdShape % params->local_work_size != 0) {
        printf("Incorrect local group size\n");
        return EXIT_FAILURE;
    }

    size_t firstSize = firstShape * secondShape * sizeof(cl_float);
    size_t secondSize = secondShape * thirdShape * sizeof(cl_float);
    size_t resultSize = firstShape * thirdShape * sizeof(cl_float);

    auto *firstMatrix = static_cast<cl_float *>(malloc(firstSize));
    auto *secondMatrix = static_cast<cl_float *>(malloc(secondSize));
    auto *resultMatrix = static_cast<cl_float *>(malloc(resultSize));

    printf("First matrix:\n");
    for (size_t i = 0; i < firstShape; ++i) {
        for (size_t j = 0; j < secondShape; ++j) {
            firstMatrix[i * secondShape + j] = i;
            if (j == 0) {
                printf("%zu", i);
            } else {
                printf(" %zu", i);
            }
        }
        printf("\n");
    }

    printf("Second matrix:\n");
    for (size_t i = 0; i < secondShape; ++i) {
        for (size_t j = 0; j < thirdShape; ++j) {
            secondMatrix[i * thirdShape + j] = i;
            if (j == 0) {
                printf("%zu", i);
            } else {
                printf(" %zu", i);
            }
        }
        printf("\n");
    }

    cl_int result;
    const char* funName = "matrixMultiplication";

    init(params);
    create_context(params);
    read_and_build(params);

    params->kernel = clCreateKernel(params->program, funName, nullptr);

    cl_mem firstBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, firstSize, nullptr, &result);
    cl_mem secondBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, secondSize, nullptr, &result);
    cl_mem resultBuffer = clCreateBuffer(params->context, CL_MEM_READ_WRITE, resultSize, nullptr, &result);

    cl_mem firstSizeBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, sizeof(cl_int), nullptr, &result);
    cl_mem secondSizeBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, sizeof(cl_int), nullptr, &result);
    cl_mem thirdSizeBuffer = clCreateBuffer(params->context, CL_MEM_READ_ONLY, sizeof(cl_int), nullptr, &result);

    clEnqueueWriteBuffer(params->commandQueue, firstBuffer, CL_TRUE, 0, firstSize, firstMatrix, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(params->commandQueue, secondBuffer, CL_TRUE, 0, secondSize, secondMatrix, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(params->commandQueue, firstSizeBuffer, CL_TRUE, 0, sizeof(cl_int), &firstShape, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(params->commandQueue, secondSizeBuffer, CL_TRUE, 0, sizeof(cl_int), &secondShape, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(params->commandQueue, thirdSizeBuffer, CL_TRUE, 0, sizeof(cl_int), &thirdShape, 0, nullptr, nullptr);

    clSetKernelArg(params->kernel, 0, sizeof(cl_mem), &firstBuffer);
    clSetKernelArg(params->kernel, 1, sizeof(cl_mem), &secondBuffer);
    clSetKernelArg(params->kernel, 2, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(params->kernel, 3, sizeof(cl_mem), &firstSizeBuffer);
    clSetKernelArg(params->kernel, 4, sizeof(cl_mem), &secondSizeBuffer);
    clSetKernelArg(params->kernel, 5, sizeof(cl_mem), &thirdSizeBuffer);

    constexpr size_t workDims = 2;
    size_t globalWorkSize[workDims] = {firstShape, thirdShape};
    size_t localWorkSize[workDims] = {params->local_work_size, params->local_work_size};

    result = clEnqueueNDRangeKernel(params->commandQueue, params->kernel, workDims, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    clFinish(params->commandQueue);
    result = clEnqueueReadBuffer(params->commandQueue, resultBuffer, CL_TRUE, 0, resultSize, resultMatrix, 0, nullptr, nullptr);

    for (size_t i = 0; i < firstShape; ++i) {
        for (size_t j = 0; j < thirdShape; ++j) {
            printf("%.3f ", resultMatrix[i * thirdShape + j]);
        }
        printf("\n");
    }

    free(firstMatrix);
    free(secondMatrix);
    free(resultMatrix);
    return EXIT_SUCCESS;
}