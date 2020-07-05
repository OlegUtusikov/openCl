#pragma once
#include <CL/opencl.h>
#include <memory>
#include <vector>

struct Environment {
    Environment() = default;
    ~Environment() {
        if (devices != nullptr)      free(devices);
        if (context != nullptr)      free(context);
        if (commandQueue != nullptr) free(commandQueue);
        if (program != nullptr)      free(program);
        if (kernels != nullptr)      free(kernels);
    }

    cl_uint             devicesCount        { 0 };
    cl_device_id*       devices             { nullptr };
    cl_context          context             { nullptr };
    cl_device_id        device              { nullptr };
    cl_command_queue    commandQueue        { nullptr };
    cl_program          program             { nullptr };
    cl_kernel*          kernels             { nullptr };
    size_t              localWorkSize       { 1 };
    size_t              elementsOneThread   { 1 };
    cl_queue_properties queueProps          { CL_QUEUE_PROFILING_ENABLE };
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
    params->commandQueue = clCreateCommandQueue(params->context, params->devices[0], params->queueProps, &result);
    if (result == CL_SUCCESS && params->commandQueue != nullptr) {
        params->device = params->devices[0];
        printf("Command queue created!\n");
    }
}

void read_and_build(const char* filename, const std::shared_ptr<Environment> &params, const std::string& options) {
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
                            options.c_str(),
                            nullptr,
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
    if (current <= mode) return mode;
    return current % mode != 0 ? mode * (1 + current / mode) : current;
}

template<typename T>
T* alloc_array(size_t size) {
    auto* res = static_cast<T*>(malloc(size * sizeof(T)));
    std::memset(res, 0, size * sizeof(T));
    return res;
}

template<typename T>
void clear_array(T* matrix) {
    free(matrix);
}