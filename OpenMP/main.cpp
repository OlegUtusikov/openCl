#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <omp.h>

float* createMatrix(size_t shapeX, size_t shapeY) {
    size_t len = sizeof(float) * shapeX * shapeY;
    auto* res = static_cast<float*>(malloc(len));
    memset(res, 0, len);
    return res;
}

void randomMatrix(float* matrix, size_t shapeX, size_t shapeY) {
    for(size_t i = 0; i < shapeX * shapeY; ++i) {
        matrix[i] = rand() % 100;
    }
}

void clearMatrix(float* matrix) {
    free(matrix);
}

void transpose(const float* src, float* dst, size_t shapeX, size_t shapeY) {
    for(size_t i = 0; i < shapeX; ++i) {
        for(size_t j = 0; j < shapeY; ++j) {
            dst[j * shapeX + i] = src[i * shapeY + j];
        }
    }
}

void mulMatrixSeq(const float* firstMatrix, size_t shapeX1, size_t shapeY1,
                  const float* secondMatrix, size_t shapeX2, size_t shapeY2, float* resultMatrix) {
    for (int i = 0; i < shapeX1; ++i) {
        for (int j = 0; j < shapeY2; ++j) {
            for (int k = 0; k < shapeY1; ++k) {
                resultMatrix[i * shapeY2 + j] += firstMatrix[i * shapeY1 + k] * secondMatrix[k * shapeY2 + j];
            }
        }
    }
}

void mulMatrix(const float* firstMatrix, size_t shapeX1, size_t shapeY1,
               const float* secondMatrix, size_t shapeX2, size_t shapeY2, float* resultMatrix) {
    #pragma omp parallel shared(firstMatrix, secondMatrix, resultMatrix)
    {
        int i, j, k;
        #pragma omp for schedule (static)
        for (i = 0; i < shapeX1; ++i) {
            for (j = 0; j < shapeY2; ++j) {
                float sum;
                for (k = 0; k < shapeY1; ++k) {
                    sum += firstMatrix[i * shapeY1 + k] * secondMatrix[j * shapeX2 + k];
                }
                resultMatrix[i * shapeY2 + j] = sum;
            }
        }
    }
}

void printMatrix(const char* name, const float* matrix, size_t shapeX, size_t shapeY) {
    std::cout << "Matrix: " << name << std::endl;
    for (size_t i = 0; i < shapeX; ++i) {
        for (size_t j = 0; j < shapeY; ++j) {
            std::cout << matrix[i * shapeY + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    srand(time(nullptr));
    omp_set_num_threads(8);
    size_t shapeX1 = 4000;
    size_t shapeY1 = 1000;
    size_t shapeX2 = 1000;
    size_t shapeY2 = 3000;

    if (shapeY1 != shapeX2) {
        printf("Incorrect dims. shapeY1: %zu, shapeX2: %zu", shapeY1, shapeX2);
        return 0;
    }

    float* firstMatrix  = createMatrix(shapeX1, shapeY1);
    float* secondMatrix = createMatrix(shapeX2, shapeY2);
    float* resultMatrix = createMatrix(shapeX1, shapeY2);

    randomMatrix(firstMatrix, shapeX1, shapeY1);
    randomMatrix(secondMatrix, shapeX2, shapeY2);

    float* secondMatrixT = createMatrix(shapeX2, shapeY2);
    transpose(secondMatrix, secondMatrixT, shapeX2, shapeY2);

    auto startSEQ = std::chrono::steady_clock::now();
    mulMatrixSeq(firstMatrix, shapeX1, shapeY1, secondMatrix, shapeX2, shapeY2, resultMatrix);
    auto endSEQ = std::chrono::steady_clock::now();
    auto elapsedSEQ = std::chrono::duration_cast<std::chrono::microseconds>(endSEQ - startSEQ);

    auto startMP = std::chrono::steady_clock::now();
    mulMatrix(firstMatrix, shapeX1, shapeY1, secondMatrixT, shapeX2, shapeY2, resultMatrix);
    auto endMP = std::chrono::steady_clock::now();
    auto elapsedMP = std::chrono::duration_cast<std::chrono::microseconds>(endMP - startMP);

    std::cout << "TimeSEQ: " << elapsedSEQ.count() << "mc (" << (elapsedSEQ.count() / 1000000.0) << " s)." << std::endl;
    std::cout << "TimeMP: " << elapsedMP.count() << "mc (" << (elapsedMP.count() / 1000000.0) << " s)." << std::endl;
    std::cout << "Speed up :" << ((float)elapsedSEQ.count() / elapsedMP.count()) << std::endl;

    //printMatrix("result", resultMatrix, shapeX1, shapeY2);

    clearMatrix(firstMatrix);
    clearMatrix(secondMatrix);
    clearMatrix(secondMatrixT);
    clearMatrix(resultMatrix);
    return 0;
}