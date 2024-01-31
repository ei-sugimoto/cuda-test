#include "test_cuda.cuh"
#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include "wmma.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

void testMain() {
    const int sizeMatrix = 16;
    __half* testMatrix;
    __half* matrix, *A, *result;
    cudaMallocManaged(&matrix, sizeMatrix * sizeMatrix * sizeof(__half));
    cudaMallocManaged(&testMatrix, sizeMatrix * sizeMatrix * sizeof(int));
    cudaMallocManaged(&A, sizeMatrix * sizeMatrix * sizeof(__half));
    cudaMallocManaged(&result, sizeMatrix * sizeMatrix * sizeof(__half));
    generateIdentityMatrix<__half><<<1, dim3(16, 16)>>>(sizeMatrix, matrix);
    generateIdentityMatrix<__half><<<1, dim3(16, 16)>>>(sizeMatrix, A);
    generateIdentityMatrix<__half><<<1, dim3(16, 16)>>>(sizeMatrix, testMatrix);
    wmma_ker<__half, __half><< <dim3(sizeMatrix / 16, sizeMatrix / 16), 32 >> >(A, A, result, sizeMatrix);
    cudaDeviceSynchronize();


    printMatrix<__half>(sizeMatrix,sizeMatrix, result);

    Expect<__half> expect(matrix);
    bool flag = expect.toBeMatrix(result, sizeMatrix, /*block*/dim3(16, 16), /*grid*/1);

    if (flag) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
    }
}

int main(void) {
    testMain();
    return 0;
}

