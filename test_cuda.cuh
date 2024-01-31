#ifndef test_cuda_cuh
#define test_cuda_cuh

#include <iostream>
#include <functional>
#include <string>
#include <vector>

template <typename T>
__global__ void generateIdentityMatrix(int sizeMatrix, T* identityMatrix){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < sizeMatrix && j < sizeMatrix){
        if(i == j){
            identityMatrix[i * sizeMatrix + j] = 1;
        }else{
            identityMatrix[i * sizeMatrix + j] = 0;
        }
    }
};

template <typename T>
__global__ void errchkMatrix(T* results, T* expect, int sizeMatrix, bool* isEquals){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if(i < sizeMatrix && j < sizeMatrix){
       if(results[i * sizeMatrix + j] != expect[i * sizeMatrix + j]){
                    *isEquals = false;
    }
    }
};

template <typename T>
class Expect {
    T value;
    T* matrixValue;
public:
    Expect(T value): value(value){}
    Expect(T* matrixValue): matrixValue(matrixValue){}
    bool toBe(T results){
        return results == value;
    }
    bool toBeMatrix(T* results, int sizeMatrix, dim3 block, dim3 grid){
        bool* d_isEquals;
        cudaMalloc(&d_isEquals, sizeof(bool));
        bool h_isEquals = true;
        cudaMemcpy(d_isEquals, &h_isEquals, sizeof(bool), cudaMemcpyHostToDevice);
        errchkMatrix<T><<<block, grid>>>(results, matrixValue, sizeMatrix, d_isEquals);
        cudaMemcpy(&h_isEquals, d_isEquals, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(d_isEquals);
        return h_isEquals;
    }
};




#endif /* test_cuda_cuh */