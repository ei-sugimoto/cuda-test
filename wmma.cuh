#ifndef WMMA_HPP
#define WMMA_HPP
#include <mma.h>
using namespace nvcuda;
template <typename T, typename C>
__global__ void wmma_ker(const T* a, const T* b, C* c, const int N) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, C> c_frag;
       
    // Initialize the output to zero
    wmma::fill_fragment(c_frag, __float2half(.0f));
    // Load the inputs

    for (auto k = 0; k < N; k += 16) {
        wmma::load_matrix_sync(a_frag, &a[blockIdx.y * N * 16 + k], N);
        wmma::load_matrix_sync(b_frag, &b[k * N + blockIdx.x * 16], N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // Store the output
    wmma::store_matrix_sync(&c[blockIdx.y * N * 16 + blockIdx.x * 16], c_frag, N, wmma::mem_row_major);

}

template<class Element>
void printMatrix(int m, int n, const Element* A)
/*
  要素がElement型のm行n列の行列Aを表示する．
*/
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) std::cout << __half2float(A[i * n + j]) << ", ";
        std::cout << std::endl;
    }

}


#endif // WMMA_HPP
