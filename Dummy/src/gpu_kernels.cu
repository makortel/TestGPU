#include "TestGPU/Dummy/interface/gpu_kernels.h"

#include <stdio.h>

#define NUM_VALUES 10000

//
// Vector Addition Kernel
//
template<typename T>
__global__
void vectorAdd(T *a, T *b, T *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

namespace testgpu {

//
// The following macros to simplify the template instantiation
//
#define ALLOCATE(NUM_OF_VALUES, TYPE) \
    template void allocate<NUM_OF_VALUES, TYPE>(TYPE**)
#define COPY(NUM_OF_VALUES, TYPE) \
    template void copy<NUM_OF_VALUES, TYPE>(TYPE*, TYPE*, bool)
#define WRAPPERVECTORADD(NUM_OF_VALUES, TYPE) \
    template void wrapperVectorAdd<NUM_OF_VALUES, TYPE>(TYPE*, TYPE*, TYPE*)
#define RELEASE(TYPE) \
    template void release<TYPE>(TYPE*)

template<int NUM_OF_VALUES, typename T>
void allocate(T** values) {
    cudaMalloc(values, NUM_OF_VALUES*sizeof(T));
}

// FIXME: can be put into a separate file
ALLOCATE(10000, int);

template<int NUM_OF_VALUES, typename T>
void copy(T* h_values, T* d_values, bool direction) {
    if (direction) 
        cudaMemcpy(d_values, h_values, NUM_OF_VALUES*sizeof(T), cudaMemcpyHostToDevice);
    else
        cudaMemcpy(h_values, d_values, NUM_OF_VALUES*sizeof(T), cudaMemcpyDeviceToHost);
}

// FIXME: can be put into a separate file
COPY(10000, int);

template<int NUM_OF_VALUES, typename T>
void wrapperVectorAdd(T* d_a, T* d_b, T* d_c) {
    int threadsPerBlock {256};
    int blocksPerGrid = (NUM_OF_VALUES + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
}

// FIXME: can be put into a separate file
WRAPPERVECTORADD(10000, int);

template<typename T>
void release(T* d_values) {
    cudaFree(d_values);
}

// FIXME: can be put into a separate file
RELEASE(int);

//
// Standalone function that allocates/copies/launches/frees and prints the results
//
void launch_on_gpu() {
    printf("start launch_on_gpu\n");
    int h_a[NUM_VALUES], h_b[NUM_VALUES], h_c[NUM_VALUES];
    for (auto i=0; i<NUM_VALUES; i++) {
        h_a[i] = i;
        h_b[i] = i*i;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, NUM_VALUES*sizeof(int));
    cudaMalloc(&d_b, NUM_VALUES*sizeof(int));
    cudaMalloc(&d_c, NUM_VALUES*sizeof(int));

    cudaMemcpy(d_a, h_a, NUM_VALUES*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NUM_VALUES*sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock {256};
    int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, NUM_VALUES*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (auto i=0; i<10; i++) {
        printf("c[%d] = %d\n", i, h_c[i]);
    }

    printf("\n");
    printf("stop launch_on_gpu\n");
}

}
