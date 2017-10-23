#include <stdio.h>

#define N 10000

__global__ 
void vectorAdd(int *a, int *b, int *c) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__
void matrixAdd(int **a,int **b, int**c) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    c[i][j] = a[i][j] + b[i][j];
}

int main(int argc, char** argv) {
    printf("Hello World\n");
    int h_a[N], h_b[N], h_c[N];
    for (auto i=0; i<N; i++) {
        h_a[i] = i;
        h_b[i] = i*i;
    }

    // allocation
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));

    // copy from host to device memory
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

    // vector addition
    int threadsPerBlock(256);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    // copy the result output
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (auto i=0; i<10; i++) {
        printf("c[%d] = %d\n", i, h_c[i]);
    }
    printf("\n");
    
    // matrix addition
    /*
    dim3 threadsPerBlock(10, 10);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    maxtrixAdd<<<numBlocks, threadsPerBlock>>>(dma, dmb, dmc);
    */

    printf("Goodbye World\n");
}
