#include <stdio.h>

#include "TestGPU/Dummy/interface/gpu_kernels.h"

int main(int argc, char** argv) {
    printf("Hello World\n");

    // run the kernel
    launch_on_gpu();

    printf("Goodbye World\n");
}
