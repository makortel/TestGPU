#ifndef TestGPU_Dummy_interface_gpu_kernels_h
#define TestGPU_Dummy_interface_gpu_kernels_h

//#include <cuda_runtime.h>

namespace testgpu {

// simple test 
void launch_on_gpu();

// allocate 
template<int NUM_OF_VALUES, typename T>
void allocate(T**);

// copy to GPU (true) / from (false)
template<int NUM_OF_VALUES, typename T>
void copy(T* /* h_values */, T* /* d_values */,  bool /* direction */);

// a wrapper for the kernel for vector addition
template<int NUM_OF_VALUES, typename T>
void wrapperVectorAdd(T*, T*, T*);

// free the memory on the device
template<typename T>
void release(T*);

}

#endif
