#ifndef TestGPU_Dummy_interface_gpu_kernels_h
#define TestGPU_Dummy_interface_gpu_kernels_h

//#include <cuda_runtime.h>

namespace testgpu {

// simple test 
void launch_on_gpu();

// allocate 
template<int NUM_OF_VALUES, typename T = int>
void allocate(T*);

// copy to GPU (true) / from (false)
template<int NUM_OF_VALUES, bool TOGPU = true, typename T>
void copy(T*, T*);

template<int NUM_OF_VALUES, typename T>
void wrapperVectorAdd(T*, T*, T*);

template<typename T>
void release(T*);

}

#endif
