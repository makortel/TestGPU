#include "TestGPU/Dummy/interface/Vector.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
    struct dictinoary {
        testgpu::Vector<int> vints;
        testgpu::Vector<float> vfloats;
        testgpu::Vector<double> vdoubles;
    };
}
