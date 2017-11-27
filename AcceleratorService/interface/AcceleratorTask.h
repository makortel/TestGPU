#ifndef TestGPU_AcceleratorService_AcceleratorTask_h
#define TestGPU_AcceleratorService_AcceleratorTask_h

namespace accelerator {
  // TODO: Would a diamond inheritance be simpler?

  // similar to FWCore/Framework/interface/moduleAbilityEnums.h
  enum class Capabilities {
    kCPU,
    kGPU_cuda
  };
  
  namespace CapabilityBits {
    enum Bits {
      kCPU = 1,
      kGPU_cuda = 2
    };
  }

  // similar to e.g. FWCore/Framework/interface/one/moduleAbilities.h
  struct CPU {
    static constexpr Capabilities kCapability = Capabilities::kCPU;
  };

  struct GPU_cuda {
    static constexpr Capabilities kCapability = Capabilities::kGPU_cuda;
  };

  // similar to e.g. FWCore/Framework/interface/one/implementors.h
  namespace impl {
    
  }

  // similar to e.g. FWCore/Framework/interface/one/producerAbilityToImplementor.h
  template <typename T> struct CapabilityToImplementor;
  
  template<>
  CapabilityToImplementor<CPU> {
    using impl::CPU Type;
  };

  template<>
  CapabilityToImplementor<GPU_cuda> {
    using impl::GPU_cuda Type;
  };
}


class AcceleratorTaskBase {
public:
  AcceleratorTaskBase() = default;
  virtual ~AcceleratorTaskBase() = 0;
};

namespace accelerator {
}

class AcceleratorCPU {
};

class AcceleratorGPU {
};

template <typename ... T>
class AcceleratorTask:
  public AcceleratorTaskBase

 {
  
};



#endif
