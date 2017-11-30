#ifndef TestGPU_AcceleratorService_AcceleratorTask_h
#define TestGPU_AcceleratorService_AcceleratorTask_h

class AcceleratorTaskBase {
public:
  AcceleratorTaskBase() = default;
  virtual ~AcceleratorTaskBase() = 0;

  // CPU functions
  virtual bool runnable_CPU() const { return false; }
  virtual void call_run_CPU() {}


  // GPU functions
  virtual bool runnable_GPUCuda() const { return false; }
  virtual void call_run_GPUCuda() {}
  virtual void call_copyToCPU_GPUCuda() {}
};

namespace accelerator {
  // Below we could assume that the CPU would be always present. For
  // the sake of demonstration I'll keep it separate entity.

  // similar to FWCore/Framework/interface/moduleAbilityEnums.h
  enum class Capabilities {
    kCPU,
    kGPUCuda
  };
  
  namespace CapabilityBits {
    enum Bits {
      kCPU = 1,
      kGPUCuda = 2
    };
  }

  // similar to e.g. FWCore/Framework/interface/one/moduleAbilities.h
  struct CPU {
    static constexpr Capabilities kCapability = Capabilities::kCPU;
  };

  struct GPUCuda {
    static constexpr Capabilities kCapability = Capabilities::kGPUCuda;
  };

  // similar to e.g. FWCore/Framework/interface/one/implementors.h
  namespace impl {
    class CPU: public virtual AcceleratorTaskBase {
    public:
      CPU() = default;
      bool runnable_CPU() const override { return true; }

    private:
      void call_run_CPU() override {
        run_CPU();
      };

      virtual void run_CPU() = 0;
    };

    class GPUCuda: public virtual AcceleratorTaskBase {
    public:
      GPUCuda() = default;
      bool runnable_GPUCuda() const override { return true; }

    private:
      void call_run_GPUCuda() override {
        run_GPUCuda();
      };

      void call_copyToCPU_GPUCuda() override {
        copyToCPU_GPUCuda();
      };

      virtual void run_GPUCuda() = 0;
      virtual void copyToCPU_GPUCuda() = 0;
    };
  }

  // similar to e.g. FWCore/Framework/interface/one/producerAbilityToImplementor.h
  template <typename T> struct CapabilityToImplementor;
  
  template<>
  struct CapabilityToImplementor<CPU> {
    using Type = impl::CPU;
  };

  template<>
  struct CapabilityToImplementor<GPUCuda> {
    using Type = impl::GPUCuda;
  };
}

template <typename ... T>
class AcceleratorTask:
  public virtual AcceleratorTaskBase,
  public accelerator::CapabilityToImplementor<T>::Type... {

 public:
  AcceleratorTask() = default;
  
};


#endif
