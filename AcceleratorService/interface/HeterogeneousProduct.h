#ifndef TestGPU_AcceleratorService_interface_HeterogeneousData_h
#define TestGPU_AcceleratorService_interface_HeterogeneousData_h

#include "TestGPU/AcceleratorService/interface/AcceleratorTask.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <bitset>
#include <mutex>

enum class HeterogeneousLocation {
  kCPU = 0,
  kGPU,
  kSize
};

template <typename CPUProduct, typename GPUProduct>
class HeterogeneousProduct {
public:
  using TaskGetter = std::function<const AcceleratorTaskBase *(void)>;
  using TransferCallback = std::function<void(const GPUProduct&, CPUProduct&)>;

  HeterogeneousProduct() = default;
  HeterogeneousProduct(CPUProduct&& data):
    cpuProduct_(std::move(data)) {
    location_.set(static_cast<unsigned int>(HeterogeneousLocation::kCPU));
  }
  HeterogeneousProduct(GPUProduct&& data, TransferCallback transfer):
    gpuProduct_(std::move(data)),
    transfer_(std::move(transfer)) {
    location_.set(static_cast<unsigned int>(HeterogeneousLocation::kGPU));
  }

  bool isProductOn(HeterogeneousLocation loc) const {
    return location_[static_cast<unsigned int>(loc)];
  }

  const CPUProduct& getCPUProduct() const {
    if(!isProductOn(HeterogeneousLocation::kCPU)) transferToCPU();
    return cpuProduct_;
  }

  const GPUProduct& getGPUProduct() const {
    if(!isProductOn(HeterogeneousLocation::kGPU))
      throw cms::Exception("LogicError") << "Called getGPUProduct(), but the data is not on GPU! Location bitfield is " << location_.to_string();
    return gpuProduct_;
  }

private:
  void transferToCPU() const {
    if(!isProductOn(HeterogeneousLocation::kGPU)) {
      throw cms::Exception("LogicError") << "Called transferToCPU, but the data is not on GPU! Location bitfield is " << location_.to_string();
    }
    
    std::lock_guard<std::mutex> lk(mutex_);
    transfer_(gpuProduct_, cpuProduct_);
    location_.set(static_cast<unsigned int>(HeterogeneousLocation::kCPU));
  }
  
  mutable std::mutex mutex_;
  mutable CPUProduct cpuProduct_;
  GPUProduct gpuProduct_;
  mutable std::bitset<static_cast<unsigned int>(HeterogeneousLocation::kSize)> location_;
  TransferCallback transfer_;
};


#endif
