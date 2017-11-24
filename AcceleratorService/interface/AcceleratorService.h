#ifndef TestGPU_AcceleratorService_AcceleratorService_h
#define TestGPU_AcceleratorService_AcceleratorService_h

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
}

class AcceleratorService {
public:
  AcceleratorService(const edm::ParameterSet& iConfig, edm::ActivityRegistry& iAR);

  void print();

};

#endif
