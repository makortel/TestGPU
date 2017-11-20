#ifndef TestGPU_DummyService_interface_AService_h
#define TEstGPU_DummyService_interface_AService_h

#include <stdio.h>

namespace edm {
class ParameterSet;
class ActivityRegistry;
}

// https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMServiceDocWriting
class AService {
public:
    AService() = default;
    AService(edm::ParameterSet const& ps,
             edm::ActivityRegistry& iAR) 
    {
        printf("AService::ctor has been called\n");
    }

    void print(void) { printf("Hello World from a Service\n"); }
};

#endif
