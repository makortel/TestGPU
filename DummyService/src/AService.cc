// otherwise fails to compile
//#undef __CUDACC_VER__
//#define __CUDACC_VER__ ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))

#include "TestGPU/DummyService/interface/AService.h"

// for the macros
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

DEFINE_FWK_SERVICE(AService);
