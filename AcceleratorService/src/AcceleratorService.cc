#include "TestGPU/AcceleratorService/interface/AcceleratorService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

AcceleratorService::AcceleratorService(const edm::ParameterSet& iConfig, edm::ActivityRegistry& iAR) {
  edm::LogWarning("AcceleratorService") << "Constructor";
}

void AcceleratorService::print() {
  edm::LogPrint("AcceleratorService") << "Hello world";
}

// for the macros
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using AcceleratorServiceMaker = edm::serviceregistry::AllArgsMaker<AcceleratorService>;
DEFINE_FWK_SERVICE_MAKER(AcceleratorService, AcceleratorServiceMaker);
