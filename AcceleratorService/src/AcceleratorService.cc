#include "TestGPU/AcceleratorService/interface/AcceleratorService.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <limits>
#include <algorithm>
#include <thread>
#include <random>
#include <chrono>
#include <cassert>

thread_local unsigned int AcceleratorService::currentModuleId_ = std::numeric_limits<unsigned int>::max();

AcceleratorService::AcceleratorService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
  edm::LogWarning("AcceleratorService") << "Constructor";

  iRegistry.watchPreallocate(           this, &AcceleratorService::preallocate );
  iRegistry.watchPreModuleConstruction (this, &AcceleratorService::preModuleConstruction );
  iRegistry.watchPostModuleConstruction(this, &AcceleratorService::postModuleConstruction );
}

// signals
void AcceleratorService::preallocate(edm::service::SystemBounds const& bounds) {
  numberOfStreams_ = bounds.maxNumberOfStreams();
  edm::LogPrint("Foo") << "AcceleratorService: number of streams " << numberOfStreams_;
  // called after module construction, so initialize data_ here
  data_.insert(data_.end(), moduleIds_.size()*numberOfStreams_, 0);
}

void AcceleratorService::preModuleConstruction(edm::ModuleDescription const& desc) {
  currentModuleId_ = desc.id();
}
void AcceleratorService::postModuleConstruction(edm::ModuleDescription const& desc) {
  currentModuleId_ = std::numeric_limits<unsigned int>::max();
}


// actual functionality
AcceleratorService::Token AcceleratorService::book() {
  if(currentModuleId_ == std::numeric_limits<unsigned int>::max())
    throw cms::Exception("AcceleratorService") << "Calling AcceleratorService::register() outside of EDModule constructor is forbidden.";

  unsigned int index=0;

  std::lock_guard<std::mutex> guard(moduleMutex_);

  auto found = std::find(moduleIds_.begin(), moduleIds_.end(), currentModuleId_);
  if(found == moduleIds_.end()) {
    index = moduleIds_.size();
    moduleIds_.push_back(currentModuleId_);
  }
  else {
    index = std::distance(moduleIds_.begin(), found);
  }

  edm::LogPrint("Foo") << "AcceleratorService::book for module " << currentModuleId_ << " token id " << index << " moduleIds_.size() " << moduleIds_.size();

  return Token(index);
}

void AcceleratorService::async(Token token, edm::StreamID streamID, std::function<int(void)> task) {
  // not really async but let's "simulate"

  edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launching thread";
  auto asyncThread = std::thread([=](){
      edm::LogPrint("Foo") << "   AcceleratorService token " << token.id() << " stream " << streamID << " launching task";
      data_[tokenStreamIdsToDataIndex(token.id(), streamID)] = task();
      edm::LogPrint("Foo") << "   AcceleratorService token " << token.id() << " stream " << streamID << " task finished";
    });
  std::random_device r;
  std::mt19937 gen(r());
  auto dist = std::uniform_real_distribution<>(0.1, 5.0); 
  auto dur = dist(gen);
  edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " doing something else for some time (" << dur << " seconds)";
  std::this_thread::sleep_for(std::chrono::seconds(1)*dur);
  asyncThread.join();
  edm::LogPrint("Foo") << "  AcceleratorService token " << token.id() << " stream " << streamID << " async finished";
}

void AcceleratorService::print() {
  edm::LogPrint("AcceleratorService") << "Hello world";
}

unsigned int AcceleratorService::tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const {
  assert(streamId < numberOfStreams_);
  return tokenId*numberOfStreams_ + streamId;
}

// for the macros
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using AcceleratorServiceMaker = edm::serviceregistry::AllArgsMaker<AcceleratorService>;
DEFINE_FWK_SERVICE_MAKER(AcceleratorService, AcceleratorServiceMaker);
