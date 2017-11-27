#ifndef TestGPU_AcceleratorService_AcceleratorService_h
#define TestGPU_AcceleratorService_AcceleratorService_h

#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>
#include <mutex>

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ModuleDescription;
  namespace service {
    class SystemBounds;
  }
}

class AcceleratorService {
public:
  class Token {
  public:
    explicit Token(unsigned int id): id_(id) {}

    unsigned int id() const { return id_; }
  private:
    unsigned int id_;
  };


  AcceleratorService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);

  Token book(); // TODO: better name, unfortunately 'register' is a reserved keyword...

  void async(Token token, edm::StreamID streamID, std::function<int(void)> task);

  int result(Token token, edm::StreamID streamID) const {
    return data_[tokenStreamIdsToDataIndex(token.id(), streamID)];
  }

  void print();

private:
  // signals
  void preallocate(edm::service::SystemBounds const& bounds);
  void preModuleConstruction(edm::ModuleDescription const& desc);
  void postModuleConstruction(edm::ModuleDescription const& desc);


  // other helpers
  unsigned int tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const;

  unsigned int numberOfStreams_ = 0;

  // nearly (if not all) happens multi-threaded, so we need some
  // thread-locals to keep track in which module we are
  static thread_local unsigned int currentModuleId_;

  // TODO: how to treat subprocesses?
  std::mutex moduleMutex_;
  std::vector<unsigned int> moduleIds_; // list of module ids that have registered something on the service
  std::vector<int> data_;      // numberOfStreams x moduleIds_.size(), indexing defined by moduleStreamIdsToDataIndex
};

#endif
