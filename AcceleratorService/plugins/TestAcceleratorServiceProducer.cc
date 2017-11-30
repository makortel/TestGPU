#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TestGPU/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <thread>
#include <random>
#include <chrono>

namespace {
  class TestTask: public AcceleratorTask<accelerator::CPU> {
  public:
    TestTask(unsigned int eventId, unsigned int streamId):
      eventId_(eventId), streamId_(streamId) {}
    ~TestTask() override = default;

    void run_CPU() override {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(0.1, 2.0); 
      auto dur = dist(gen);
      edm::LogPrint("Foo") << "    Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";

      output_ = streamId_*100 + eventId_;
    }

    unsigned int getOutput() const { return output_; }

  private:
    // input
    unsigned int eventId_;
    unsigned int streamId_;

    // output
    unsigned int output_;
  };
}

class TestAcceleratorServiceProducer: public edm::stream::EDProducer<> {
public:
  explicit TestAcceleratorServiceProducer(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  AcceleratorService::Token accToken_;


  // to mimic external task worker interface
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void produceReal(edm::Event& iEvent, const edm::EventSetup& iSetup);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    edm::LogPrint("Foo") << "TestAcceleratorServiceProducer::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID();
    acquire(iEvent, iSetup);
    produceReal(iEvent, iSetup);
    edm::LogPrint("Foo") << "TestAcceleratorServiceProducer::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID() << "\n";
  }
};


TestAcceleratorServiceProducer::TestAcceleratorServiceProducer(const edm::ParameterSet& iConfig):
  accToken_(edm::Service<AcceleratorService>()->book())
{
  produces<int>();
}

void TestAcceleratorServiceProducer::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID();
  edm::Service<AcceleratorService> acc;
  acc->async(accToken_, iEvent.streamID(), std::make_unique<::TestTask>(iEvent.id().event(), iEvent.streamID()));
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestAcceleratorServiceProducer::produceReal(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::produceReal begin event " << iEvent.id().event() << " stream " << iEvent.streamID();
  edm::Service<AcceleratorService> acc;
  auto ret = std::make_unique<int>(dynamic_cast<const ::TestTask&>(acc->getTask(accToken_, iEvent.streamID())).getOutput());
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::produceReal end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << *ret;
  iEvent.put(std::move(ret));
}

void TestAcceleratorServiceProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("testAcceleratorServiceProducer", desc);
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducer);
