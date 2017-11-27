#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TestGPU/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <thread>
#include <random>
#include <chrono>

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
  acc->async(accToken_, iEvent.streamID(), [&]() {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(0.1, 2.0); 
      auto dur = dist(gen);
      edm::LogPrint("Foo") << "    Task for event " << iEvent.id().event() << " in stream " << iEvent.streamID() << " will take " << dur << " seconds";
      std::this_thread::sleep_for(std::chrono::seconds(1)*dur);
      return iEvent.streamID()*100 + iEvent.id().event();
    });
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestAcceleratorServiceProducer::produceReal(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::produceReal begin event " << iEvent.id().event() << " stream " << iEvent.streamID();
  edm::Service<AcceleratorService> acc;
  auto ret = std::make_unique<int>(acc->result(accToken_, iEvent.streamID()));
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::produceReal end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " result " << *ret;
  iEvent.put(std::move(ret));
}

void TestAcceleratorServiceProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("testAcceleratorServiceProducer", desc);
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducer);
