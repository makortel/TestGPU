#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TestGPU/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class TestAcceleratorServiceProducer: public edm::stream::EDProducer<> {
public:
  explicit TestAcceleratorServiceProducer(const edm::ParameterSet& iConfig);
  ~TestAcceleratorServiceProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
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


TestAcceleratorServiceProducer::TestAcceleratorServiceProducer(const edm::ParameterSet& iConfig) {
  produces<int>();
}

void TestAcceleratorServiceProducer::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID();
  edm::Service<AcceleratorService> acc;
  acc->print();
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestAcceleratorServiceProducer::produceReal(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::produceReal begin event " << iEvent.id().event() << " stream " << iEvent.streamID();
  auto ret = std::make_unique<int>(iEvent.id().event());
  iEvent.put(std::move(ret));
  edm::LogPrint("Foo") << " TestAcceleratorServiceProducer::produceReal end event " << iEvent.id().event() << " stream " << iEvent.streamID();
}

void TestAcceleratorServiceProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("testAcceleratorServiceProducer", desc);
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducer);
