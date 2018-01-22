#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TestGPU/AcceleratorService/interface/AcceleratorService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TestGPU/AcceleratorService/interface/TestProxyProduct.h"

#include <thread>
#include <random>
#include <chrono>

namespace {
  class TestTask: public AcceleratorTask<accelerator::CPU, accelerator::GPUCuda> {
  public:
    TestTask(int input, unsigned int eventId, unsigned int streamId):
      input_(input), eventId_(eventId), streamId_(streamId) {}
    ~TestTask() override = default;

    void run_CPU() override {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(1.0, 3.0); 
      auto dur = dist(gen);
      edm::LogPrint("Foo") << "   Task (CPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";

      output_ = input_ + streamId_*100 + eventId_;
    }

    void run_GPUCuda() override {
      std::random_device r;
      std::mt19937 gen(r());
      auto dist = std::uniform_real_distribution<>(0.1, 1.0); 
      auto dur = dist(gen);
      edm::LogPrint("Foo") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " will take " << dur << " seconds";

      gpuOutput_ = input_ + streamId_*100 + eventId_;
    }

    void copyToCPU_GPUCuda() override {
      edm::LogPrint("Foo") << "   Task (GPU) for event " << eventId_ << " in stream " << streamId_ << " copying to CPU";
      output_ = gpuOutput_;
    }

    unsigned int getOutput() const { return output_; }

  private:
    // input
    int input_;
    unsigned int eventId_;
    unsigned int streamId_;

    // simulating GPU memory
    unsigned int gpuOutput_;

    // output
    unsigned int output_;
  };
}

class TestAcceleratorServiceProducer: public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit TestAcceleratorServiceProducer(edm::ParameterSet const& iConfig);
  ~TestAcceleratorServiceProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string label_;
  AcceleratorService::Token accToken_;

  edm::EDGetTokenT<TestProxyProduct> srcToken_;

  // to mimic external task worker interface
  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTask) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
};


TestAcceleratorServiceProducer::TestAcceleratorServiceProducer(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  accToken_(edm::Service<AcceleratorService>()->book())
{
  auto srcTag = iConfig.getParameter<edm::InputTag>("src");
  if(!srcTag.label().empty()) {
    srcToken_ = consumes<TestProxyProduct>(srcTag);
  }

  produces<TestProxyProduct>();
}

void TestAcceleratorServiceProducer::acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  int input = 0;
  if(!srcToken_.isUninitialized()) {
    edm::Handle<TestProxyProduct> hint;
    iEvent.getByToken(srcToken_, hint);
    input = hint->value();
  }

  edm::LogPrint("Foo") << "TestAcceleratorServiceProducer::acquire begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " input " << input;
  edm::Service<AcceleratorService> acc;
  acc->async(accToken_, iEvent.streamID(), std::make_unique<::TestTask>(input, iEvent.id().event(), iEvent.streamID()), std::move(waitingTaskHolder));
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducer::acquire end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
}

void TestAcceleratorServiceProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducer::produce begin event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_;
  edm::Service<AcceleratorService> acc;
  auto value = dynamic_cast<const ::TestTask&>(acc->getTask(accToken_, iEvent.streamID())).getOutput();
  auto ret = std::make_unique<TestProxyProduct>(value);
  edm::LogPrint("Foo") << "TestAcceleratorServiceProducer::produce end event " << iEvent.id().event() << " stream " << iEvent.streamID() << " label " << label_ << " result " << value;
  iEvent.put(std::move(ret));
}

void TestAcceleratorServiceProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag());
  descriptions.add("testAcceleratorServiceProducer", desc);
}

DEFINE_FWK_MODULE(TestAcceleratorServiceProducer);
