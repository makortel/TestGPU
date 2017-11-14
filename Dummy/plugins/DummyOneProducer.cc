// -*- C++ -*-
//
// Package:    TestGPU/Dummy
// Class:      DummyOneProducer
// 
/**\class DummyOneProducer DummyOneProducer.cc TestGPU/Dummy/plugins/DummyOneProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Viktor Khristenko
//         Created:  Tue, 14 Nov 2017 10:33:24 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// use the new product
#include "TestGPU/Dummy/interface/Vector.h"
#include "TestGPU/Dummy/interface/gpu_kernels.h"

#define NUM_VALUES 10000

//
// class declaration
//

class DummyOneProducer : public edm::one::EDProducer<> {
   public:
      explicit DummyOneProducer(const edm::ParameterSet&);
      ~DummyOneProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
DummyOneProducer::DummyOneProducer(const edm::ParameterSet& iConfig)
{
    testgpu::Vector<int> v;

    produces<testgpu::Vector<int> >("VectorForGPU");

/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
 
   //if you want to put into the Run
   produces<ExampleData2,InRun>();
*/
   //now do what ever other initialization is needed
  
}


DummyOneProducer::~DummyOneProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DummyOneProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
    // 
    // initialize vars on the host's side
    //
    int h_a[NUM_VALUES], h_b[NUM_VALUES], h_c[NUM_VALUES];
    for (auto i=0; i<NUM_VALUES; i++) {
        h_a[i] = i;
        h_b[i] = i+1;
    }

    //
    // allocate memory on the GPU (device) side. Just a wrapper around the cudaMalloc
    //
    int *d_a, *d_b, *d_c;
    testgpu::allocate<NUM_VALUES>(&d_a);
    testgpu::allocate<NUM_VALUES>(&d_b);
    testgpu::allocate<NUM_VALUES>(&d_c);

    //
    // copy arrays from Host to Device (true)
    //
    testgpu::copy<NUM_VALUES>(h_a, d_a, true);
    testgpu::copy<NUM_VALUES>(h_b, d_b, true);

    //
    // launch kernel
    //
    testgpu::wrapperVectorAdd<NUM_VALUES>(d_a, d_b, d_c);

    //
    // copy data back
    //
    testgpu::copy<NUM_VALUES>(h_c, d_c, false);

    //
    // free data on GPU
    //
    testgpu::release(d_a);
    testgpu::release(d_b);
    testgpu::release(d_c);

    // 
    // print 
    //
    for (auto i=0; i<10; i++)
        printf("c[%d] = %d\n", i, h_c[i]);
    
    // 
    // put into the edm::Event
    //
    testgpu::Vector<int> v1;
    v1.m_values = std::vector<int>(h_c, h_c + NUM_VALUES);
    iEvent.put(std::make_unique<testgpu::Vector<int> >(v1), "VectorForGPU");

}

void
DummyOneProducer::beginJob()
{
}

void
DummyOneProducer::endJob() {
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DummyOneProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DummyOneProducer);
