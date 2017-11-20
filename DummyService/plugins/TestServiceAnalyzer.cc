// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TestGPU/DummyService/interface/AService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cuda.h>
#include <cuda_runtime.h>

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class TestServiceAnalyzer : public edm::stream::EDAnalyzer<>  {
   public:
      explicit TestServiceAnalyzer(const edm::ParameterSet&);
      ~TestServiceAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
//      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
//      virtual void endJob() override;

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
TestServiceAnalyzer::TestServiceAnalyzer(const edm::ParameterSet& iConfig)
{
}


TestServiceAnalyzer::~TestServiceAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
TestServiceAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Service<AService> aservice;
   aservice->print();
}

/*
// ------------ method called once each job just before starting event loop  ------------
void 
TestServiceAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestServiceAnalyzer::endJob() 
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TestServiceAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestServiceAnalyzer);
