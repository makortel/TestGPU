// -*- C++ -*-
//
// Package:    TestGPU/Dummy
// Class:      Raw2Digi
// 
/**\class Raw2Digi Raw2Digi.cc TestGPU/Dummy/plugins/Raw2Digi.cc

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
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class Raw2Digi : public edm::stream::EDProducer<> {
   public:
      explicit Raw2Digi(const edm::ParameterSet&);
      ~Raw2Digi();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<FEDRawDataCollection> m_tFEDRawDataCollection;

      std::vector<unsigned int> m_fedIds;
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
Raw2Digi::Raw2Digi(const edm::ParameterSet& iConfig)
{
    // init the token
    m_tFEDRawDataCollection = consumes<FEDRawDataCollection>(
        iConfig.getParameter<edm::InputTag>("InputLabel"));
}


Raw2Digi::~Raw2Digi()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
Raw2Digi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // get conditions
    if (recordWatcher.check(iSetup)) {
        edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
        es.get<SiPixelFedCablingMapRcd>().get( cablingMapLabel, cablingMap );
        m_fedIds   = cablingMap->fedIds();
    }

    // get the collection of raw buffers
    edm::Handle<FEDRawDataCollection> fedRawDataCollection;
    iEvent.getByToken(m_tFEDRawDataCollection, fedRawDataCollection);

    // initialize the collections to be put into the edm::Event
    auto collection = std::make_unique<edm::DetSetVector<PixelDigi>>();

    // For each Pixel FED
    for (auto it = m_fedIds.begin(); it!=m_fedIds.end(); it++) {
        int fed = *it;
        printf("fed = %d\n", fed);
    }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
Raw2Digi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Raw2Digi);
