import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("TestGPU")

# conditions
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')

# debugging/messaging
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring("*")
)

# n events only
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# Source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("/store/relval/CMSSW_9_4_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_upgrade2018_realistic_v5-v1/10000/F87005CD-CBC8-E711-A9F5-0CC47A4D7694.root"))

# which module to run
process.testGPU = cms.EDProducer('Raw2Digi',
                                 InputLabel = cms.InputTag("rawDataCollector"))

process.p = cms.Path(process.testGPU)

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("test_raw2digi.root")
)
process.finalize = cms.EndPath(process.out)
#process.options = cms.untracked.PSet(
#    numberOfThreads = cms.untracked.uint32(4),
#    numberOfStreams = cms.untracked.uint32(4)
#)
