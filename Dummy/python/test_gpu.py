import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("TestGPU")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring("*")
#        'streamAnalyzer',
#        "globalAnalyzer")
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

pathToFiles = "root://eoscms.cern.ch//eos/cms/store/data/Run2016H/SingleMuon/MINIAOD/03Feb2017_ver2-v1/80000/"
pathToFiles = "file:/afs/cern.ch/work/v/vkhriste/data/cmssw/test"

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        os.path.join(pathToFiles, "FC557485-9FEA-E611-B9B1-1CB72C0A3A5D.root"),
#        os.path.join(pathToFiles, "EAD81DEF-C0EA-E611-AA91-002590E39D8A.root")
    )
)

process.testGPU = cms.EDAnalyzer('Dummy')

process.p = cms.Path(process.testGPU)
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(4)
)
