import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.options = cms.untracked.PSet(
#    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(0)
)


process.AcceleratorService = cms.Service("AcceleratorService")
process.producer = cms.EDProducer('TestAcceleratorServiceProducer')

#process.t = cms.Task(process.producer)

process.eca = cms.EDAnalyzer("EventContentAnalyzer",
    getData = cms.untracked.bool(True),
    getDataForModuleLabels = cms.untracked.vstring("producer"),
    listContent = cms.untracked.bool(True),
)
process.p = cms.Path(process.producer)#+process.eca)
#process.p.associate(process.t)
