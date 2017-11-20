import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("TestAService")

process.AService = cms.Service("AService")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring(
        'cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring("*")
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("EmptySource")

process.testAService = cms.EDAnalyzer('TestServiceAnalyzer')

process.p = cms.Path(process.testAService)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(4)
)
