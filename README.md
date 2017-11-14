# TestGPU - Test Cuda GPU workflows within cmssw and in standalone


## Setup on felk40.cern.ch
- Pascal Architecture (Compute Capability 6.1)
- export SCRAM\_ARCH=slc6\_amd64\_gcc630 
- cmsrel CMSSW\_9\_4\_0\_pre2
- cd CMSSW\_9\_4\_0\_pre2/src
- cmsenv
- git clone https://github.com/vkhristenko/TestGPU
- scram setup TestGPU/Dummy/config/cuda\_9.0\_pascal.xml
- scram b -v -j 8

## Dummy Standalone Test
- Execute `standalone_test0` (available in the PATH)
- 2 Vector Addition
- Output should show first 10 elements:
```
...
c[0] = 0
c[1] = 2
c[2] = 6
c[3] = 12
c[4] = 20
c[5] = 30
c[6] = 42
c[7] = 56
c[8] = 72
c[9] = 90
```

## Dummy CMSSW Analyzer
- Execute `cmsRun TestGPU/Dummy/python/test_oneanalyzer_gpu.py`
- 2 Vector Addition but being called from within the edm::one::EDAnalyzer
- Output should show for each event:
```
c[0] = 0
c[1] = 2
c[2] = 6
c[3] = 12
c[4] = 20
c[5] = 30
c[6] = 42
c[7] = 56
c[8] = 72
c[9] = 90
```

## Dummy CMSSW Producer
- Execute `cmsRun TestGPU/Dummy/python/test_oneproducer_gpu.p`
- 2 Vector addition. Memory Allocation / Transfer / Kernel Launching / Memory Freeing are launched from edm::one::EDProducer  - __note: it's just an example - edm::one modules are not efficient__
- stdout should show for each event:
```
c[0] = 1
c[1] = 3
c[2] = 5
c[3] = 7
c[4] = 9
c[5] = 11
c[6] = 13
c[7] = 15
c[8] = 17
c[9] = 19
```
- a file `test_oneproducer.root` should be produced in the directory from which you run.
```
Events -> inttestgpuVector_testGPU_VectorForGPU_TestGPU is the new branch that contains the data generated on the GPU.
```

## Dummy Standalone that uses the same kernel as CMSSW EDAnalyzer
- Execute `standalone_test1`
- The output should be identical to the above... but printed out only once
