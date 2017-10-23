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
- Execute standalone\_test0 (available in the PATH)
- 2 Vector Addition
- Output should show first 10 elements:
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

## Dummy CMSSW Test
- Execute `cmsRun TestGPU/Dummy/python/test_gpu.py`
- 2 Vector Addition but being called from within the edm::one::EDAnalyzer
- Output should show for each event:
```
Adding Vector element: c[4000] = i*i + i = 16004000
Adding Vector element: c[8000] = i*i + i = 64008000
Adding Vector element: c[0] = i*i + i = 0
Adding Vector element: c[5000] = i*i + i = 25005000
Adding Vector element: c[9000] = i*i + i = 81009000
Adding Vector element: c[1000] = i*i + i = 1001000
Adding Vector element: c[10000] = i*i + i = 0
Adding Vector element: c[2000] = i*i + i = 4002000
Adding Vector element: c[6000] = i*i + i = 36006000
Adding Vector element: c[3000] = i*i + i = 9003000
Adding Vector element: c[7000] = i*i + i = 49007000
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
