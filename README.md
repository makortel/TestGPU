# TestGPU - test gpu workflows within cmssw

## Requirements 
- machine with GPU (should be compilable though...)
- nvcc installed
- __proper gcc versions for nvcc__

## Setup on felk40.cern.ch
- export SCRAM\_ARCH=slc6\_amd64\_gcc530 
- cmsrel CMSSW\_9\_1\_2 
- cd CMSSW\9\_1\_2/src
- cmsenv
- scram setup /home/fpantale/cuda\_8.0.27.xml
- git clone https://github.com/vkhristenko/TestGPU
- scram b -v -j 8


