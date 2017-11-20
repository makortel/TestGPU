# Proposal `0` for implementing/integrating the CUDA enabled EDM Plugins

__Goal:__ I believe the goal for this iteration is to clearly establish:
- how GPU code should be structured/compiled/linked within CMSSW
- test memory management techniques (*potentially identify the interfaces for the future to be used by the Service*)
- Concurrency on the host with the concurrency on the GPU (e.g. how many GPU streams per 1 CPU stream, etc...)
- which api to use (Runtime or Driver). For now __Runtime__
- which interfaces are handy and should be introduced to the Accelerator Service.
- __Most importantly, exercise the procedure for RAW -> DIGI conversion__
- __Establish valid complex examples__

## Implementation Assumptions:
- No Accelerator Service for now -> Exclude it for this iteration 
- __Assume__ a single CUDA-enabled device present, or a default device.
- __Assume__ that CMSSW streams are independent - no synching.
- __Assume__ a constant number of streams per cpu thread.

## Implementation Details:
- __No Globals__ - current impolementation has a bunch of global variables: __This is not thread safe__
```
//CablingMap *Map;
//GPU specific
uint *word_d, *fedIndex_d, *eventIndex_d;       // Device copy of input data
uint *xx_d, *yy_d,*xx_adc, *yy_adc, *moduleId_d, *adc_d, *layer_d;  // Device copy
// store the start and end index for each module (total 1856 modules-phase 1)
cudaStream_t stream[NSTREAM];
int *mIndexStart_d, *mIndexEnd_d; 
CablingMap *Map;
```
- Limit the amount of copying by copying/allocating in chuncks. __Watch out for the proper alignment__
- Compile with `--default-stream per-thread` (done already, just for reference)
- Separate CUDA kernels into a separate interface/src folders and make them part of the shared library to make them testable in standalone (*Might be changed in the future*)

## RAW -> DIGI Specifics:
- Emulate the loading of conditions.
- Rearrange things slight/make code more readable.
- Remove the globals
- Add the `event.put()` - inclusion of the computed collections into the event.

__pros:__
- __Simplify - Memory to be managed directly by the plugin__. Currently, not clear what exactly service should be doing yet (besides counting the number of devices, managing memory)
- Will allow to quickly adapt the PRs `https://github.com/cms-sw/cmssw/pull/21321` or `https://github.com/cms-sw/cmssw/pull/21341` and put them

__cons:__
- Not final version and will need to be sufficiently improved.
