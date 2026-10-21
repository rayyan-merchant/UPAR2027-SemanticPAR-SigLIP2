# DINOv3 runtime report: short GPU gate passed

DINOv2 cancellation is terminal CANCEL_ACKNOWLEDGED; its last saved log reached frozen extraction at119s. No completed DINOv2 result is claimed. Quota10.97h before DINO work,10.85h after cancellation,10.83h after the new gate; display rounding prevents exact charge attribution.

Private upar2027-dinov3-timm-feasibility v1 COMPLETE, execution102.32s. Actual native/timm FP32 and AMP token/descriptor max-absolute and relative-L2 differences0.0. Torch2.11.0+cu128/CUDA12.8/timm1.0.24; twoTeslaT4,4CPUcores. Same native implementation also passed Python3.11/torch2.4.1CPU offline parity; official GPU deployment remains to be benchmarked on the selected trained candidate.

Concurrent independent workers each timed2048images after2warm-up batches on256unique source crops:286.133/288.309images/s,peak allocated0.667GiB each,batch64,workers2/threads2. Mode includes decode,aspect-preserving preprocessing,nativeAMPforward,six-descriptor pooling. Warmed repeated-source measurements do not certify cold full-cache throughput. CPU reached100%,GPUs typically96-99%,loader wait0.034/0.024s across~7.1timed seconds; no idleGPU I/O starvation observed in this short test.

Aggregate full-extraction projection228s,342s with50percent margin; conservative single-worker projection458s. Cache131076x6x768FP16=1,207,996,416payload bytes,8GiB working-space gate. Estimated standalone29248-image decode/forward/pool102s is not an exact trained API/ensemble/A10 benchmark. Full deployment objective<=240s still requires selected trained weights,loading,decoding,bothbranches ifapplicable,andserialization.

Measurements,download/hash/license and parity were reported to the user before full extraction. Full frozen diagnostic is now prepared with provider6600s/internal6300s,within the original cumulative2GPU-sessionh diagnostic cap after~.14h prior spending. Estimate~6minutes extraction plus setup/cached heads; plan30-60minutes,hard1.83h additional. Preserve about9GPUhours. No optional fine-tuning,all-domain training or upload is in this job.

Reuse the four opaque archives once in each fresh ephemeral session; wholearchiveSHA/CRC/exactunion,stage120s/idle30s,loaderI/O30s,totalpretraining600s. Never copy/hash individual mounted images. Actual full-extraction rates and independent head timing will replace projections when saved evidence exists. Original square-root sampler/source-only balancedBCE/selection/calibration are retained. Separate optimizer/RNG/checkpoints per head,PA0/PETA1 thenMarket0.

Predeclared parity tolerances remain FP32max_abs<=5e-4/relL2<=1e-5 andAMPmax_abs<=.05/relL2<=.005; no post-result relaxation. Gatecode/weights/runtime hashes pinned before reuse. Short source-only throughput check repeats in the extraction session and stops below75images/s or above12GiB or if projected extraction exceeds1800s. Completed source/target locks must never be refit or reevaluated during recovery.
