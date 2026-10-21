# DINOv3 runtime report: short GPU gate passed

DINOv2 cancellation is terminal CANCEL_ACKNOWLEDGED; its last saved log reached frozen extraction at119s. No completed DINOv2 result is claimed. Quota10.97h before DINO work,10.85h after cancellation,10.83h after the new gate; display rounding prevents exact charge attribution.

Private upar2027-dinov3-timm-feasibility v1 COMPLETE, execution102.32s. Actual native/timm FP32 and AMP token/descriptor max-absolute and relative-L2 differences0.0. Torch2.11.0+cu128/CUDA12.8/timm1.0.24; twoTeslaT4,4CPUcores. Same native implementation also passed Python3.11/torch2.4.1CPU offline parity; official GPU deployment remains to be benchmarked on the selected trained candidate.

Concurrent independent workers each timed2048images after2warm-up batches on256unique source crops:286.133/288.309images/s,peak allocated0.667GiB each,batch64,workers2/threads2. Mode includes decode,aspect-preserving preprocessing,nativeAMPforward,six-descriptor pooling. Warmed repeated-source measurements do not certify cold full-cache throughput. CPU reached100%,GPUs typically96-99%,loader wait0.034/0.024s across~7.1timed seconds; no idleGPU I/O starvation observed in this short test.

Aggregate full-extraction projection228s,342s with50percent margin; conservative single-worker projection458s. Cache131076x6x768FP16=1,207,996,416payload bytes,8GiB working-space gate. Estimated standalone29248-image decode/forward/pool102s is not an exact trained API/ensemble/A10 benchmark. Full deployment objective<=240s still requires selected trained weights,loading,decoding,bothbranches ifapplicable,andserialization.

Measurements,download/hash/license and parity were reported to the user before full extraction. Full frozen diagnostic is now prepared with provider6600s/internal6300s,within the original cumulative2GPU-sessionh diagnostic cap after~.14h prior spending. Estimate~6minutes extraction plus setup/cached heads; plan30-60minutes,hard1.83h additional. Preserve about9GPUhours. No optional fine-tuning,all-domain training or upload is in this job.

Reuse the four opaque archives once in each fresh ephemeral session; wholearchiveSHA/CRC/exactunion,stage120s/idle30s,loaderI/O30s,totalpretraining600s. Never copy/hash individual mounted images. Actual full-extraction rates and independent head timing will replace projections when saved evidence exists. Original square-root sampler/source-only balancedBCE/selection/calibration are retained. Separate optimizer/RNG/checkpoints per head,PA0/PETA1 thenMarket0.

Predeclared parity tolerances remain FP32max_abs<=5e-4/relL2<=1e-5 andAMPmax_abs<=.05/relL2<=.005; no post-result relaxation. Gatecode/weights/runtime hashes pinned before reuse. Short source-only throughput check repeats in the extraction session and stops below75images/s or above12GiB or if projected extraction exceeds1800s. Completed source/target locks must never be refit or reevaluated during recovery.


Frozen v1 stopped before staging at~76s on an inherited SigLIP HF reference preprocessing test (18passed,1failed). No DINOtraining or targetinference. Restore original Transformers4.57.1 test-reference pin;keep all transforms/tolerances unchanged,recordbefore/afterversions.10.81GPUh remain;retryprovider6500/internal6200within original cumulative2hcap. See reports/dino_v3/FAILED_V1_RECOVERY.md.


## Frozen threefold v2 completed

All19Kaggletests passed with Transformers4.57.1 after the baseline reference pin was restored. ENVIRONMENT_BEFORE records5.16.1 in the freshv2base and AFTER4.57.1;the failedv1environment was not logged,so this is strong drift evidence,not a direct measurement ofv1's version. Torch2.11.0cu128/CUDA12.8,Pillow12.3.0,NumPy2.1.3 retained. No preprocessing/model/test/tolerance changes.

Actual archive staging36.855s,wholearchiveSHA/ZIPCRC/exact131076pathunion passed. Two extraction ranges each65538images completed249.932/250.260s (~524images/saggregate). CacheFP16shape131076x6x768,recordedSHA bd7991deafba88f07edd0d53525252b20989a52ac2e355846b4ce8aa62a2285d. Parent entered extraction146.47s,heads410.10s,complete520.14s from scriptstart;runner446.02s. The broader quota increment~.15h also includes setup/export/rounding and should not be confused with runner-only7.43min.

Cached head wallPA24.32s,PETA85.78s,Market78.99s;parallel scheduling finishes headphase~110s. Each30epochs,best30/27/25;attempted4380/20880/20550,successful4380/20876/20546,AMPskips0/4/4. These are cached-head timings,not estimates of backpropagation through two backbone blocks. All actualimages/sourceexposure/sourcecurves retained in reports/dino_v3/results.

Fresh quota19.34hused10.66hremaining. TotaldisplayedDINOspending~.31hfrom10.97initial,wellbelow cumulative2hcap. No GPUjob active,no all-domainrun,newZIP or submission. Trainedselectedheads also loaded strictly and inferred on real source-only images under Python3.11.16/torch2.4.1CPU with socketsdisabled,no timm/Transformers import,40orderedfinite probabilities/calibration/serialization verified. This does not certify full exactdeployment runtime;A10benchmark remains gated on a promoted candidate.
