# Final B v1 technical recovery

Version1 ended ERROR before archive staging, feasibility or training. All25cloud tests passed. The600.3067-second pretraining watchdog stopped setup; noTRAINING_START marker, optimizer updates or trained finalB checkpoints were produced. The saved output inventory shows Candidate A index/label/partition/head copies completed but source split CSVs had not been written.

The new annotation audit incorrectly indexed NpzFile members y/valid/domains for each row. NpzFile indexing decompresses the entire compressed member each time. Local reproduction using the actual pinned label file took26.9057seconds for100rows, projecting35266.87seconds for131076rows (a linear estimate, not observed complete runtime). Loading each array once took.281645seconds, with100cached row accesses taking.0002695seconds.

Fix: materialize each compressed member once and compare paths, labels (including NaNs), invalid masks and domains as complete annotation batches. No label, partition, model, initialization, sampler, class-weight calculation, augmentation, optimizer or epoch/selection/calibration rule changes. Added explicit auditing_source_metadata progress before this phase. Eightfocusedlocaltests pass, including exact-once decompression and deliberate path/label/mask/domain corruption rejection. The full corrected audit over all131076original annotation rows passed in28.4369seconds locally; no images were read.

Primary quota after failedv1:22.39hused/7.61hremaining. Debit the earlier600.3067seconds; no artificial training time cap is added and no quota reset is assumed. Fresh worst8epochprojection must fit remainingquota minus2hdeploymentreserve (5.61h). Source-only per-epoch extension remains six reference plus at most two conditional epochs. Version1 is terminal before version2 submission; preserve v1 diagnostics and the original1003877/A artifacts unchanged.

V2 prepared script SHA25631dcd2e19803a3451e217b52d62e3d88e2fc217af1a7e70323fac202c7eda572. Actual new job state must be verified after submission. Do not claim cloud gates or training have passed based on local audit timing.
