# Final B v1 technical recovery

Version1 ended ERROR before archive staging, feasibility or training. All25cloud tests passed. The600.3067-second pretraining watchdog stopped setup; noTRAINING_START marker, optimizer updates or trained finalB checkpoints were produced. The saved output inventory shows Candidate A index/label/partition/head copies completed but source split CSVs had not been written.

The new annotation audit incorrectly indexed NpzFile members y/valid/domains for each row. NpzFile indexing decompresses the entire compressed member each time. Local reproduction using the actual pinned label file took26.9057seconds for100rows, projecting35266.87seconds for131076rows (a linear estimate, not observed complete runtime). Loading each array once took.281645seconds, with100cached row accesses taking.0002695seconds.

Fix: materialize each compressed member once and compare paths, labels (including NaNs), invalid masks and domains as complete annotation batches. No label, partition, model, initialization, sampler, class-weight calculation, augmentation, optimizer or epoch/selection/calibration rule changes. Added explicit auditing_source_metadata progress before this phase. Eightfocusedlocaltests pass, including exact-once decompression and deliberate path/label/mask/domain corruption rejection. The full corrected audit over all131076original annotation rows passed in28.4369seconds locally; no images were read.

Primary quota after failedv1:22.39hused/7.61hremaining. Debit the earlier600.3067seconds; no artificial training time cap is added and no quota reset is assumed. Fresh worst8epochprojection must fit remainingquota minus2hdeploymentreserve (5.61h). Source-only per-epoch extension remains six reference plus at most two conditional epochs. Version1 is terminal before version2 submission; preserve v1 diagnostics and the original1003877/A artifacts unchanged.

V2 prepared script SHA25631dcd2e19803a3451e217b52d62e3d88e2fc217af1a7e70323fac202c7eda572. Actual new job state must be verified after submission. Do not claim cloud gates or training have passed based on local audit timing.

## Version2 locator recovery

Version2 is terminal ERROR before staging/backward/training. All27cloudtests passed and the corrected metadata audit completed; runner reached the next SigLIP prediction lookup at39.52seconds. Failure: pinned original SigLIP raw source predictions missing. The reused finetune_runner.find helper deliberately excludes runs/ to avoid other training outputs, but the all-domain production predictions are stored under runs/siglip2/best_val_predictions.npz. Candidate A's earlier finder included runs/.

Use a scoped metadata lookup that includes runs/ while still pruning staged image/archive/cache trees. Keep the exact pinned SHA8c7e126392b32dee9dbf6119f60ff55ac10fa0c2afc176f43e42c5df72d2ff65; a fresh local hash of the preserved original file matches. Identical matching copies are interchangeable by byte hash; mismatched artifacts cannot be used. Save a discovery receipt before copying only this compact prediction file. Added regression test locates a file beneath runs/ and rejects a wrong hash. Six targeted finalBtests passed after this locator change; previously verified model-gradient tests remain unchanged. No training recipe or data split changes.

Quota afterv2:22.42hused/7.58hremaining. V2 script failure~102.29s plusv1~600.31s remain charged; no budget reset or artificial training walltime cap. Worst-case source projection must fit7.58minus2=5.58hreserve gate. V3 prepared scriptSHAd7d5da5ef67fa6d6fdba00959b435508858eb2aff0dc6eed404c38b2e5ba88a9. Preserve v1/v2 diagnostics separately. No trained final B state exists in either failed attempt.
