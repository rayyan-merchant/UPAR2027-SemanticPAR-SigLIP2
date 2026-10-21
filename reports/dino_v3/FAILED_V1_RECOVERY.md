# Frozen diagnostic v1 technical stop and controlled recovery

V1 terminal ERROR. Before archive staging,18/19 tests passed; only inherited SigLIP2 HF preprocessing parity failed:47/12288elements,max absolute0.00784319. All8DINO-specific tests passed. No staging,descriptor cache,head training or target inference started. Small log/XML retained privately onD:,XML and receipt preserved here.

Bootstrap inspection found the original reference dependency pin was omitted: successful baseline runners explicitly install transformers4.57.1 with tokenizers>=.22,<.23 and huggingface_hub>=.34,<1;new bootstrap installed onlysafetensors/pytest. Native SigLIP preprocessing explicitly cites4.57.1. The failed run did not log its inherited Transformers version,so its exact installed version and version-specific numerical cause remain unverified. Reference dependency drift is the supported explanation,not a proven assertion of an unrecorded version number.

Recovery restores those original reference pins,records ENVIRONMENT_BEFORE/AFTER before tests and asserts4.57.1. No model,preprocessing,loss,scoring,test or tolerance changes. Targeted original parity test passed locally under4.57.1. The serialized bootstrap and its child syntax were parsed directly. Retry must still pass all19tests onKaggle before staging;do not skip or relax the failure.

Quota10.81h after failure,19.19used. Original cumulative2h cap includes~.16h already consumed. Retry provider6500s/internal6200s,remaining1.84hrounded;never reset budget. Expect>=9hremaining afterhardcap. Prior DINOv2 and DINOv3benchmark outputs protected. V1contains no trained/target locks toresume;start fresh verifieduntrainedDINOv3foundation,exactsealedsourcepartitions. No all-domainrun or upload.
