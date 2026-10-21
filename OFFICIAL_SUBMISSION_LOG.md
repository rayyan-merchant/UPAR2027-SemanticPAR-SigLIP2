# Official submission log

| Entry | Candidate | Challenge_Avg | mA | Instance-F1 | Runtime | Status |
|---|---|---:|---:|---:|---|---|
| 1003877 | Original SigLIP | .770450 | .773647 | .767279 | Reported inference109.91s /29248images | Finished, preserved |
| Not created/confirmed | Candidate A frozen75/25 | Pending | Pending | Pending | A10 test pending | Approved upload reached1% then reset; reconnection timed out |

Candidate A exact ZIP SHA256 `a54232287734b7e854ad472f1007c1830c081838121c4cd797bd343a686ac75d`, 650,810,045bytes. Original ZIP SHA256 `76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308` remains unchanged. No MEVID/Synth breakdown is available for Candidate A.

Last authenticated page showed0/3daily and1/50total, with only1003877listed, before the connection timeout. These are last observations, not a post-timeout server audit. Check for delayed entry creation before retrying. Force_Last means a new official entry becomes latest; highest-score retention must not be assumed. User approved this exact exploratory artifact and the first A10 runtime test during official evaluation. The approved upload has not yet been verified complete.


## Final B completion update

Final all-domain B completed eight epochs, source-selected best epoch 8. Independent audits and offline/reference correctness checks passed; source-selection ensemble Challenge_Avg is 0.869305241. Complete-workload runtime verification is running on T4; official A10 timing and B submission remain unverified. No new official ID or score. See FINAL_B_TRAINING_REPORT.md and CANDIDATE_B_ACCEPTANCE.md. This supersedes earlier pending-training entries without changing historical LODO gate decisions.


## Final B runtime and package decision

The exact-payload T4 surrogate hit its 300-second complete-workload timeout. Its last saved progress was 26,624 of 29,248 images at 296.717 seconds. Source archive staging passed in 9.115 seconds; all inference-file hashes matched the sealed local payload. This is an incomplete workload, not a measured full-run time. A linear extrapolation is approximately 326 seconds and is only an estimate. The T4 deployment timing gate failed; official A10 timing remains unverified. No B upload was performed. Fresh primary quota is 6.06 GPU hours remaining (23.94 used of 30).

The sealed ZIP is `UPAR2027_Candidate_B_Finetuned_DINOv3_75_25_v1.zip`, 650,813,407 bytes, with 18 whitelisted entries and passed CRC checks. SHA256: `f4eee47a694e113a7bb7c527f91592b7ff54fd8cbfdf85f70bb699014cf5f086`. It is stored privately under `D:\UPAR2027\diagnostics\competitive-final`. It excludes training images, feature caches, optimizer states and prediction arrays. Original metadata and required license notices are included.
