# Final candidates register

| Candidate | Calibrated LODO macro / worst | Decision | Official status |
|---|---:|---|---|
| Original SigLIP | .797260 / .777228 | Preserved rollback | 1003877 Finished, Challenge_Avg .770450 |
| A frozen DINO75/25 ensemble | .800964 / .780285 | Original promotion gate failed; exploratory entry explicitly approved | Upload reset; connection timeout; no new ID confirmed |
| B upper-two-block DINO ensemble | .802508 / .781356 | Research gate failed; separately authorized final all-domain run v2 RUNNING | None |
| C robustness SigLIP | .798076 / .776418 | Mixed, worse worst domain; no final training | None |

Candidate A ZIP is 650,810,045 bytes, SHA256 `a54232287734b7e854ad472f1007c1830c081838121c4cd797bd343a686ac75d`, stored privately at `D:/UPAR2027/diagnostics/competitive-final/UPAR2027_Candidate_A_Frozen_DINOv3_75_25_v1.zip`. Correctness/data/licensing checks passed; full-workload A10 runtime is unverified and user-approved for testing officially. No 240/300-second runtime pass is claimed.

Original ZIP, best/last checkpoints and production calibration were independently rehashed unchanged. Force_Last selects the latest official entry, not automatically the best. Manual selection of a previous entry is unverified. Keep rollback capacity within the 3/day, 50-total limits. Deadline verified in the competition phase: October22 2026 12:00UTC / 17:00PKT; preferred completion 13:00PKT. Last checked primary Kaggle quota was 7.78 hours remaining; no B/C final runs are active.


## Final B completion update

Final all-domain B completed eight epochs, source-selected best epoch 8. Independent audits and offline/reference correctness checks passed; source-selection ensemble Challenge_Avg is 0.869305241. Complete-workload runtime verification is running on T4; official A10 timing and B submission remain unverified. No new official ID or score. See FINAL_B_TRAINING_REPORT.md and CANDIDATE_B_ACCEPTANCE.md. This supersedes earlier pending-training entries without changing historical LODO gate decisions.


## Final B runtime and package decision

The exact-payload T4 surrogate hit its 300-second complete-workload timeout. Its last saved progress was 26,624 of 29,248 images at 296.717 seconds. Source archive staging passed in 9.115 seconds; all inference-file hashes matched the sealed local payload. This is an incomplete workload, not a measured full-run time. A linear extrapolation is approximately 326 seconds and is only an estimate. The T4 deployment timing gate failed; official A10 timing remains unverified. No B upload was performed. Fresh primary quota is 6.06 GPU hours remaining (23.94 used of 30).

The sealed ZIP is `UPAR2027_Candidate_B_Finetuned_DINOv3_75_25_v1.zip`, 650,813,407 bytes, with 18 whitelisted entries and passed CRC checks. SHA256: `f4eee47a694e113a7bb7c527f91592b7ff54fd8cbfdf85f70bb699014cf5f086`. It is stored privately under `D:\UPAR2027\diagnostics\competitive-final`. It excludes training images, feature caches, optimizer states and prediction arrays. Original metadata and required license notices are included.
