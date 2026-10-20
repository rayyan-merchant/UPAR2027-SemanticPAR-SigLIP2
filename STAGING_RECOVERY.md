# Staging recovery and revised PA-held-out schedule

2026-10-20. No replacement GPU training or competition upload has been launched.

## Cancellation and preserved evidence

User's execution log: `35703.7s 1183 Staged 60000 / 131076`.
The retrieved server STATUS.json independently records phase
`staging_private_official_images`, images_done=60000, total_images=131076.
The submitted runner calls tests and every training function only after this
loop finishes. Thus no fold training had begun at this saved staging state.
User clicked Stop and confirmed DONE. The latest API status is CANCEL_REQUESTED;
terminal cancellation is still pending verification. Quota API independently reports zero GPU time reserved and used quota stable at15.99234h; resource reservation has been released despite the stale/nonterminal cancellation state. Do not describe it as a
completed cancellation yet. CLI has no cancel command; SDK cancellation requires
a session ID, but the owned search API returned sessionId:null. No guessed ID,
kernel deletion or replacement training run was used.

Available server progress, local split CSVs/receipts and submitted-source receipts
are preserved in D:/UPAR2027/diagnostics/staging-v1-evidence. Live/full server logs
were unavailable; the supplied excerpt is preserved verbatim. Canonical accepted
ZIP, best.pt and last.pt were read and SHA256-verified unchanged; see
CANCELLATION_AND_PRESERVATION.json. Submission 1003877 remains accepted and unchanged.

## Cause and uncertainty

60000 files after 35703.7 seconds is 1.68 files/s end-to-end. Linear projection
is 21.67 hours for 131076 files, including any startup time. It is not a pure
per-file storage latency measurement. The code opens one mounted image, reads
it, hashes it and writes it locally, repeated for 131076 images with 16 threads.
The thread-pool result iterator preserves input order: reported progress can lag
other completed workers. This does not change the gate that no training begins
until all staging completes.

High per-file mounted-storage overhead is the strongest supported explanation:
there is no model work in that phase and the archive benchmark below shows fast
large sequential reads and fast local small-file extraction/access. The exact
Kaggle backend cause (cache/mount throttling/metadata service) is not exposed;
do not attribute it to a proven specific Kaggle implementation. The old design
also staged 79001 PA training images unnecessary for a PA-held-out diagnostic,
and allocated a GPU throughout preparation without a watchdog. The original
3.5–5-hour estimate omitted this observed staging cost and was unreliable.

## Replacement data workflow

Read originals from D:/UPAR2027/official/data only; byte-hash every selected image
against the prior full audit. No original file is modified. Write deterministic
ZIP_STORED containers (JPEG/PNG bytes unchanged) per domain, with opaque `.upar`
extensions and internal image hash manifests. Plain ZIP uploads were previously
automatically expanded by Kaggle. This extension was tested: Kaggle files lists
one probe_900.upar of exactly 6201861 bytes, rather than 900 mounted images.

PA-held-out package is exactly the union of the four precommitted partitions:
18668 source train, 11634 source selection, 11785 source calibration, 9986 target.
The PETA validation duplicates excluded by the original split remain excluded.
No PA training images enter this package. Domain archives:

| Archive | Images | Bytes |
|---|---:|---:|
| Market1501_PAheld.upar | 26,458 | 70,744,416 |
| PETA_PAheld.upar | 15,629 | 209,872,630 |
| PA100k_PAheld.upar | 9,986 | 41,181,208 |

Total: 52073 images, 321,798,254 archive bytes;
306369041 image bytes. Build time for the three containers was 241.0 seconds
on Windows/D:, including full hash checks. Original split CSV hashes are unchanged
and recorded in ARCHIVES.json. The annotation index is about 16.25 MB and contains
only the original officially permitted train/validation annotations. Threshold
fitting/checkpoint selection still use source-only disjoint partitions.

Private files are same-owner academic workspace copies, with license `other`,
original restrictions and citations retained; they are not relicensed. PETA's
official page states research-only/noncommercial and requires citation. The
Market1501 official page requires citation; no broader redistribution right is
asserted. No public dataset/sharing expansion is made. Sources:
[PETA](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html),
[Market1501](https://zheng-lab-anu.github.io/Project/project_reid.html),
[Kaggle datasets documentation](https://www.kaggle.com/docs/datasets).

## Actual benchmarks

Four local tests passed: valid extraction, archive-checksum rejection, path-traversal
rejection and watchdog timeout. An independent supervisor also terminated a
deliberately sleeping child at its one-second test budget (actual 1.5 seconds).
The actual 900-image local Windows benchmark took 3.735 seconds total.

Private Kaggle CPU probe v1 failed before I/O on a missing import path; its log is
retained. Fixed v2 COMPLETE, Python3.13.15, no GPU requested, no model inference:

| Measurement | Actual |
|---|---:|
| Mounted archive sequential read/hash/write | 0.092981 s |
| Local extraction/image hashes/writes, 900 images | 0.228811 s |
| 300 random reads and RGB decodes | 0.277187 s |
| Random-access median / p95 | 0.294 / 0.911 ms |
| Total measured worker pipeline | 0.601325 s |

First-order PA package projection: 4.8s streaming +
13.2s local extraction/hash/write = 18.1s,
excluding startup/index overhead. This is a 900-image projection, not measured
full-package performance or a training-throughput test. Reserve 2–5 CPU minutes
for full-package verification before any GPU restart. Do not extrapolate Windows
timing to Kaggle. CPU benchmark hardware reported 20.94 GB available working disk.

## Full selected-fold verification — completed after the small probe

The private PA-held-out archive dataset upload completed and the server lists
three opaque archives with exact expected byte sizes. CPU-only kernel version3
COMPLETE verified every selected image hash before any training:

| Measurement | Actual |
|---|---:|
| Images extracted and hash-verified | 52,073 |
| Mounted archive stream/hash/write total | 13.110s |
| Local extraction/hash/write total | 13.760s |
| Full worker pipeline, including random access | 27.389s |
| Random image reads/RGB decodes | 1000 in 0.515s |
| Random-access median / p95 | 0.283/0.856ms |

All image and archive hashes passed. No GPU, training, model inference or
attribute-performance evaluation was involved. The actual27.39s replaces the
18s small-sample projection. For GPU setup retain a conservative two-minute
archive-stage watchdog and five-minute whole-pretraining deadline. Actual
training throughput, six-epoch convergence and held-out accuracy remain unknown.
Ten focused source/isolation/archive/watchdog tests passed after rerunning with
a writable fresh temporary directory; the initial combined test attempt had
four fixture setup errors due to Temp-directory permissions, not split failures.

## Watchdogs and scope

archive_stage.py uses a separate I/O worker and parent supervisor. The parent
terminates the worker on a total deadline, 30 seconds without progress or an
excessive projected phase duration; blocked filesystem reads cannot indefinitely
hide the deadline. Every archive and extracted member is checksum verified,
entry names are safety checked, duplicates are rejected, and random local access
is measured. No mounted per-image fallback is allowed.

Prepared PA-only runner pins all three archive hashes, verifies original annotation
and split hashes, stages once, and retains the original architecture/loss/6-epoch
recipe. It has a 120-second archive-stage budget and 300-second total pretraining
setup supervisor. On failure the process exits and the scheduled GPU job can
release allocation; a watchdog cannot guarantee an instantaneous provider billing
stop, which must be checked by terminal job status. It is prepared/AST-checked,
not end-to-end GPU-tested. Existing source-only feasibility remains mandatory.

Data are reused throughout all six epochs, source selection/calibration and fixed
target evaluation. No automatic PETA/Market runs follow. If more folds are later
approved, existing source archives can be reused and only additional PA images
need another domain archive; do not repeat individual mounted-file staging.

## Resource budget and recommendation

C:1.31 GiB free, D:66.40 GiB free at the latest check; all new large files are on D:.
GPU quota latest:15.99h used /14.01h remaining of30h. Account dataset inventory is
not a reliable private-quota byte meter (some existing datasets report zero), so
no exact remaining dataset-storage entitlement is claimed. Kaggle documentation lists200GB maximum private datasets and20GB saved working
+output; account-specific entitlement still requires a fresh check. The321.8MB package
+is small compared with those documented limits; successful probe ingestion and
fresh runtime free-space checks are the operative gates. Kaggle documents up to
20GB saved working output; actual probe free disk is the stronger local evidence.

Recommended sequence after user review:

1. Private archive upload/retention and full CPU-only verification are COMPLETE.
   Original API status remains CANCEL_REQUESTED with zero GPU reserved; continue
   monitoring terminal status. Source hash/checkpoint gates remain mandatory.
2. Run only PA-held-out, maximum six epochs: reserve 1–1.5 GPU hours including
   source raw/EMA selection, calibration, one fixed-target evaluation and recovery
   checkpoint I/O. Historical all-domain feasibility measured139images/s head-step
   throughput and12-epoch full pipeline3.52hours; neither proves this fold's actual
   six-epoch runtime. Record actual updates/images/convergence and abort on gates.
3. Independently verify PA results and source locks:15–30 CPU minutes. Decide on
   one evidence-based candidate action; PA alone cannot support a three-domain
   conclusion. Further folds are deferred unless their decision value outweighs
   the deadline cost.
4. Protect at least10 GPU hours and several wall-clock hours for a justified
   all-domain improvement, offline/API/timing verification and upload/evaluation.
   No production change or new submission is automatic. No hidden-score tuning.

Highest-value next step: finish one efficient PA-held-out diagnostic, then decide
on a single change using source-held-out evidence. Do not consume the remaining
deadline on mandatory completion of all three folds. No improved leaderboard
score is promised by a data-pipeline fix, and no new GPU run is authorized by this
report alone.
