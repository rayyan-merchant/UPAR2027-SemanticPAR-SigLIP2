# Threefold execution plan — archive staging, independent GPU processes

All three diagnostic folds are authorized, six epochs each, original foundation
only. No competition upload/final all-domain retraining is authorized.

Exact union131076 images. Existing archives cover52073; one382529392-byte private
supplement contains79001 missing PA100k training images and2 PETA validation rows.
No path duplicates and no existing archive was rebuilt/uploaded. Complete CPU
archive benchmark PASSED in34.4237s, with the exact required path-set SHA and all
archive SHA256 checks/ZIP CRC checks. No mounted per-image staging, repeated full
image SHA or full-image decoding.1000 random images were read for I/O timing only.

GPU job uses one shared extraction,120s staging/30s no-progress limits, dependencies
installed once, original runtime pinned and original pretrained foundation hash
55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832.
Model/preprocess/attribute source remains unchanged. Every fold has its own
CUDA_VISIBLE_DEVICES affinity, model, optimizer, EMA, RNG, checkpoints and logs;
workers2 and torch CPU threads2. Fixed seed2027 and original training math retained.

Before folds start: source-only CPU-loader timing and48-batch real GPU training
probes run for each fold alone, followed by synchronized PA/PETA concurrent probes.
CPU/RAM/GPU utilization and memory are sampled. Concurrent mode requires measured
stability and a faster conservative makespan; otherwise all three run sequentially.
In parallel mode PA usesGPU0, PETA usesGPU1, Market starts onGPU0 after PA finishes.
No DDP. No target accuracy enters any scheduling decision.

Runtime projection includes1.5x training margin, six raw/EMA source validations,
source calibration/fixed target inference and300s/fold checkpoint allowance.
Maximum session6h; at least8GPUh reserved using fresh quota. If the measured
projection cannot fit with15min margin, no folds launch. Subsequent source epoch
timings and ETA are checked against the deadline; infrastructure errors stop all
active processes, retain atomic best/last recovery states and report unfinished
folds rather than claim success. Hard provider timeout21600s is also requested.

All six epochs are required before target inference, with source-selected model,
source-calibration global40 vector and checkpoint/calibration/config/split hashes
locked before held-out target use. Recovery retains completed locks and cached
target predictions to prevent repeated target evaluation. Source-only frequencies,
sampling, selection and calibration use only the two permitted source domains.

Actual GPU throughput, chosen scheduling mode, epochs and target metrics remain
pending the run. Both scorer conventions and unsupported-label sensitivity will
be independently recomputed after all three complete. Unequal source updates and
six-epoch convergence/EMA lag must be reported before causal claims. Exactly one
final A–E improvement will be recommended; none will be deployed automatically.
