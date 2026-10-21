# Candidate B — DINOv3 limited fine-tuning

Status: preparation; no new fine-tuning results. Frozen ensemble reference macro.800964/worst.780285 is preserved. See COMPETITIVE_FINAL_PLAN.md for fixed recipe, source-only gates and advancement criteria. Each fold starts from official general-purpose DINOv3 weights and its own frozen source-trained head; all-domain head weights are prohibited. Actual gradients, backward throughput, epoch counts, source curves and target/group metrics remain pending.
