# Assignment repository cleanup

The release was prepared from the canonical Git repository at commit b5cfe99, not the older Downloads directory without Git history. It preserves all completed experiment code, aggregate evidence, failed gates, licensing notices and original submission metadata.

The top-level README now identifies the three assignment deliverables, one confirmed official score, local results, code entry points and data/weight restrictions. STATUS was reduced from a repeated chronology to a single final-state summary. The original implementation README remains in `docs/ORIGINAL_IMPLEMENTATION.md`; Git history and the experiment supplement retain earlier decisions. `docs/EXPERIMENT_INDEX.md` maps the detailed reports, and `docs/REPRODUCING.md` separates CPU checks from private-data training and deployment.

Generated caches, virtual environments, pytest scratch, local editor state, transient archives, weights, features, private prediction arrays and credentials are ignored. The source ZIP is built from the tracked release tree, so it contains no `.git`, runtime environments or untracked scratch. No original dataset, accepted checkpoint/calibration or A/B competition artifact is deleted. Historical logs and aggregate results remain because they support the report; they are evidence rather than clutter.

Thirty-one historical baseline files are hash-locked. Fourteen differed in a fresh clone only because text line endings had been normalized by Git. Their exact original evidence bytes were restored, and per-file `-text` attributes now prevent future line-ending conversion. The historical baseline-lock JSON is unchanged. The assignment verifier checks these original hashes directly; it does not bypass them by accepting alternate digests.

Training dependencies now retain the original Transformers-compatible hub/tokenizer constraints, and a separate DINO reference dependency file includes timm. Deployment still imports no optional reference-model libraries. A recorded local test environment and test outcome are included in `assignment/LOCAL_VERIFICATION.json`; this is distinct from actual cloud environment receipts.

No new cloud training, hidden-data access, account switching or competition submission was performed during assignment preparation.
