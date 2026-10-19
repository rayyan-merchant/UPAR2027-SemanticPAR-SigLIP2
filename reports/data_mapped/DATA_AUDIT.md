# Data audit

Read-only audit of actual official annotations and all referenced images.

Train: 97,669; validation: 33,407; unique paths: 131,076.
Fully decoded: 131,076; file errors: 0.
Path overlap: 0; cross-split byte duplicate groups: 0;
cross-split decoded-pixel duplicate groups: 1.

40 ordered attribute names match both official CSV headers.
Paths resolved via official PETA mapping: 15,631. Source files unchanged.

| Split | Domain | Records | Invalid label cells |
|---|---|---:|---:|
| train | Market1501 | 10,000 | 0 |
| train | PA100k | 79,001 | 0 |
| train | PETA | 8,668 | 0 |
| val | Market1501 | 16,458 | 0 |
| val | PA100k | 9,986 | 0 |
| val | PETA | 6,963 | 0 |

Expected domain sampling mass (per-image N_domain^-0.5):

- Market1501: 0.210893
- PA100k: 0.592760
- PETA: 0.196346

Per-domain, per-attribute positive/negative counts and prevalence: data_audit.json.
Image dimensions, byte and decoded-pixel hashes: image_manifest.json.

Exact byte/pixel duplicates only; near duplicates and shared identities need separate review.
Duplicate findings require review; original splits have not been changed.

## Reviewed exact-pixel leakage

Same decoded RGB image, distinct JPEG bytes; visually inspected original MIT source image.

Preserve official splits and annotations. Report official-validation and duplicate-excluded sensitivity metrics. Exclude these two validation rows when fitting thresholds. Validation is not completely leakage-free.

## Reviewed exact-pixel leakage

Same decoded RGB image, distinct JPEG bytes; visually inspected original MIT source image.

Preserve official splits and annotations. Report official-validation and duplicate-excluded sensitivity metrics. Exclude these two validation rows when fitting thresholds. Validation is not completely leakage-free.
