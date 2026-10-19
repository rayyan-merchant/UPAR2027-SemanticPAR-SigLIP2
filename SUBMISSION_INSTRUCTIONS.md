# Competition submission and recovery

## Current restriction (2026-10-20)

Submission1003877 is Finished at Challenge_Avg .770450. Its server file exactly
matches the preserved ZIP. **Do not upload again automatically.** User requested
generalization diagnosis and findings before new GPU consumption. See
GENERALIZATION_DIAGNOSIS.md and STATUS.md; earlier upload directions below are
historical recovery instructions, not authorization for another submission.

## Verified package

The submission file is `UPAR2027_SemanticPAR_SigLIP2.zip` in the sibling
`outputs/submission/` directory. It is 358,029,220 bytes; SHA256:
`76df2273852daa15cf968bcb9384c5e1602ff0bd76dde49adfa879d814f86308`.
It contains root `run.py`, exact original `metadata.yaml`, four native `upar`
modules, trained epoch 7 EMA weights and one global calibrated threshold vector.
Its inference batch size remains 32. There is no text encoder or runtime
Transformers/timm/network requirement.

Actual trained offline API tests passed on Python 3.11/torch 2.4.1 CPU and
Kaggle CUDA. ZIP CRC, original metadata, attribute ordering, 40 finite probabilities,
empty/single/chunked batching and blocked-network optional imports were checked.
The final local-storage T4 measurement was 2,000 images in 10.263 seconds,
14.576 seconds for the complete benchmark process. Staging from Kaggle's remote
input mount is reported separately. This does not certify the unknown official
A10 workload; actual competition evaluation is required.

## Upload

Open https://www.codabench.org/competitions/18180/#/participate-tab and verify the
signed-in account and available submission quota. The authorized account observed
in this session is `rayyanmerchant`. Select the verified ZIP through Submission
upload. Keep the tab open until upload completes and a submission row/ID appears.
Do not start another upload while an existing transfer is active.

Record submission ID, exact file hash, timestamp, terminal status, score and a
screenshot. An uploaded file or queued evaluation alone is not a valid-submission
result. The latest valid submission counts. Consult STATUS.md and the actual
competition page before taking action; an upload was in progress when these
instructions were first written.

Do not change weights, thresholds or model selection based on hidden image data,
metadata or a hidden-score tuning loop. Resolve an actual runtime/interface error
from its diagnostic evidence, verify the same architecture locally, and document
any technical resubmission. Registration terms or account actions may require the
user. No organizer email has been authorized.

## Preserved training evidence

`outputs/kaggle-v4-artifacts/runs/siglip2/` contains the selected `best.pt`, full
`last.pt`, validation predictions, configuration, history, calibration and metrics.
Best weights SHA256:
`18a48956eac97d16709e355d57e747a9944c872f844e20645d1dc5aae86ae0d6`.
Resume state SHA256:
`64d23096ce17db94f667055c65dfddbf3861844c4e8867e64186ed2eb9259f74`.
The resume checkpoint was loaded and structurally verified: schema 2, 12 completed
epochs, 18,324 training steps, optimizer, EMA and Python/NumPy/torch/CUDA/sampler RNG states.
No additional training is required for submission.

Training source receipt: reports/kaggle_v4_receipt.json. Training commit bc987db;
full dependency freeze and measured data/parity/inference evidence are in reports/.
TRAINING_RESULTS.md distinguishes known-domain validation, in-sample calibration
and pending official results. REPRODUCIBILITY.md covers organizer verification.
Original datasets remain read-only and images/checkpoints are excluded from public
Git. Preserve source model/data licenses and citations in any organizer release.
