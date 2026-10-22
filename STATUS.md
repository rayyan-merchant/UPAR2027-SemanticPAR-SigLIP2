# Assignment final state

The project contains one confirmed official evaluation and a completed local research sequence. **Original SigLIP2 submission 1003877 is Finished at Challenge_Avg 0.770450** and remains the evaluated fallback. Candidate A's approved upload reset during the reported site outage; no new submission ID or official score is confirmed. Candidate B completed all-domain training and source-only selection/calibration; its improved transfer remains locally evaluated. No official A/B result is fabricated.

| Branch | Final state |
|---|---|
| Original SigLIP2 | Twelve epochs completed; epoch-7 EMA selected; official score 0.770450 |
| Original three-fold LODO | Complete; calibrated macro 0.797260 / worst 0.777228 |
| Equal-domain sampling | Complete; macro 0.796089 / worst 0.776778; rejected |
| DINOv2 fallback | Job cancelled after DINOv3 access passed; no completed score |
| Frozen DINOv3 / A ensemble | Three-fold complete; ensemble macro 0.800964 / worst 0.780285; original promotion gate failed; separate all-domain head selected epoch 30 and package preserved |
| B partial DINOv3 adaptation | Three-fold complete; macro 0.802508 / worst 0.781356; +0.002 advancement gate failed; separately authorized all-domain run completed eight epochs, best epoch 8 |
| C SigLIP robustness augmentation | Three-fold complete; macro 0.798076 / worst 0.776418; rejected; no final C training |
| B final model | Eight epochs completed, best epoch 8; source-selection ensemble 0.869305; no confirmed official score |

No additional training or official submission was started by assignment preparation. Historical root reports and hash-bound receipts remain evidence of the stage at which they were written; this summary supersedes their pending-status text without changing negative research decisions. Start with the [three-page report](assignment/UPAR2027_Assignment_Report.pdf), [complete experiment record](assignment/EXPERIMENTS_AND_DECISIONS.md), [results table](assignment/RESULTS.csv) and [reproduction guide](docs/REPRODUCING.md).

The assignment narrative focuses on modeling choices, reasoning, local comparisons, challenges and improvements. Both A/B improved all three local held-out domains, motivating an expectation of improved official generalization; the interrupted follow-up evaluation leaves exact official scores unknown. Original technical audit records remain unchanged for provenance.
