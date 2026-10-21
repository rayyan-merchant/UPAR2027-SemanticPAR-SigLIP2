# DINOv3 + SigLIP2 comparison — predeclared, pending

Use only the fixed .75SigLIP2/.25DINO raw sigmoid-probability average, independently calibrating its40thresholds from corresponding source-calibration data. Reuse byte-hash-pinned original SigLIP per-fold source/target predictions with exact row/label/valid/domain alignment. Never tune ensemble weight on targets or hidden scores.

Promotion: calibrated LODO macro>=.8023,worst>=.7772,no domain regression>.005. Record standalone and ensemble raw/calibrated mA,instance-F1,P/R,attribute groups,source-target gaps,error correlation and runtime. Report outcomes under both available scorer conventions. No DINOv3 result exists yet. No final all-domain training or upload has started.
