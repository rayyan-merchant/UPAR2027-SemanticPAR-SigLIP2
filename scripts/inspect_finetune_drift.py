"""Read-only backbone drift audit, not weight interpolation or model evaluation."""
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
foundation = torch.load(root / 'reports/foundation_native.pt', map_location='cpu', weights_only=False, mmap=True)
trained = torch.load(root.parent / 'kaggle-v4-artifacts/runs/siglip2/best.pt', map_location='cpu', weights_only=False, mmap=True)
summary = {}
for key, initial in foundation['state_dict'].items():
    if not key.startswith('vision_model.'):
        continue
    current = trained['state_dict'][key]
    group = 'other_backbone'
    if '.encoder.layers.' in key:
        group = 'block_' + key.split('.encoder.layers.')[1].split('.')[0].zfill(2)
    diff = current.float() - initial.float()
    row = summary.setdefault(group, {'elements': 0, 'base_sq': 0., 'delta_sq': 0., 'max_abs': 0.})
    row['elements'] += initial.numel()
    row['base_sq'] += float((initial.float() ** 2).sum(dtype=torch.float64))
    row['delta_sq'] += float((diff ** 2).sum(dtype=torch.float64))
    row['max_abs'] = max(row['max_abs'], float(diff.abs().max()))
for row in summary.values():
    row['relative_l2'] = (row['delta_sq'] / max(row['base_sq'], 1e-30)) ** .5
result = {'groups': summary, 'interpretation': 'Parameter movement only; no evidence that interpolation improves held-out-domain predictions. Foundation PAR classifier is untrained, so full-model zero-shot/finetuned interpolation is not a valid WiSE-FT baseline here.',
          'modified_model_created': False, 'GPU_used': False}
out = root.parent / 'diagnosis_1003877/finetune_drift.json'
out.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
