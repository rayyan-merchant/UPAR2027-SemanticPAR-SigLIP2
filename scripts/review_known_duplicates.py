"""Record review of the specific official-split duplicate found in this audit."""
import argparse
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--audit-dir',required=True)
args=ap.parse_args();root=Path(args.audit_dir)
path=root/'data_audit.json';audit=json.loads(path.read_text())
expected=[['PETA/images/08572.png','PETA/images/08573.png','PETA/images/08574.png']]
if audit['cross_split_duplicates']['pixel_sha256']!=expected:
    raise ValueError('This review only covers the inspected three-file MIT duplicate')
audit['duplicate_review']={
    'groups':expected,'train_paths':['PETA/images/08572.png'],
    'val_paths':['PETA/images/08573.png','PETA/images/08574.png'],
    'finding':'Same decoded RGB image, distinct JPEG bytes; visually inspected original MIT source image.',
    'label_conflicts':[],
    'policy':'Preserve official splits and annotations. Report official-validation and duplicate-excluded sensitivity metrics. '
             'Exclude these two validation rows when fitting thresholds. Validation is not completely leakage-free.',
    'near_duplicate_or_identity_leakage':'not assessed'}
audit['leakage_review_required']=False
path.write_text(json.dumps(audit,indent=2))
with (root/'DATA_AUDIT.md').open('a') as f:
    f.write('\n## Reviewed exact-pixel leakage\n\n'+audit['duplicate_review']['finding']+'\n\n'+audit['duplicate_review']['policy']+'\n')
print('Recorded review; no annotations or splits changed.')
