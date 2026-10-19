"""Fit and save one global calibration vector on held-out validation data."""
import argparse, json
from pathlib import Path
import numpy as np
from .metrics import macro_score,calibrate_thresholds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',required=True)
    ap.add_argument('--output',default='runs/siglip2/calibration.json')
    args=ap.parse_args()
    z=np.load(args.predictions)
    if 'calibration_eligible' not in z:
        raise ValueError('Predictions lack the audited calibration eligibility mask')
    keep=z['calibration_eligible'].astype(bool)
    excluded=int((~keep).sum())
    z={key:z[key][keep] for key in ['prob','y','valid','domains']}
    before=macro_score(z['prob'],z['y'],z['valid'],z['domains'].tolist())
    thresholds=calibrate_thresholds(z['prob'],z['y'],z['valid'],z['domains'].tolist())
    after=macro_score(z['prob'],z['y'],z['valid'],z['domains'].tolist(),thresholds)
    from .attributes import ATTRIBUTES
    import hashlib
    result={'attributes':ATTRIBUTES,'thresholds':thresholds.tolist(),'before':before,'after':after,
            'estimation':'in-sample calibration fit on official validation; not an unbiased held-out estimate',
            'known_duplicate_validation_rows_excluded':excluded,
            'prediction_sha256':hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
            'scorer_status':'provisional until organizer differential verification'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'before':before['macro'],'after':after['macro']},indent=2))
if __name__=='__main__':main()
