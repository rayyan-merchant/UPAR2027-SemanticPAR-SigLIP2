"""Create a private-upload archive from the passed audit, without altering source images."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit-dir',required=True)
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--owner',required=True)
    args=ap.parse_args()
    audit_dir=Path(args.audit_dir)
    evidence=json.loads((audit_dir/'data_audit.json').read_text())
    if not evidence['passed'] or evidence['leakage_review_required']:
        raise ValueError('Data integrity and duplicate review gates must pass before transfer')
    manifest=json.loads((audit_dir/'image_manifest.json').read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    archive=out/'official_data.zip'
    root=Path(args.data_root)
    # Byte-for-byte copies under official annotation paths, exactly as the official
    # download script renames PETA originals. No conversion/re-encoding.
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for i,(name,item) in enumerate(manifest.items(),1):
            source=root/item['source_relative_path']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError(f'Source changed after audit: {source}')
            z.write(source,name)
            if i % 20000==0:print(f'Archived {i}/{len(manifest)}',flush=True)
        for split in ['train','val']:
            source=root/f'annotations/task1/{split}/gt.csv'
            if hashlib.sha256(source.read_bytes()).hexdigest()!=evidence['splits'][split]['sha256']:
                raise ValueError('Annotations changed after audit')
            z.write(source,f'annotations/task1/{split}/gt.csv')
    # Dataset API creates private by default; never pass --public/-u.
    metadata={'title':'UPAR2027 official private research data',
              'id':f'{args.owner}/upar2027-official-private',
              'licenses':[{'name':'other'}],
              'description':'Private academic competition workspace. Official UPAR2027 training/validation data only. '
              'Source dataset licenses retained: PETA research only; PA100K CC BY 4.0; Market1501 original terms. '
              'No permission for public redistribution is asserted. Cite original dataset papers.'}
    (out/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
    receipt={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
             'size_bytes':archive.stat().st_size,'image_count':len(manifest),'private_required':True}
    (out/'provenance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
