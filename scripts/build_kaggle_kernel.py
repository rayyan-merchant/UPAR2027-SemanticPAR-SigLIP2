"""Embed source-only code into one private Kaggle CLI script; data stays a private source."""
import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--owner',required=True);ap.add_argument('--output',required=True)
ap.add_argument('--resume-source',help='Kaggle kernel source with saved last.pt; optional')
args=ap.parse_args()
root=Path(__file__).resolve().parents[1]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for folder in ['upar','tests','submission_template','scripts','kaggle']:
        for path in sorted((root/folder).glob('*.py')):
            archive.write(path,path.relative_to(root).as_posix())
    archive.write(root/'submission_template/metadata.yaml','submission_template/metadata.yaml')
payload=base64.b64encode(buf.getvalue()).decode('ascii')
script="""# Private UPAR2027 academic research pipeline. No hidden test access.
import base64,io,zipfile,runpy
from pathlib import Path
root=Path('/kaggle/working/upar_code');root.mkdir(parents=True,exist_ok=True)
payload=PAYLOAD
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:archive.extractall(root)
runpy.run_path(str(root/'kaggle/runner.py'),run_name='__main__')
""".replace('PAYLOAD',repr(payload))
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
(out/'run_upar.py').write_text(script)
metadata={'id':args.owner+'/upar2027-siglip2-semanticpar','title':'UPAR2027 SigLIP2 SemanticPAR',
    'code_file':'run_upar.py','language':'python','kernel_type':'script','is_private':True,
    'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4',
    'dataset_sources':[args.owner+'/upar2027-official-private'],
    'kernel_sources':[args.resume_source] if args.resume_source else [],'competition_sources':[]}
(out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print('Prepared private Kaggle kernel:',out)
