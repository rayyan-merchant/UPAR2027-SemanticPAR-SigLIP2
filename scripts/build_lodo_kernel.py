"""Build a private sequential LODO job; attach existing foundation/data, never PAR weights."""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile
import hashlib

parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
parser.add_argument('--recovery-source', help='Explicit earlier LODO kernel for interruption recovery, never all-domain training')
args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
buf = io.BytesIO(); hashes = {}
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in ['upar', 'scripts', 'tests', 'kaggle', 'submission_template']:
        for path in sorted((root / folder).glob('*.py')):
            relative = path.relative_to(root).as_posix(); archive.write(path, relative)
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['reports/lodo/EXPECTED_SPLITS.json', 'submission_template/metadata.yaml', 'pytest.ini']:
        path = root / name; archive.write(path, name); hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = base64.b64encode(buf.getvalue()).decode('ascii')
runner = '''import base64,io,zipfile,runpy
from pathlib import Path
code=Path('/kaggle/working/lodo_code');code.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:archive.extractall(code)
runpy.run_path(str(code/'kaggle/lodo_runner.py'),run_name='__main__')
'''.replace('PAYLOAD', repr(payload))
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
(out / 'run_lodo.py').write_text(runner, encoding='utf-8')
sources = ['rayyannmerchant/upar2027-siglip2-semanticpar']
if args.recovery_source:
    if 'lodo' not in args.recovery_source: raise ValueError('Recovery must be a LODO kernel')
    sources.append(args.recovery_source)
metadata = {'id': 'rayyannmerchant/upar2027-three-domain-lodo-diagnosis', 'title': 'UPAR2027 three-domain LODO diagnosis',
    'code_file': 'run_lodo.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
    'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
    'dataset_sources': ['rayyannmerchant/upar2027-official-private'], 'kernel_sources': sources, 'competition_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
receipt = {'kernel': metadata['id'], 'source_sha256': hashes, 'code_bytes': len(runner),
           'foundation_source': sources[0], 'fresh_native_foundation_sha256': '55779d993f026e4dd54f320fceebe78640179ff9cb73edc9b2b245694434e832',
           'all_domain_PAR_weights_used': False, 'fold_order': ['PA100k', 'PETA', 'Market1501'],
           'new_competition_submission_authorized': False}
(root / 'reports/lodo/KERNEL_BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
print('Prepared private sequential LODO kernel:', out, 'bytes:', len(runner))
