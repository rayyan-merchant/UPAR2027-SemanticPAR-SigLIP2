"""Fallback bounded HTTPS download of the exact already-resolved public foundation revision."""
import hashlib
import re
import time
from pathlib import Path
import requests

root=Path(__file__).resolve().parents[3]/'work/hf-cache/hub/models--google--siglip2-base-patch16-naflex'
revision=(root/'refs/main').read_text().strip()
if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Invalid cached foundation revision')
snapshot=root/'snapshots'/revision
target=snapshot/'model.safetensors'
part=target.with_suffix('.safetensors.partial')
expected='ac5f28bbdf92c0c1696ccbd3ce716426049cd67ad8045b66d0d938b0f9c8bbec'
url=f'https://huggingface.co/google/siglip2-base-patch16-naflex/resolve/{revision}/model.safetensors'
for attempt in range(4):
    offset=part.stat().st_size if part.exists() else 0
    try:
        with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(15,60)) as response:
            response.raise_for_status()
            append=offset>0 and response.status_code==206
            if not append:offset=0
            with part.open('ab' if append else 'wb') as output:
                last=offset
                for chunk in response.iter_content(1024*1024):
                    output.write(chunk);offset+=len(chunk)
                    if offset-last>=100*1024**2:
                        print(f'Foundation downloaded {offset/1024**2:.0f} MiB',flush=True);last=offset
        break
    except requests.RequestException as error:
        print(f'Attempt {attempt+1} interrupted: {type(error).__name__}',flush=True)
        if attempt==3:raise
digest=hashlib.file_digest(part.open('rb'),'sha256').hexdigest()
if digest!=expected:raise ValueError(f'Foundation SHA256 differs from resolved HF LFS object: {digest}')
part.replace(target)
print(f'Foundation verified: {target.stat().st_size} bytes, revision {revision}',flush=True)
