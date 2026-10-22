"""Check hand-in navigation and exclude private/generated files from a source tree."""
import argparse,json,re
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,help='Optional newline-delimited release paths from git ls-files');a=p.parse_args()
    paths=[ROOT/x.strip() for x in a.manifest.read_text(encoding='utf-8-sig').splitlines() if x.strip()] if a.manifest else [x for x in ROOT.rglob('*') if x.is_file() and '.git' not in x.parts and '__pycache__' not in x.parts]
    denied={'.pt','.pth','.npz','.npy','.safetensors','.ckpt','.upar','.jpg','.jpeg','.png','.bmp','.zip','.7z'}
    problems=[]
    for path in paths:
        if path.suffix.lower() in denied:problems.append('Private/generated extension in release: '+str(path.relative_to(ROOT)))
        if path.name in {'kaggle.json','credentials.json','.env'}:problems.append('Credential file in release: '+str(path.relative_to(ROOT)))
        if path.stat().st_size>2*1024*1024:problems.append('Unexpected large source artifact: '+str(path.relative_to(ROOT)))
    links=0
    for path in [ROOT/'README.md',ROOT/'STATUS.md',*(ROOT/'docs').glob('*.md'),*(ROOT/'assignment').glob('*.md'),*[ROOT/d/'README.md' for d in ['upar','dino','tests','scripts','reports']]]:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#') or target.startswith('mailto:'):continue
            target=unquote(target.split('#',1)[0])
            if target and not (path.parent/target).exists():problems.append('Broken local link in '+str(path.relative_to(ROOT))+': '+target)
            links+=1
    if problems:raise SystemExit('\n'.join(problems))
    print(json.dumps({'passed':True,'release_files_checked':len(paths),'local_links_checked':links,'private_weights_images_arrays_credentials_included':False},indent=2))
if __name__=='__main__':main()
