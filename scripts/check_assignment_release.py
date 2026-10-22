"""Check the submitted Git tree and documentation, excluding ignored local research."""
import json,re,subprocess
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[1]
def main():
    try:
        raw=subprocess.run(['git','ls-files','-z'],cwd=ROOT,check=True,capture_output=True).stdout
        paths=[ROOT/p.decode('utf-8') for p in raw.split(bytes([0])) if p]
    except (OSError,subprocess.CalledProcessError):
        paths=[p for p in ROOT.rglob('*') if p.is_file() and not {'.git','__pycache__','.pytest_cache'}&set(p.parts)]
    errors=[];links=0
    for p in paths:
        rel=p.relative_to(ROOT)
        if rel.parts[0] in {'reports','internal','private-research-archive'}:errors.append('Internal directory tracked: '+str(rel))
        if p.suffix.lower() in {'.pt','.pth','.npz','.npy','.safetensors','.upar','.zip','.png','.jpg','.bmp','.log'}:errors.append('Private/generated file tracked: '+str(rel))
        if p.name in {'kaggle.json','credentials.json','.env'}:errors.append('Credentials tracked: '+str(rel))
        if p.stat().st_size>2*1024*1024:errors.append('Unexpected large source file: '+str(rel))
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
                if '://' in target or target.startswith(('#','mailto:')):continue
                target=unquote(target.split('#',1)[0]);links+=1
                if target and not (p.parent/target).exists():errors.append('Broken link in '+str(rel)+': '+target)
    if errors:raise SystemExit('\n'.join(errors))
    print(json.dumps({'passed':True,'submitted_files_checked':len(paths),'local_links_checked':links,'internal_reports_tracked':False,'private_weights_data_credentials_tracked':False},indent=2))
if __name__=='__main__':main()
