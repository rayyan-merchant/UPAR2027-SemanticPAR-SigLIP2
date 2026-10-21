"""Bounded selected-image extraction for the exact sealed A ZIP benchmark."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path,PurePosixPath

BASE=Path('/kaggle/working/a_benchmark')
def write(name,value):
    p=BASE/name;p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def child(spec):
    BASE.mkdir(exist_ok=True);started=time.monotonic();local=BASE/'archives';local.mkdir(exist_ok=True)
    def copy_blob(source,target,expected):
        h=hashlib.sha256()
        with Path(source).open('rb') as src,Path(target).open('wb') as dest:
            for block in iter(lambda:src.read(4*1024*1024),b''):
                h.update(block);dest.write(block);write('STAGE_PROGRESS.json',{'phase':'copy_verified_archive','elapsed_seconds':time.monotonic()-started})
        if h.hexdigest()!=expected:raise ValueError('Whole archive checksum mismatch')
    package=local/'candidate.upar';copy_blob(spec['package_path'],package,spec['package_sha256'])
    folder=BASE/'package';folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(package) as z:
        if sorted(z.namelist())!=sorted(spec['package_members']):raise ValueError('Unexpected package payload')
        z.extractall(folder)
    images=BASE/'images';images.mkdir(exist_ok=True);remaining=set(spec['images']);done=0
    for archive in spec['archives']:
        target=local/Path(archive['path']).name;copy_blob(archive['path'],target,archive['archive_sha256'])
        with zipfile.ZipFile(target) as z:
            members=set(z.namelist())&remaining
            for name in sorted(members):
                path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:raise ValueError('Unsafe annotation path')
                dest=images/name;dest.parent.mkdir(exist_ok=True,parents=True)
                with z.open(name) as src,dest.open('wb') as output:shutil.copyfileobj(src,output,64*1024)
                remaining.remove(name);done+=1
                if done%256==0:write('STAGE_PROGRESS.json',{'phase':'extract_selected_images','images':done,'elapsed_seconds':time.monotonic()-started})
    if remaining or done!=29248:raise ValueError('Incomplete representative workload')
    (BASE/'images.json').write_text(json.dumps([str(images/p) for p in spec['images']]))
    write('STAGING.json',{'complete':True,'images':done,'elapsed_seconds':time.monotonic()-started,'whole_archive_hashes_verified':True,'ZIP_CRC_checked_on_extraction':True,'individual_mounted_image_copy_or_hash':False,'package_sha256':spec['package_sha256']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--child',action='store_true');a=p.parse_args();spec=json.loads(Path(a.spec).read_text())
    if a.child:return child(spec)
    BASE.mkdir(exist_ok=True);begin=time.monotonic();proc=subprocess.Popen([sys.executable,__file__,'--spec',a.spec,'--child'])
    try:
        while proc.poll() is None:
            elapsed=time.monotonic()-begin;progress=BASE/'STAGE_PROGRESS.json';idle=time.time()-progress.stat().st_mtime if progress.exists() else elapsed
            if elapsed>120 or idle>30:raise TimeoutError('Staging120s/idle30s watchdog')
            time.sleep(.5)
        if proc.returncode:raise RuntimeError('Selected-image archive staging failed')
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)

if __name__=='__main__':main()
