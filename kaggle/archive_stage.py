"""Stream opaque archives once, extract locally, verify bytes; watchdog is external to worker I/O."""
import argparse, hashlib, io, json, multiprocessing as mp, os, queue, random, shutil, time, zipfile
from pathlib import Path, PurePosixPath

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))

def worker(archives, dest, q, access_samples):
    try:
        from PIL import Image
        root=Path(dest).resolve();root.mkdir(parents=True,exist_ok=True)
        records=[];all_names=[];declared=set();started=time.monotonic()
        for item in archives:
            src=Path(item['path']);cache=root.parent/'archive_cache';cache.mkdir(exist_ok=True)
            local=cache/src.name
            if src.resolve()==local.resolve():raise ValueError('Source cannot be cache destination')
            t=time.monotonic();h=hashlib.sha256();done=0
            with src.open('rb') as f,local.open('wb') as g:
                for b in iter(lambda:f.read(4*1024*1024),b''):
                    g.write(b);h.update(b);done+=len(b)
                    q.put({'phase':'stream_archive','bytes':done,'total_bytes':item['archive_bytes'],
                           'elapsed':time.monotonic()-t,'archive':src.name})
            if h.hexdigest()!=item['archive_sha256']:raise ValueError('Archive checksum mismatch')
            copy_s=time.monotonic()-t;t=time.monotonic()
            with zipfile.ZipFile(local) as z:
                manifest=json.loads(z.read('ARCHIVE_MANIFEST.json'))
                if len(z.namelist())!=len(manifest)+1 or set(z.namelist())!=set(manifest)|{'ARCHIVE_MANIFEST.json'}:
                    raise ValueError('Unexpected or duplicate archive entries')
                for i,(name,meta) in enumerate(manifest.items(),1):
                    relative=PurePosixPath(name)
                    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                        raise ValueError('Unsafe archive member')
                    target=(root/name).resolve()
                    if not target.is_relative_to(root) or name in declared:raise ValueError('Overlapping/unsafe member')
                    declared.add(name)
                    if z.getinfo(name).file_size!=meta['bytes']:raise ValueError('Member size mismatch')
                    data=z.read(name)
                    if item.get('verify_member_sha256',False) and hashlib.sha256(data).hexdigest()!=meta['sha256']:raise ValueError('Image checksum mismatch')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);all_names.append(name)
                    if i%50==0 or i==len(manifest):q.put({'phase':'extract_local','files':i,'total_files':len(manifest),
                        'elapsed':time.monotonic()-t,'archive':src.name})
            records.append({'archive':src.name,'images':len(manifest),'stream_seconds':copy_s,
                'extraction_hash_write_seconds':time.monotonic()-t,'archive_bytes':done})
        rng=random.Random(2027);sample=rng.sample(all_names,min(access_samples,len(all_names)))
        times=[];read_bytes=0
        for n in sample:
            t=time.monotonic();b=(root/n).read_bytes()
            with Image.open(io.BytesIO(b)) as im:im.convert('RGB').load()
            times.append(time.monotonic()-t);read_bytes+=len(b)
        times.sort()
        q.put({'complete':True,'archives':records,'images':len(all_names),'elapsed_seconds':time.monotonic()-started,
            'random_access':{'samples':len(times),'bytes':read_bytes,'total_seconds':sum(times),
                'p50_ms':1000*times[len(times)//2] if times else None,'p95_ms':1000*times[min(len(times)-1,int(len(times)*.95))] if times else None},
            'all_archive_hashes_passed':True,'member_integrity':'ZIP CRC; source SHA verified at packaging; no repeat image SHA/decoding', 'path_set_sha256':hashlib.sha256('\n'.join(sorted(all_names)).encode()).hexdigest(),'training_started':False,'model_inference':False})
    except BaseException as e:q.put({'error':type(e).__name__+': '+str(e)})

def stage(archives,dest,report,max_seconds=600,max_idle=45,max_projected=600,access_samples=300):
    context=mp.get_context('spawn');q=context.Queue();p=context.Process(target=worker,args=(archives,dest,q,access_samples))
    start=last=time.monotonic();p.start();events=[];result=None
    try:
        while True:
            now=time.monotonic()
            if now-start>max_seconds:raise TimeoutError('Staging wall-time budget exceeded')
            if now-last>max_idle:raise TimeoutError('No I/O progress within idle budget; GPU training prohibited')
            try:r=q.get(timeout=.5)
            except queue.Empty:
                if not p.is_alive():raise RuntimeError('Worker exited without completion evidence')
                continue
            last=time.monotonic()
            if 'error' in r:raise RuntimeError(r['error'])
            if r.get('complete'):result=r;break
            events.append(r)
            unit='bytes' if 'bytes' in r else 'files';done=r[unit];total=r['total_'+unit]
            projected=r['elapsed']*total/max(done,1)
            if r['elapsed']>=5 and projected>max_projected:raise TimeoutError('Projected phase time exceeds budget')
        p.join(timeout=5)
        result.update({'passed':True,'watchdog':{'max_seconds':max_seconds,'max_idle_seconds':max_idle,
            'max_projected_phase_seconds':max_projected},'events':events})
    except BaseException as e:
        result={'passed':False,'error':type(e).__name__+': '+str(e),'events':events,'elapsed_seconds':time.monotonic()-start,
            'training_started':False}
    finally:
        if p.is_alive():p.terminate();p.join(timeout=5)
    write(Path(report),result)
    if not result['passed']:raise RuntimeError(result['error'])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--archive-root',required=True)
    ap.add_argument('--names',nargs='+',required=True);ap.add_argument('--dest',required=True);ap.add_argument('--report',required=True)
    ap.add_argument('--max-seconds',type=float,default=600);ap.add_argument('--max-idle',type=float,default=45)
    a=ap.parse_args();items=json.loads(Path(a.manifest).read_text())['archives']
    chosen=[dict(x,path=str(Path(a.archive_root)/x['archive'])) for x in items if x['archive'] in a.names]
    if len(chosen)!=len(set(a.names)):raise ValueError('Missing/duplicate selected archive')
    r=stage(chosen,a.dest,a.report,a.max_seconds,a.max_idle)
    print(json.dumps({k:v for k,v in r.items() if k!='events'}),flush=True)

if __name__=='__main__':main()
