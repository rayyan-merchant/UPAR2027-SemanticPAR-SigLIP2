"""Read-only original data packaging. JPEG/PNG bytes remain unchanged."""
import argparse, csv, hashlib, json, time, zipfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def package(names, manifest, source, target):
    started=time.monotonic(); entries={}; total=0
    if target.exists():raise FileExistsError(target)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for n in sorted(names):
            p=source/manifest[n]['source_relative_path']; data=p.read_bytes()
            h=hashlib.sha256(data).hexdigest()
            if h!=manifest[n]['sha256']:raise ValueError('Original hash changed: '+n)
            info=zipfile.ZipInfo(n,date_time=(2026,1,1,0,0,0)); info.external_attr=0o100644<<16
            z.writestr(info,data); entries[n]={'sha256':h,'bytes':len(data)};total+=len(data)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(entries,sort_keys=True))
    r={'archive':target.name,'archive_sha256':sha(target),'archive_bytes':target.stat().st_size,
       'image_bytes':total,'images':len(entries),'build_seconds':time.monotonic()-started,
       'format':'ZIP_STORED with .upar extension; verify server retains opaque archive',
       'data_root_read_only':str(source)}
    print(json.dumps(r),flush=True);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--splits',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);m=json.loads(Path(a.manifest).read_text())
    groups={d:sorted(n for n in m if n.startswith(d+'/')) for d in ['Market1501','PETA','PA100k']}
    sample=[]
    for names in groups.values(): sample += sorted(names,key=lambda n:hashlib.sha256(('archive-probe-2027:'+n).encode()).digest())[:300]
    selected=set()
    for p in Path(a.splits).glob('*.csv'):
        with p.open(newline='',encoding='utf-8-sig') as f:
            rows=csv.reader(f);next(rows)
            selected.update(row[0] for row in rows)
    reports=[package(sample,m,Path(a.source),out/'probe_900.upar')]
    # Exact PA-held fold union; no PA training images staged for that diagnostic.
    for d,names in groups.items():
        needed=[n for n in names if n in selected]
        reports.append(package(needed,m,Path(a.source),out/(d+'_PAheld.upar')))
    (out/'ARCHIVES.json').write_text(json.dumps({'archives':reports,'selection_images':len(selected),
        'source_manifest_sha256':sha(Path(a.manifest)),'source_split_sha256':{p.name:sha(p) for p in Path(a.splits).glob('*.csv')}},indent=2))

if __name__=='__main__':main()
