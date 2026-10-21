"""Byte-verified archives. Does not load tensors or claim visual correctness."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def inventory(root, paths):
    root=Path(root).resolve()
    rows=[]
    for rel in sorted(set(map(str,paths))):
        p=root/rel
        if not p.resolve().is_relative_to(root) or p.is_symlink() or not p.is_file():
            raise ValueError('only regular files inside root: '+rel)
        rows.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha256(p)})
    return rows

def create(root, paths, archive):
    root=Path(root).resolve();archive=Path(archive)
    rows=inventory(root, paths)
    with tarfile.open(archive,'x:gz' if str(archive).endswith('.gz') else 'x') as t:
        for r in rows: t.add(root/r['path'],arcname=r['path'],recursive=False)
    result={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha256(archive),'files':rows}
    archive.with_name(archive.name+'.json').write_text(json.dumps(result,indent=2))
    return result

def verify(archive, manifest, target):
    archive=Path(archive);target=Path(target)
    if sha256(archive)!=manifest['sha256']:raise ValueError('archive checksum mismatch')
    if target.exists():raise FileExistsError('fresh extraction target required')
    target.mkdir(parents=True)
    with tarfile.open(archive) as t:
        members=t.getmembers()
        expected={r['path'] for r in manifest['files']}
        if len(members)!=len(expected) or {m.name for m in members}!=expected:raise ValueError('member list mismatch')
        for m in members:
            if not (target/m.name).resolve().is_relative_to(target.resolve()) or not (m.isfile() or m.islnk()):
                raise ValueError('unsafe member')
        t.extractall(target,filter='data')
    actual=inventory(target,expected)
    if actual!=manifest['files']:raise ValueError('extracted member checksum mismatch')
    return {'PASS':True,'members_extracted_and_hashed':len(actual),'bytes':sum(r['bytes'] for r in actual)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('create');c.add_argument('--root',type=Path,required=True);c.add_argument('--paths-json',type=Path,required=True);c.add_argument('--archive',type=Path,required=True)
    v=sub.add_parser('verify');v.add_argument('--archive',type=Path,required=True);v.add_argument('--manifest',type=Path,required=True);v.add_argument('--target',type=Path,required=True)
    a=p.parse_args()
    result=create(a.root,json.loads(a.paths_json.read_text()),a.archive) if a.action=='create' else verify(a.archive,json.loads(a.manifest.read_text()),a.target)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
