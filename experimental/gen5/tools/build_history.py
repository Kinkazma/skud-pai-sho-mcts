#!/usr/bin/env python3
"""Build the last campaign algorithm with the documented portable file readers."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experimental/gen5'

def stage(out):
    if out.exists():raise ValueError('Historical build directory already exists')
    names=subprocess.check_output(['git','ls-files','experimental/gen5'],cwd=ROOT,text=True).splitlines()
    # Include new uncommitted portability sources when preparing a local draft.
    for p in (EXP/'crates').rglob('*.rs'):
        if 'target' not in p.parts:names.append(str(p.relative_to(ROOT)))
    for name in sorted(set(names)):
        rel=Path(name).relative_to('experimental/gen5')
        if rel.parts[0] in {'target','history','models','inputs','tools'}:continue
        source=ROOT/name
        if source.is_file():
            dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    overlay=json.loads((EXP/'history/last-campaign/overlay.json').read_text())
    for rel,expected in overlay.items():
        source=EXP/'history/last-campaign'/rel;b=source.read_bytes()
        if hashlib.sha256(b).hexdigest()!=expected:raise ValueError('Historical source overlay changed')
        (out/rel).write_bytes(b)
    # Bundle content remains checked; the original prefix preserves sorted recall order.
    current=(EXP/'crates/paisho-train/src/micro_learning/gen5/durable.rs').read_text()
    start=current.index('    // A relocated record may prefix')
    end=current.index('    let mut decoder',start)
    p=out/'crates/paisho-train/src/micro_learning/gen5/durable.rs';s=p.read_text()
    old='''    if sha256(&bytes) != expected {
        return Err(invalid("durable lesson hash mismatch"));
    }
'''
    if s.count(old)!=1:raise ValueError('Historical durable reader differs')
    p.write_text(s.replace(old,current[start:end]))
    # Expose offline verification in this build too; it never runs during games.
    module=out/'crates/paisho-train/src/micro_learning/gen5/mod.rs'
    module.write_text(module.read_text()+'\nmod portable_recovery;\npub use portable_recovery::{finalize as finalize_portable_recovery,verify as verify_portable_recovery};\n')
    probe=out/'crates/paisho-train/src/micro_learning/gen5/portable_recovery.rs'
    body=probe.read_text()
    body=body.replace('    if let Some(anchors)=&o.structured_recall_anchors {recall.restore_structured_anchors(anchors)?;}\n','')
    probe.write_text(body)
    receipt={'schema' :'paisho-gen5-historical-build-v1','algorithm':'last-campaign-2026-09-26',
      'frozen_overlay':overlay,'portable_io_only':['opaque original sequence exclusion IDs','original-order bundle names with verified content hashes','lossless compressed revision files'],
      'staged_draw_repair_active':False,'staged_structured_retention_active':False}
    (out/'historical-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'portable-models/gen5-history-engine');p.add_argument('--stage-only',action='store_true');a=p.parse_args()
    stage(a.output)
    if not a.stage_only:subprocess.run(['cargo','build','--manifest-path',str(a.output/'Cargo.toml'),'--release','--locked','-p','paisho-train','--bin','paisho-gen5'],cwd=ROOT,check=True)
