"""Separate source identity; never invalidates the human-training checkpoint."""
import hashlib
from pathlib import Path
from paisho_ppo import storage
from paisho_ppo.engine import ROOT


def source_identity():
    paths=sorted(p for p in (ROOT/'paisho_selfplay').glob('*.py') if p.name!='benchmark.py')
    paths += sorted((ROOT/'native/vendor/paisho-core/src').rglob('*.rs'))
    paths += [ROOT/'native/vendor/paisho-core/Cargo.toml', ROOT/'training/recipe.json',
              ROOT/'training/opponents.json', ROOT/'training/paisho-league.rs']
    profile=ROOT/'selfplay/runtime-profile.json'
    if profile.exists():paths.append(profile)
    rows=[storage.source_identity()]+[
        str(p.relative_to(ROOT))+':'+hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest()
