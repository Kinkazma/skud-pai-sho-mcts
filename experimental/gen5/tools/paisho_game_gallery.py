#!/usr/bin/env python3
"""Local PSR-only gallery and opt-in stable-generation backfill/watch helper.

Controller config: dashboard.examples_directory points to the gallery folder.
HTTP reads meta.json and selected best-game.psr only; it never opens replay shards
or invokes an exporter. Published layout: generation-NNN...NN/best-game.psr plus
internal meta.json. Legacy game.psr/metadata.json remains readable. Only PSR is offered for download/copy, never JSON.

Backfill uses the standalone offline exporter:
  python3 tools/paisho_game_gallery.py --campaign CAMPAIGN --examples EXAMPLES \
    --exporter /path/paisho-export-games --generation 59
It invokes --snapshot PATH --output DIR --generation N --behavior-producer HASH.
No shell is used. The exporter selects a compatible neural game (win, then draw,
then loss) or records no eligible game; both outcomes are retained without repeated export.
Also reads live flat highlights: generation-N.psr and .metadata/generation-N.json
(or .metadata/N.json). JSON stays internal and is never offered for download.
Use --watch-seconds N for explicit polling; default is one scan, --jobs defaults
to 1. No daemon installation, campaign edits, controller restart, or GPU use.
Run export work outside performance benchmark windows; polling alone is cheap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

GENERATION = re.compile(r"generation-([0-9]{1,20})\Z")
DOWNLOAD = re.compile(r"/examples/files/(generation-[0-9]{1,20})\.psr\Z")
METADATA_LIMIT = 65536


def read_member(root, generation, member, limit=None):
    """Fixed two-component paths with no symlink following, even during open."""
    if not GENERATION.fullmatch(generation) or member not in ("game.psr", "metadata.json", "best-game.psr", "meta.json"):
        raise ValueError("invalid gallery member")
    return read_relative(root, [generation, member], limit)


def read_relative(root, parts, limit=None):
    if any(not part or part in (".", "..") or "/" in part or "\\" in part for part in parts):
        raise ValueError("invalid gallery path")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    descriptors = [root_fd]
    try:
        for part in parts[:-1]:
            descriptors.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptors[-1]))
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptors[-1])
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("not a regular gallery file")
            data = stream.read() if limit is None else stream.read(limit + 1)
            if limit is not None and len(data) > limit:
                raise ValueError("gallery metadata too large")
            return data
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


class Gallery:
    def __init__(self, directory):
        self.directory = Path(directory).resolve() if directory is not None else None

    def entry(self, name):
        match = GENERATION.fullmatch(name)
        if not match or self.directory is None:
            return None
        path = self.directory / name
        try:
            if path.exists():
                try:
                    meta = json.loads(read_member(self.directory, name, "meta.json", METADATA_LIMIT))
                    psr_parts = [name, "best-game.psr"]
                except FileNotFoundError:
                    meta = json.loads(read_member(self.directory, name, "metadata.json", METADATA_LIMIT))
                    psr_parts = [name, "game.psr"]
            else:
                meta = None
                for filename in (name + ".json", match[1] + ".json", str(int(match[1])) + ".json"):
                    try:
                        meta = json.loads(read_relative(self.directory, [".metadata", filename], METADATA_LIMIT))
                        break
                    except FileNotFoundError:
                        continue
                if meta is None:
                    return None
                psr_parts = [name + ".psr"]
            if type(meta.get("generation")) is not int or meta["generation"] != int(match[1]):
                return None
            psr = self.directory.joinpath(*psr_parts)
            available = psr.is_file() and not psr.is_symlink()
            no_eligible = meta.get("status") == "no_eligible_psr" or ("selected" in meta and meta["selected"] is None)
            if not available and not no_eligible:
                return None
            return {**meta, "available": available, "psr_parts": psr_parts}
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def entries(self):
        if self.directory is None or not self.directory.is_dir():
            return []
        names = set()
        for path in self.directory.iterdir():
            if not path.is_symlink() and GENERATION.fullmatch(path.name):
                names.add(path.name)
            elif not path.is_symlink() and path.suffix == ".psr" and GENERATION.fullmatch(path.stem):
                names.add(path.stem)
        metadata_dir = self.directory / ".metadata"
        if metadata_dir.is_dir() and not metadata_dir.is_symlink():
            for path in metadata_dir.glob("*.json"):
                name = path.stem if GENERATION.fullmatch(path.stem) else "generation-" + path.stem
                if GENERATION.fullmatch(name):
                    names.add(name)
        entries = {}
        for name in sorted(names):
            meta = self.entry(name)
            if meta and (meta["generation"] not in entries or meta["available"]):
                entries[meta["generation"]] = (name, meta)
        return [entries[g] for g in sorted(entries, reverse=True)]

    def page(self, entries=None):
        rows = []
        for name, meta in (self.entries() if entries is None else entries):
            opponent = html.escape(str(meta.get("opponent", "—")))
            side = (meta.get("selected") or {}).get("neural_side")
            other = "Bot aléatoire" if meta.get("opponent") == "random" else (
                "Adversaire du réseau" if opponent == "—" else opponent)
            host, guest = (("Réseau neuronal", other) if side == "host" else
                           (other, "Réseau neuronal") if side == "guest" else
                           ("Non renseigné", "Non renseigné"))
            outcome = (meta.get("selected") or {}).get("terminal_outcome")
            winner = {"host-wins": f"Hôte — {host}",
                      "guest-wins": f"Visiteur — {guest}",
                      "draw": "Partie nulle"}.get(outcome, "Non renseigné")
            url = meta.get("download_url", "/examples/files/" + name + ".psr")
            action = (f'<a href="{url}" download>Télécharger PSR</a> '
                      f'<button data-psr="{url}">Copier le texte PSR</button>') if meta["available"] else "Aucune partie neuronale compatible à exporter."
            rows.append(f'<tr><th>Collecte G{meta["generation"]}</th><td>{host}</td><td>{guest}</td><td>{winner}</td><td>{action}</td></tr>')
        empty = "Aucun dossier d’exemples configuré." if self.directory is None else "Aucune partie publiée pour le moment."
        content = '<table><thead><tr><th>Génération de collecte</th><th>Hôte</th><th>Visiteur</th><th>Gagnant</th><th>Partie sélectionnée</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>' if rows else f'<p>{empty}</p>'
        return ('''<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pai Sho · Parties</title>
<style>body{font:16px system-ui;background:#101b14;color:#eeeada;margin:32px auto;padding:0 18px;max-width:1050px}
a{color:#9de5b4}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:14px;border-bottom:1px solid #395040}
button{margin:8px;padding:9px;background:#28432f;color:#fff;border:1px solid #57765e;border-radius:8px;cursor:pointer}
textarea{width:100%;min-height:220px}#status{min-height:2em}small{color:#bfcebd}</style>
<a href="/">Retour au contrôle</a><h1>Parties par génération</h1>
<p>Une partie neuronale sélectionnée par génération de collecte, avant son apprentissage.
Ce numéro ne désigne pas la génération du modèle après entraînement. Le PSR contient la partie à copier ou télécharger.</p>
<p>La sélection privilégie une victoire du réseau, puis le score final : ce n’est pas une mesure de qualité stratégique.
Le début peut être un préfixe aléatoire du curriculum, non joué par le réseau ; il est conservé intégralement pour restituer la partie exacte.</p>
<p><a href="/examples">Actualiser</a></p>''' + content + '''
<p id="status" role="status" aria-live="polite"></p><textarea id="copy" hidden readonly aria-label="Texte PSR à copier"></textarea>
<script>for(const button of document.querySelectorAll('button[data-psr]')){button.onclick=async()=>{
const status=document.getElementById('status');button.disabled=true;
try{const r=await fetch(button.dataset.psr,{cache:'no-store'});if(!r.ok)throw new Error('HTTP '+r.status);
const text=await r.text();try{await navigator.clipboard.writeText(text);status.textContent='PSR copié.';}
catch(e){const box=document.getElementById('copy');box.hidden=false;box.value=text;box.focus();box.select();status.textContent='Texte sélectionné : utilisez Copier (⌘C ou Ctrl+C).';}}
catch(e){status.textContent='Copie impossible : '+e.message;}finally{button.disabled=false;}}}</script></html>''').encode()

    def response(self, path):
        if path in ("/examples", "/examples/"):
            return 200, "text/html; charset=utf-8", self.page(), None
        match = DOWNLOAD.fullmatch(path)
        if match and self.directory is not None:
            try:
                name = match[1]
                meta = self.entry(name)
                if meta is None or not meta["available"]:
                    raise ValueError("unavailable game")
                return 200, "text/plain; charset=utf-8", read_relative(self.directory, meta["psr_parts"]), name + ".psr"
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return 404, "text/plain; charset=utf-8", b"Not found\n", None


class ArchiveGallery:
    """One recent-first table across campaigns; no replay shard reads."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def sources(self):
        paths = set()
        for pattern in ("game-examples", "live-curriculum-*/highlights",
                        "teacher-program-*/warmup-ppo/highlights",
                        "teacher-program-*/occurrence-*/ppo/highlights",
                        "teacher-program-*/post-goal-ppo/highlights"):
            paths.update(p for p in self.directory.glob(pattern)
                         if p.is_dir() and not p.is_symlink()
                         and p.resolve().is_relative_to(self.directory))
        return {hashlib.sha256(str(p.relative_to(self.directory)).encode()).hexdigest()[:16]: p
                for p in sorted(paths, reverse=True)}

    def response(self, path):
        sources = self.sources()
        if path in ("/examples", "/examples/"):
            entries = []
            for key, directory in sources.items():
                for name, meta in Gallery(directory).entries():
                    if not meta["available"]:
                        continue
                    try:
                        published = directory.joinpath(*meta["psr_parts"]).stat().st_mtime_ns
                    except OSError:
                        continue
                    url = "/examples/archive/" + key + "/files/" + name + ".psr"
                    entries.append((published, key, name, {**meta, "download_url": url}))
            # Generation numbers can restart after a migration. Publication time
            # puts the latest actual games first, even ahead of older higher Gs.
            entries.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            latest = {}
            for _, _, name, meta in entries:
                latest.setdefault(meta["generation"], (name, meta))
            page = Gallery(self.directory).page(list(latest.values()))
            return 200, "text/html; charset=utf-8", page, None
        match = re.fullmatch(r"/examples/archive/([0-9a-f]{16})(/files/generation-[0-9]{1,20}\.psr)?/?", path)
        if match and match[1] in sources:
            suffix = match[2] or ""
            result = Gallery(sources[match[1]]).response("/examples" + suffix)
            if not suffix and result[0] == 200:
                prefix = "/examples/archive/" + match[1]
                payload = result[2].decode().replace('/examples/files/', prefix + '/files/')
                payload = payload.replace('href="/examples">Actualiser', 'href="' + prefix + '">Actualiser')
                payload = payload.replace('Retour au contrôle</a>', 'Retour au contrôle</a> · <a href="/examples">Toutes les campagnes</a>')
                return result[0], result[1], payload.encode(), result[3]
            return result
        return 404, "text/plain; charset=utf-8", b"Not found\n", None


def small_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(METADATA_LIMIT + 1)
    if len(data) > METADATA_LIMIT:
        raise ValueError("metadata too large: " + str(path))
    return json.loads(data)


def completed_generations(campaign):
    """Read committed stage metadata, never replay content, to choose exact sources."""
    campaign = Path(campaign).resolve(strict=True)
    base = campaign / "generations" if (campaign / "generations").is_dir() else campaign
    for directory in sorted(base.glob("generation-*")):
        if not re.fullmatch(r"generation-[0-9]+", directory.name) or directory.is_symlink():
            continue
        if not (directory / "outcome.json").is_file():
            continue
        plan = small_json(directory / "plan.json")
        outcome = small_json(directory / "outcome.json")
        actor = small_json(directory / "actor-stage.json")
        if plan["format"] != "paisho-generation-plan-v1" or outcome["format"] != "paisho-generation-outcome-v1" or actor["format"] != "paisho-generation-actor-stage-v1":
            raise ValueError("unsupported generation archive")
        generation = int(directory.name.split("-")[1])
        if plan["payload"]["generation"] != generation or outcome["payload"]["generation"] != generation:
            raise ValueError("generation metadata mismatch")
        snapshot = (campaign / actor["payload"]["snapshot"]).resolve(strict=True)
        if not snapshot.is_relative_to((directory / "actors").resolve()) or snapshot.name != "snapshot.psrsnap":
            raise ValueError("actor snapshot is outside committed generation actors")
        yield {"generation": generation, "snapshot": str(snapshot), "plan": str(directory / "plan.json"),
               "producer": actor["payload"]["behavior_producer"], "opponent": plan["payload"]["actor"]["opponent"],
               "snapshot_sha256": actor["payload"]["snapshot_sha256"]}


def export_generation(examples, job, exporter):
    examples = Path(examples)
    name = f'generation-{job["generation"]:020}'
    destination = examples / name
    previous = Gallery(examples).entry(name)
    if previous:
        return {"generation": job["generation"], "status": "existing" if previous["available"] else "no_eligible_psr"}
    if destination.exists():
        raise ValueError("incomplete published destination: " + str(destination))
    attempts = examples / ".exports"
    attempts.mkdir(exist_ok=True)
    attempt = None
    # Recover a completed Rust export after a helper publication/layout failure.
    for old in sorted(attempts.glob(name + "-*")):
        try:
            meta = small_json(old / "output" / name / "meta.json")
            if small_json(old / "command.json")["source"] == job and (meta["selected"] is None or (old / "output" / name / "best-game.psr").is_file()):
                attempt = old
                break
        except (OSError, ValueError, KeyError):
            continue
    if attempt is None:
        attempt = Path(tempfile.mkdtemp(prefix=name + "-", dir=attempts))
        output = attempt / "output"
        argv = [str(exporter), "--snapshot", job["snapshot"], "--output", str(output),
                "--generation", str(job["generation"]), "--behavior-producer", job["producer"]]
        (attempt / "command.json").write_text(json.dumps({"argv": argv, "source": job}, indent=2))
        environment = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (attempt / "stdout.log").open("x") as out, (attempt / "stderr.log").open("x") as err:
            result = subprocess.run(argv, stdout=out, stderr=err, env=environment, check=False)
        if result.returncode:
            raise RuntimeError(f"exporter exited {result.returncode}; preserved {attempt}")
    output = attempt / "output" / name
    exported = small_json(output / "meta.json")
    if exported.get("format") != "paisho-local-neural-highlight-v1" or exported["generation"] != job["generation"] or exported["neural_producer"] != job["producer"]:
        raise ValueError("export metadata disagrees with committed collection")
    psr = output / "best-game.psr"
    selected = exported["selected"] is not None
    if selected and (not psr.is_file() or psr.is_symlink()):
        raise ValueError(f"exporter selected a game without publishing its PSR; preserved {attempt}")
    staging = Path(tempfile.mkdtemp(prefix="publish-", dir=attempt))
    checksum = None
    if selected:
        shutil.copyfile(psr, staging / "best-game.psr")
        with (staging / "best-game.psr").open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {**exported, **job, "psr_sha256": checksum,
                "status": "exported" if selected else "no_eligible_psr"}
    (staging / "meta.json").write_text(json.dumps(metadata, indent=2))
    staging.rename(destination)
    return {"generation": job["generation"], "status": metadata["status"]}


def backfill(campaign, examples, exporter, jobs=1, generations=None, allowed=None):
    if jobs <= 0:
        raise ValueError("positive jobs required")
    examples = Path(examples).resolve()
    examples.mkdir(parents=True, exist_ok=True)
    with (examples / ".backfill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work = [job for job in completed_generations(campaign)
                if generations is None or job["generation"] in generations]
        def export_if_allowed(job):
            if allowed is not None and not allowed():
                return {"generation": job["generation"], "status": "deferred_while_paused"}
            return export_generation(examples, job, exporter)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(export_if_allowed, work))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, default=Path(__file__).resolve().parents[1] / "target/release/paisho-export-games")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--watch-seconds", type=float, default=0)
    parser.add_argument("--generation", type=int, action="append", help="limit to these collection generations; repeatable")
    args = parser.parse_args()
    if args.watch_seconds < 0:
        parser.error("watch interval must be nonnegative")
    while True:
        print(json.dumps(backfill(args.campaign, args.examples, args.exporter, args.jobs,
                                  set(args.generation) if args.generation else None)), flush=True)
        if not args.watch_seconds:
            return
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"gallery: {error}", file=sys.stderr)
        sys.exit(1)
