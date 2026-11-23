"""Local GPT-SKUD pretraining and mixed PPO entry points."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_metadata():
    return load_json(ROOT / "weights/model.json")


def recipe():
    return load_json(ROOT / "training/recipe.json")


def stage_inference(inference, overlay):
    """Reuse the upstream crates, with the historical empty-start rules."""
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
        shutil.copyfile(inference / name, overlay / name)
    for crate in sorted((inference / "crates").iterdir()):
        if not (crate / "Cargo.toml").is_file():
            continue
        destination = overlay / "crates" / crate.name
        destination.mkdir(parents=True, exist_ok=True)
        manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
        if crate.name == "paisho-train":
            bridge = json.dumps(str(ROOT / "training/paisho-league.rs"), ensure_ascii=False)
            manifest += f'\n[[bin]]\nname = "paisho-skud-league"\npath = {bridge}\n'
        (destination / "Cargo.toml").write_text(manifest, encoding="utf-8")
        for name in ("build.rs", "src"):
            source = crate / name
            if not source.exists():
                continue
            child = destination / name
            if crate.name == "paisho-core" and name == "src":
                child.mkdir(exist_ok=True)
                for path in sorted(source.glob("*.rs")):
                    target = child / path.name
                    if path.name in ("record.rs", "setup.rs"):
                        shutil.copyfile(ROOT / "native/vendor/paisho-core/src" / path.name, target)
                    elif not target.is_symlink():
                        target.symlink_to(path)
            elif not child.is_symlink():
                child.symlink_to(source, target_is_directory=source.is_dir())


def build_mcts(upstream):
    """Compile the adapter against the published inference workspace in place."""
    from paisho_ppo import budget

    upstream = upstream.resolve(strict=True)
    inference = upstream / "inference"
    target = ROOT / "native/target"
    key = sha256(os.fsencode(inference)).hexdigest()[:12]
    overlay = target / "mcts-workspace" / key
    for crate in ("paisho-ai", "paisho-core", "paisho-train"):
        if not (inference / "crates" / crate / "Cargo.toml").is_file():
            raise ValueError("Expected the published Gen3 inference workspace at --mcts-root")
    if not (inference / "Cargo.lock").is_file():
        raise ValueError("The upstream inference workspace is missing Cargo.lock")
    git_dir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                      cwd=inference, text=True).strip()
    with budget.project_write_guard():
        budget.check(reserve=300_000_000)
        overlay.mkdir(parents=True, exist_ok=True)
        stage_inference(inference, overlay)
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(overlay / "Cargo.toml"), "--target-dir", str(target),
                        "-p", "paisho-train", "--bin", "paisho-skud-league"], cwd=inference, check=True,
                       env={**os.environ, "GIT_DIR": git_dir, "GIT_WORK_TREE": str(upstream)})
        budget.check()


def prepare_human(upstream):
    from paisho_ppo import storage
    from paisho_ppo.human import prepare

    config = recipe()["human"]
    index = upstream / config["index"]
    if storage.digest(index) != config["index_sha256"]:
        raise ValueError("The human index differs from the one used for GPT-SKUD")
    return prepare(index, config["validation_fraction"], config["split_seed"])


def start_ppo(directory, source, mcts_root, device):
    import torch
    from paisho_ppo import storage
    from paisho_ppo.config import Config
    from paisho_selfplay.engine import SelfplayEngine
    from paisho_selfplay.league import BUDGETS, Population, initial
    from paisho_selfplay.opponents import identity
    from paisho_selfplay.trainer import SelfplayTrainer, runtime

    os.environ["GPT_SKUD_MCTS_ROOT"] = str(mcts_root.resolve(strict=True))
    opponent_identity = identity()
    metadata = reference_metadata()
    settings = recipe()["mixed_ppo"]
    config = replace(Config.from_dict(metadata["config"]), entropy_weight=settings["entropy_weight"])
    if source is None:
        path = ROOT / "weights" / metadata["weights"]
        if storage.digest(path) != metadata["sha256"]:
            raise ValueError("The reference weights do not match the release checksum")
        weights = torch.load(path, map_location="cpu", weights_only=True)
    else:
        with storage.run_lock(source):
            saved, _, _ = storage.load(source)
            if saved["training_stage"] != "pretrain" or not saved["pretraining"]["steps"]:
                raise ValueError("A checkpoint with at least one human pretraining update is required")
            weights = saved["model"]

    with storage.run_lock(directory), SelfplayEngine(workers=runtime()["engine_workers"]) as engine:
        if any(p.name != ".lock" for p in directory.iterdir()):
            raise ValueError("Choose a new directory for the mixed PPO run")
        trainer = SelfplayTrainer(config, engine, device)
        trainer.model.load_state_dict(weights, strict=True)
        trainer.training_stage = "ppo"
        trainer.reward_profile = settings["reward_profile"]
        state = initial()
        state["opponent_identity"] = opponent_identity
        if source is None:
            for generation, simulations in settings["budgets_at_reference"].items():
                state["ladders"][generation]["index"] = BUDGETS.index(simulations)
        trainer.population = Population(state, trainer.population_rng)
        return storage.save(directory, trainer.payload(), trainer.summary(), config.storage_limit_mb)


def run_ppo(directory, mcts_root, seconds, device):
    from paisho_ppo import storage
    from paisho_selfplay.engine import SelfplayEngine
    from paisho_selfplay.trainer import SelfplayTrainer, runtime

    os.environ["GPT_SKUD_MCTS_ROOT"] = str(mcts_root.resolve(strict=True))
    with storage.run_lock(directory), SelfplayEngine(workers=runtime()["engine_workers"]) as engine:
        trainer = SelfplayTrainer.restore(directory, engine, device)
        trainer.run(directory, seconds)


def positive_integer(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def duration(raw):
    value = float(raw)
    if not math.isfinite(value) or not 0 < value <= 604800:
        raise argparse.ArgumentTypeError("Choose a duration in seconds, up to seven days")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser("prepare-human")
    build = commands.add_parser("build-mcts")
    init = commands.add_parser("init")
    pretrain = commands.add_parser("pretrain")
    prepare_ppo = commands.add_parser("prepare-ppo")
    ppo = commands.add_parser("ppo")
    for command in (corpus, build, prepare_ppo, ppo):
        command.add_argument("--mcts-root", type=Path, required=True, help="Published Gen3 checkout")
    for command in (init, pretrain, prepare_ppo, ppo):
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    for command in (pretrain, ppo):
        command.add_argument("--seconds", type=duration, required=True)
    pretrain.add_argument("--epochs", type=positive_integer)
    pretrain.add_argument("--steps", type=positive_integer)
    seed = prepare_ppo.add_mutually_exclusive_group(required=True)
    seed.add_argument("--source", type=Path, help="Run with human pretrained weights")
    seed.add_argument("--reference", action="store_true", help="Use the published model weights")
    args = parser.parse_args()
    try:
        if args.command == "prepare-human":
            result = prepare_human(args.mcts_root)
        elif args.command == "build-mcts":
            build_mcts(args.mcts_root)
            return
        elif args.command == "init":
            from paisho_ppo.config import Config
            from paisho_ppo.trainer import initialize
            result = initialize(args.run, Config.from_dict(reference_metadata()["config"]), args.device)
        elif args.command == "pretrain":
            from paisho_ppo.pretrain import pretrain as execute
            execute(args.run, args.seconds, args.device, epochs=args.epochs, max_steps=args.steps)
            return
        elif args.command == "prepare-ppo":
            result = start_ppo(args.run, args.source, args.mcts_root, args.device)
        else:
            run_ppo(args.run, args.mcts_root, args.seconds, args.device)
            return
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"GPT-SKUD: {error}\n")


if __name__ == "__main__":
    main()
