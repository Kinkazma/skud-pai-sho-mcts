"""Score legal moves in an empty or recorded Skud Pai Sho position."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys

import torch

from paisho_ppo.config import Config
from paisho_ppo.model import TransformerPlayer, choose_device, pack

ROOT = Path(__file__).resolve().parent
TILES = ("", "R3", "R4", "R5", "W3", "W4", "W5", "L", "O", "R", "W", "K", "B")


def coordinate(index):
    return f"{index % 17 - 8},{8 - index // 17}"


def notation(action):
    family, tile, source, destination = action
    if family == 0:
        return f"plant {TILES[tile]} {coordinate(destination)}"
    if family == 1:
        return f"arrange {coordinate(source)} {coordinate(destination)}"
    if family == 2:
        return "skip-bonus"
    if family == 3:
        return f"accent {TILES[tile]} at {coordinate(destination)}"
    if family == 4:
        return f"accent {TILES[tile]} move {coordinate(source)} {coordinate(destination)}"
    if family == 5:
        return f"plant-special {TILES[tile]} {coordinate(destination)}"
    if family == 6:
        return f"bonus-plant {TILES[tile]} {coordinate(destination)}"
    raise ValueError(f"Unknown action family: {family}")


def request(engine, operation, **values):
    engine.stdin.write(json.dumps({"op": operation, **values}) + "\n")
    engine.stdin.flush()
    response = engine.stdout.readline()
    if not response:
        raise RuntimeError("The rules engine closed its output")
    reply = json.loads(response)
    if not reply["ok"]:
        raise ValueError(reply["error"])
    return reply["data"]


def read_position(config, record):
    target = ROOT / "native" / "target" / "release"
    binary = target / "paisho-selfplay-engine"
    if sys.platform == "win32":
        binary = binary.with_suffix(".exe")
    if not binary.is_file():
        raise ValueError("Run cargo build --release --locked from the native directory")
    with subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8") as engine:
        description = request(engine, "describe")
        if description["protocol"] != 1 or config.rules not in description["rules"]:
            raise ValueError("Incompatible rules engine")
        if record is None:
            observation, = request(engine, "reset_empty", rules=config.rules, slots=1)
        else:
            observation = request(engine, "load_record", record=record)
            if observation["rules"] != config.rules:
                raise ValueError(f"Expected a record using {config.rules}")
    return description["points"], observation


@torch.inference_mode()
def rank_moves(record, top, device):
    weights_dir = ROOT / "weights"
    metadata = json.loads((weights_dir / "model.json").read_text(encoding="utf-8"))
    config = Config.from_dict(metadata["config"])
    points, observation = read_position(config, record)
    result = {
        "rules": observation["rules"],
        "player": ("Host", "Guest")[observation["player"]],
        "decisions": len(observation["history"]),
        "legal_actions": len(observation["actions"]),
        "outcome": {-2: "ongoing", -1: "draw", 0: "host-win", 1: "guest-win"}[observation["outcome"]],
        "moves": [],
    }
    if result["outcome"] != "ongoing" or not observation["actions"]:
        return result

    model_path = weights_dir / metadata["weights"]
    with model_path.open("rb") as model_file:
        if file_digest(model_file, "sha256").hexdigest() != metadata["sha256"]:
            raise ValueError("The weights do not match the release checksum")
        model_file.seek(0)
        weights = torch.load(model_file, map_location="cpu", weights_only=True)
    model = TransformerPlayer(config, points)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    logits = model(pack([observation], device, config.context_moves))[0]
    probabilities = logits.softmax(dim=0).tolist()
    indices = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
    result["moves"] = [
        {"action": notation(observation["actions"][index]), "probability": probabilities[index]}
        for index in indices[:top]
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", help="PSR file, or '-' for standard input. Defaults to an empty board.")
    parser.add_argument("--top", type=int, default=1, help="Maximum number of moves to show.")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    try:
        if args.record == "-":
            record = sys.stdin.read()
        else:
            record = Path(args.record).read_text(encoding="utf-8") if args.record else None
        torch.set_num_threads(2)
        result = rank_moves(record, args.top, choose_device(args.device))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"GPT-SKUD: {error}\n")


if __name__ == "__main__":
    main()
