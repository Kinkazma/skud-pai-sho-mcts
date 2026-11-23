# Train GPT-SKUD

`train.py` runs pretraining on human games and mixed PPO training. The [training recipe](training/recipe.json) records the reference phase settings. Start with a fresh pretraining run or with the weights supplied on this branch.

## Corpus and opponents

Build the rules engine and set up the Python environment using the [README](README.md). Place a separate checkout of `main` from `skud-pai-sho-mcts` alongside this checkout. The tested revision is recorded in the recipe. From that checkout, install the two published asset groups.

```sh
python3 manage.py install --resource-profile human
python3 manage.py install --resource-profile gen3
```

The `human` group contains the 1,375 game records and index used for GPT-SKUD. The preparation step verifies the index checksum, replays the games to check the moves, and reconstructs the training and validation split. The `gen3` profile installs the `core-memory` group required by the Gen3.1–Gen3.5 models. The [opponent manifest](training/opponents.json) records the five model checksums and which generations use the solver.

## Pretraining

From the GPT-SKUD directory, set the path to the `main` checkout.

```sh
SKUD_UPSTREAM=../skud-pai-sho-mcts-main
.venv/bin/python train.py prepare-human --mcts-root "$SKUD_UPSTREAM"
.venv/bin/python train.py init --run runs/human
.venv/bin/python train.py pretrain --run runs/human --seconds 600 --epochs 100
```

Run `pretrain` again on the same directory to continue. The commands use the CPU by default. Use `--device mps` to select the GPU on a compatible Mac. `--steps` limits the number of updates for a short test.

## Mixed training

The adapter compiles the inference code from the `main` checkout and loads the models installed there. When starting from a human-pretrained run, all five opponents begin at one simulation per decision. Their search budgets advance independently based on their results.

```sh
.venv/bin/python train.py build-mcts --mcts-root "$SKUD_UPSTREAM"
.venv/bin/python train.py prepare-ppo --run runs/mixed --source runs/human --mcts-root "$SKUD_UPSTREAM"
.venv/bin/python train.py ppo --run runs/mixed --mcts-root "$SKUD_UPSTREAM" --seconds 600
```

To start from the supplied weights, replace `--source runs/human` with `--reference` when preparing PPO. This sets the search budgets recorded at the reference save and initializes Adam, the games, and the counters for a new run. The code runs the protocol with the supplied data and settings. The original training trajectory depended on the intermediate states of the campaign.

Training rotates between two checkpoint slots under `runs/`. Prepared data, binaries, and checkpoints are local files excluded from Git.
