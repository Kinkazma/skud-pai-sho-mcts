# GPT-SKUD

GPT-SKUD is an experimental 5.2-million-parameter Transformer for Skud Pai Sho. **It plays badly.** This branch contains the reference model and the code to run it. The network ranks the legal moves supplied by the rules engine without running a tree search.

## Run the model

Install Python 3.11 and Rust via rustup. Run these commands from this directory.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
(cd native && cargo build --release --locked)
.venv/bin/python play.py --top 3
```

The last command lists the three highest-ranked moves from an empty board, with the probability the model assigns to each. To load a position from a game recorded under `skud-pai-sho-2022-03-14-v2`, use `--record game.psr`. Use `--record -` to read the record from standard input. The program runs on the CPU by default.

The [model manifest](weights/model.json) contains the configuration and the SHA-256 checksum of the weights. The bundled rules engine retains its [MIT license](native/vendor/paisho-core/LICENSE).

## Training

The network sees the board from the side to move, the pieces in reserve, and up to 12,000 previous decisions. Pretraining asked it to predict the next move in human games. By the end, it matched 88.6% of moves in the training games and 23.2% in held-out games. The late rise in training accuracy, together with this gap, suggests overfitting.

Training continued with PPO, launching as many self-play games as games against Gen3.1–Gen3.5 MCTS opponents. The Transformer learned from its own decisions and the final game result. The reference model was saved during this mixed phase, at a total of 2,144 PPO optimizer steps and an entropy coefficient of 0.01.

The branch includes the pretraining code, the PPO code, and the opponent configuration. The corpus is available in the `human` asset group published on `main`. [Training instructions](TRAINING.md)

## The Gen3.2 matchup

Gen3.2 consistently beat GPT-SKUD while Gen3.4 and Gen3.5 were losing to it regularly. The table covers twenty minutes around the reference save. The network continued learning during these games.

| MCTS opponent | Simulations per decision (budget) | Wins | Draws | Losses | Stopped |
|---|---:|---:|---:|---:|---:|
| Gen3.1 | 1 | 4 | 0 | 13 | 0 |
| Gen3.2 | 1 | 0 | 0 | 17 | 0 |
| Gen3.3 | 1 | 8 | 0 | 8 | 0 |
| Gen3.4 | 2 | 11 | 0 | 6 | 0 |
| Gen3.5 | 2 | 15 | 0 | 2 | 0 |

A separate probe compared the moves chosen by Gen3.2 and Gen3.5. The engines chose different moves in 27 of 36 queries at each of the two search budgets, 1 and 8. Some positions appeared more than once. What led to the losses remains unresolved.

## Where training ended

Later trials varied the entropy coefficient, used earlier versions of the Transformer as opponents, alternated pure self-play with mixed training, and changed the reward tied to game length. With the revised reward, the median of the last 100 completed games fell from 531 to 504 decisions. In the final quarter of the trial, the median was back to 532. The completion rate for the full hour was lower than in the shorter preceding window. The revised reward was dropped.
