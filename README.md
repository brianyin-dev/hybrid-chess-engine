# Hybrid Chess Engine

[![CI](https://github.com/brianyin-dev/hybrid-chess-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/brianyin-dev/hybrid-chess-engine/actions/workflows/ci.yml)

A browser chess app with a custom Python search engine and an experimental
neural evaluation pipeline. Built with **Python, Flask, JavaScript, PyTorch,
NumPy, and python-chess**.

The app uses the handcrafted evaluator. The neural evaluator is available for
experiments but has not demonstrated a repeatable playing-strength improvement.

![Chess app with board, controls, and move history](docs/images/local-game.png)

## Features

- Play White or Black, view move history, start a new game, and resign.
- Legal-move highlighting, thinking feedback, and game-end messages.
- Iterative-deepening alpha-beta/PVS search with quiescence, transposition tables,
  killer/history move ordering, and late-move reductions.
- Handcrafted evaluation of material, piece placement, pawn structure, mobility,
  development, and king threats.
- Polyglot opening-book integration and time-bounded search with safe interruption
  handling. python-chess provides board representation and legal moves.
- Flask move API with input validation, rate limiting, and concurrent-search limits.

## Run locally

Requires Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-engine.txt
.venv/bin/python books/install_gm2001.py
.venv/bin/python -m backend.app
```

Open **http://127.0.0.1:5000**. Select a piece, then a highlighted destination.
Pawns automatically promote to queens. The board adapts to desktop and mobile.

The app gives the engine **750 ms per move**, with a nominal search ceiling of
**eight plies**. Book lookup is enabled through at most ten full moves; a line
can leave the book earlier. Actual search depth depends on the position and
hardware. There is currently no hosted demo.

## Neural evaluation

The experimental evaluator learns a bounded correction to the handcrafted score:

**1,420 inputs → 64 → 32 → 1 output**, with **93,057 trainable parameters**.

Training uses Stockfish supervision, move-ranking and retention examples, and
splits separated by game and opening family. PyTorch trains the model; NumPy
runs inference. The final blend applies 25% of the neural correction, limiting
its contribution to approximately ±62.5 centipawns outside check.

Stockfish supplies training labels and benchmark opposition; it does not choose
the app's moves. See [evaluation results](docs/RESULTS.md) for benchmark conditions,
experimental findings, and limitations.

## Testing

The full suite includes ML tests, so install the research dependencies first:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

**166 local tests pass**, covering search correctness, interrupted-search
behavior, board restoration, API validation, and neural inference consistency.
GitHub Actions runs the suite on Python 3.10 and 3.13 with CPU-only PyTorch;
no GPU is required. Some artifact-dependent tests may skip without their checkpoints.

The interface was also checked on desktop and mobile, including engine replies,
move history, playing Black, reset, and resignation.

## Documentation

- [Results and limitations](docs/RESULTS.md)
- [Release notes](docs/RELEASE_NOTES.md)
- [Mobile screenshot](docs/images/local-game-mobile.png)
- [Opening-book provenance](books/README.md)

Automatic fivefold repetition and the 75-move rule are supported. Draw-claim
controls, underpromotion selection, online multiplayer, and accounts are outside
the current scope. Playing strength has not been assigned a verified human Elo.
