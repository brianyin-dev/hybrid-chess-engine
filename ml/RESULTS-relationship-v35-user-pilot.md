# User-requested v35 NN versus heuristic match

**4 wins, 3 draws, 3 losses over 10 games: 55% score, 40% raw win rate.** Opponent: unchanged strategic-v26 heuristic alone. Candidate: current-input v35 epoch 40, heuristic plus 25% bounded NN correction, not the augmented-feature candidate. Both sides receive 250 ms per move.

This explicitly requested exploratory match follows the failed original position-test gate. No model, search or gate was tuned. Five fresh weighted standard-book starts, seed 350041, each played with colors swapped; alternate first color per pair. The book is disabled after the prefix. Starts are new relative to earlier benchmark starts and do not alias v35 labels/roots; not every opening family is historically unseen. Independent full-strength Stockfish reviews after the completed match found +60, +15, +13, +14, +37 cp for White at 128,000 nodes. No score-based opening filtering occurred.

All ten games completed normally: seven checkmates and three automatic fivefold-repetition draws. No errors, interruptions or unfinished games. No score adjudication, pondering or hard clock forfeits; maximum 600 engine-played plies after opening. Common unchanged search, LMR, depth cap 64, per-game evaluator reset and sequential CPU lock.

| Metric | New NN hybrid | Heuristic |
|---|---:|---:|
| Mean completed depth | 2.745 | 3.712 |
| Mean elapsed per move | 246.2 ms | 244.5 ms |
| Depth-zero fallback moves | 4 | 1 |
| Maximum cooperative-budget overrun | 50.0 ms | 28.6 ms |

NN as White: 1 win, 3 draws, 1 loss (50% score). As Black: 3 wins, 0 draws, 2 losses (60% score). These five-game color samples do not establish a color-specific strength difference.

A slight pilot lead is encouraging, but below the 60% target and too small to establish a repeatable advantage. Earlier model matches used different openings or budgets; their percentages are not controlled proof that this checkpoint is stronger. No Elo estimate and no automatic extension beyond ten games. Production remains heuristic-only; the original acceptance criteria are unchanged.

Audit replays every move and final outcome, parses all ten PGNs, checks five identical-start color-swapped pairs, verifies checkpoint/source hashes, and records fallback positions and post-match opening reviews. Artifacts: `ml/artifacts/relationship-v35/user-pilot-10-250ms/{report.json,games.pgn,audit.json}`. Runner: `python -m ml.run_relationship_v35_user_pilot`; auditor: `python -m ml.audit_relationship_v35_pilot`. Recorded match output is protected against overwrite.
