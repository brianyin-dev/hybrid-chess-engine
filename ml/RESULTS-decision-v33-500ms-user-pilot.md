# User-requested 500 ms NN versus heuristic match

**NN result: 1 win, 1 draw, 8 losses over 10 games; 15% score, 10% raw win rate.** All games completed normally. No errors, interruptions or unfinished games.

Candidate: v33 tanh epoch 16, strategic-v26 plus 25% bounded non-check NN correction. Opponent: unchanged strategic-v26 heuristic alone. Same search, LMR, depth cap 64 and 500 ms per move for both sides; sequential CPU lock, no pondering. Automatic draws only; no score adjudication or hard clock forfeits. Maximum 600 engine-played plies after opening.

Five fixed starts from the earlier v32 250 ms match, each played with colors swapped. Their prior full-strength Stockfish start evaluations are +30, +46, +47, +33, +44 cp for White. Book disabled after the prefix. Starts are deliberately repeated, not fresh holdout confirmation. This model differs from the earlier 20-game 250 ms candidate, so the two matches cannot establish that doubling time made the same model stronger or weaker.

| Metric | New NN hybrid | Heuristic |
|---|---:|---:|
| Mean completed depth | 3.675 | 4.345 |
| Mean elapsed per move | 489.6 ms | 486.1 ms |
| Depth-zero fallback moves | 1 | 0 |
| Maximum budget overrun | 12.1 ms | 35.9 ms |

More time enabled deeper searches in this match, but did not produce a neural advantage. The sample is small and does not estimate Elo. One fallback occurred; it is recorded in audit.json and does not by itself explain the losses. In that search, 2,141 of 2,142 nodes were quiescence nodes, so the first depth iteration still exhausted the budget. Production remains heuristic-only. No automatic extension beyond 10 games.

Artifacts: `ml/artifacts/decision-v33/user-pilot-10-500ms/report.json`, `games.pgn`, `audit.json`. Audit replays all legal histories and outcomes, checks five identical-start color-swapped pairs, parses all 10 PGNs, and verifies immutable model/source hashes. Reproduction runner: `python -m ml.run_decision_v33_500ms_user_pilot`; existing output is protected against overwrite.
