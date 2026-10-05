# User-requested current NN versus current heuristic pilot

20 games at 250 ms per move; 10 fresh weighted grandmaster-book starts, each played with both colors. No book after the prescribed 10–16-ply prefix. Sequential execution with CPU lock; common unchanged search, LMR enabled, depth cap 64, maximum 600 engine-played plies. Automatic draws only; no evaluation adjudication or clock forfeits.

Candidate: standpat-v32 epoch 8, strategic-v26 plus 25% bounded neural correction at non-check evaluations. Opponent: unchanged strategic-v26 heuristic alone. The user explicitly requested this exploratory match despite the model's prior failed acceptance gate. No retraining or search changes were made.

Candidate result: **6 wins, 2 draws, 12 losses; 35% score, 30% win rate.** All 20 completed; zero errors, interruptions, or unfinished games. This small pilot does not establish a neural advantage or an Elo rating. It is against the newer heuristic, unlike the historical v22 match against the older heuristic.

| Metric | NN hybrid | Heuristic |
|---|---:|---:|
| Average completed depth | 2.503 | 3.188 |
| Average elapsed per move | 245.1 ms | 244.0 ms |
| Depth-zero fallback moves | 8 | 1 |
| Maximum cooperative budget overrun | 30.8 ms | 43.9 ms |

The NN searched less deeply, but this match alone cannot separate evaluation errors from lost search depth. Production remains heuristic-only. No automatic extension to 100 games.

Results and legal replay/hash audit: `ml/artifacts/standpat-v32/user-pilot-20/report.json`, `games.pgn`, and `audit.json`. Frozen openings: `benchmarks/openings-standpat-v32-user-pilot.json` and its manifest. Initial FENs do not overlap the v32 static dataset; no claim of complete historical opening-family independence. Reproduction runner: `python -m ml.run_standpat_v32_user_pilot`; it refuses to overwrite its output directory.

## Post-match opening balance audit

Full-strength single-thread Stockfish 19, fresh hash/game for each start and each budget, 256,000 then 1,000,000 nodes. White-perspective scores at the larger budget:

| Pair | White score (cp) |
|---|---:|
| 1 | +30 |
| 2 | +46 |
| 3 | +47 |
| 4 | +33 |
| 5 | +44 |
| 6 | +8 |
| 7 | +37 |
| 8 | +41 |
| 9 | +20 |
| 10 | +19 |

All ten starts fall within +8 to +47 cp for White (mean +32.5 cp). None is a large opening advantage. Games ended in 10 White wins, 8 Black wins, and 2 draws. This audit does not establish zero opening/style effects, but finds no gross starting imbalance explaining the candidate’s 35% score.
