# Extended v35 NN versus heuristic pilot

**5 wins, 4 draws, 11 losses over 20 games: 35% score, 25% raw win rate.** The unchanged strategic-v26 heuristic was the opponent. Both sides had 250 ms per move. Candidate remained current-input v35 epoch 40, heuristic plus 25% bounded NN correction.

The original ten games (4W/3D/3L, 55% score) are preserved exactly. The user requested five additional opening pairs after seeing that result. These additional ten games scored **1W/1D/8L, 15% score**. The cumulative result includes the original sample; it is not an independent second comparison or a predeclared 20-game experiment.

Five additional fresh weighted standard-book starts used seed 350042, without filtering by engine scores or results. Each start was played with colors swapped, with book disabled after the prefix. No model, evaluation, search, correction weight or gate changed. Root aliases against v35 collection labels and roots were checked. Fresh starts do not imply every opening family is historically unseen.

All 20 games completed: 16 checkmates and four automatic fivefold-repetition draws. No unfinished games or errors. Sequential CPU lock, common LMR search, depth cap 64, evaluator reset each game, automatic draws only and 600-ply ceiling. Search uses a cooperative budget rather than hard clock forfeits.

| Metric | NN hybrid | Heuristic |
|---|---:|---:|
| Mean completed depth | 2.586 | 3.401 |
| Mean elapsed per move | 245.0 ms | 242.9 ms |
| Depth-zero fallback moves | 10 | 3 |
| Maximum budget overrun | 105.2 ms | 79.8 ms |

The audit verified all legal move replays, final outcomes, 20 PGNs, ten identical-start color-swapped pairs, checkpoint/source hashes and exact preservation of the first ten games. Full-strength Stockfish reviews of the additional starts, performed only after completion at 128,000 nodes, gave White +76, +31, +23, 0 and +25 cp. The original five reviews were retained.

This pilot does not establish an NN advantage and misses the 60% target. The first ten games' slight lead did not persist across added openings. The sample cannot isolate evaluation errors from search-depth costs. No Elo estimate; the app remains heuristic-only. No further expansion or retraining was performed.

Artifacts: ml/artifacts/relationship-v35/user-pilot-20-250ms/{report.json,games.pgn,audit.json}.
Reproduce in a fresh output directory using python -m ml.run_relationship_v35_user_pilot_20; audit with python -m ml.audit_relationship_v35_pilot_20. The runner refuses to overwrite recorded results.
