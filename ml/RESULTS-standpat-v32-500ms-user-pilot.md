# Earlier v32 NN at 500 ms

**1 win, 0 draws, 9 losses: 10% score over 10 games against the unchanged strategic-v26 heuristic.** All games completed by checkmate; no errors, interruptions or unfinished games.

Candidate: v32 epoch 8, heuristic plus 25% bounded NN correction with the non-check stand-pat gate. Both players received 500 ms per move, identical search, LMR and depth cap 64. Sequential CPU lock; no pondering, score adjudication or clock forfeits. Maximum 600 plies after opening. No engine/model changes.

The five starting positions and color order exactly match the first ten games of the earlier v32 250 ms match and the v33 500 ms match. Each start is played with colors swapped; book disabled after the prefix. Prior Stockfish opening scores: +30, +46, +47, +33, +44 cp for White. These are repeated development starts, not fresh confirmation.

| Candidate and budget | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|
| Earlier v32, 250 ms, same ten games | 3 | 2 | 5 | 40% |
| Earlier v32, 500 ms | 1 | 0 | 9 | 10% |
| Newer v33, 500 ms | 1 | 1 | 8 | 15% |

Score counts a draw as half a win. The full earlier 20-game v32 match scored 35%; the controlled comparison above uses only the same first five opening pairs.

| Metric | Earlier v32 at 250 ms | Earlier v32 at 500 ms |
|---|---:|---:|
| Mean NN completed depth | 2.517 | 3.491 |
| Mean heuristic completed depth | 3.288 | 3.929 |
| NN depth-zero fallback moves | 4 | 0 |
| Heuristic depth-zero fallback moves | 0 | 0 |

Mean elapsed at 500 ms was 485.3 ms for the NN and 482.2 ms for the heuristic. Maximum cooperative deadline overrun was 62.8 ms and 50.1 ms respectively; no hard clock forfeits are imposed.

Extra time produced deeper searches and eliminated NN fallbacks, but did not produce an NN advantage in this sample. Both sides receive extra time. The result does not establish a general harmful effect of deeper search or an Elo estimate: ten games, repeated openings and time-dependent searches limit inference. Nor does the one-draw difference between the two 500 ms models establish a reliable model ranking. Production remains heuristic-only; no automatic extension.

Audit replays all ten legal histories and final results, parses all ten PGNs, verifies five color-swapped pairs, checks checkpoint/source hashes, and confirms identical starts and color assignments across the three matches. Artifacts: `ml/artifacts/standpat-v32/user-pilot-10-500ms/{report.json,games.pgn,audit.json,comparison.json}`. Runner: `python -m ml.run_standpat_v32_500ms_user_pilot`; existing output is protected against overwrite.
