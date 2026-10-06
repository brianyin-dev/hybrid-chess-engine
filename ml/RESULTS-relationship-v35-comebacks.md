# Comebacks and lost advantages in the v35 pilot

Yes: there was a successful comeback, and several losses involved major advantages squandered later. Full-strength Stockfish reviewed the NN-turn positions across all 20 games, with selected trajectory extrema rechecked at 512,000 nodes.

Game 3, which the NN won, had a confirmed -610 cp position before Black's move 43, then a forced mate in four available before move 86. This is a genuine recovery in the game's evaluation trajectory, helped by errors from the opposing heuristic; it does not by itself establish strong defensive play.

Five of the 11 lost games contained confirmed NN-turn positions above +300 cp:

| Lost game | Confirmed advantage available |
|---|---:|
| 7 | +810 cp |
| 12 | +730 cp |
| 13 | +764 cp |
| 16 | +702 cp |
| 17 | +602 cp |

Game 12 recovered from -326 cp before White's move 17 to +730 cp before move 22, then Nc4 allowed mate in three. Game 16 recovered from -158 cp before move 13 to +702 cp before move 19, but later lost. Game 13 also recovered from -180 cp before move 30 to +279 cp before move 34 and ultimately lost. The latter two are modest-disadvantage recoveries, not the same magnitude as game 3.

Game 1, drawn, had a confirmed +392 cp position. Other drawn games' selected peaks were +17, +267 and +220 cp. Not every draw was a squandered dominant advantage.

These scores measure the position with Stockfish's best continuation available to the NN, before its actual move. They do not imply the NN saw that advantage or chose the best move. The earlier loss report contains concrete before/after played-move comparisons. +/-300 cp denotes a large engine-assessed advantage, not a mathematical forced win.

Both sides repeatedly returned opportunities. Reaching winning positions is promising, but cannot isolate the NN's benefit: the opponent is also our heuristic with its own blunders. Reliable conversion and avoiding sudden losses remain major weaknesses.

Depth three is three plies of nominal search, not an exact universal 1.5-move horizon: quiescence continues captures and check evasions. Two initial critical mistakes completed no full depth-one iteration, including the mate-in-two loss. The same NN at fixed depth three defended that position. Other errors persisted at equal depth, so neither depth nor evaluation alone explains everything.

All NN-turn roots were screened at 16k nodes for previously unreviewed games; prior loss reviews were reused. Candidate peaks and the strongest screened disadvantage-to-advantage recovery endpoints per game were confirmed at512k with full histories. Selected extrema are a diagnostic sample, not a complete deep analysis of every move. No model/search changes or additional games were performed.

Artifact: ml/artifacts/relationship-v35/user-pilot-20-250ms/comeback-analysis.json.
Script: python -m ml.analyze_relationship_v35_comebacks.
