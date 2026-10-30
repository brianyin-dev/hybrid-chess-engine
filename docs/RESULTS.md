# Results and interpretation

## Classical engine

An earlier classical version scored **63 wins, 2 draws and 35 losses (64% score)**
against Stockfish 19 configured at UCI_Elo 1500 across 100 games at 250 ms per move.
Against the 1600 setting the same version scored 36W/1D/63L (36.5%).
Each tournament used 50 prescribed opening lines, played with both colors,
one Stockfish thread, 32 MB hash and automatic outcomes. The 1500 run reused the 1600
opening list. Approximate opening-pair bootstrap 95% intervals were 54.5–73.5%
and 28–45.5%, respectively.

These are local historical benchmarks, not the strength of the latest NN,
not a human Elo rating and not a fresh 100-game measurement of strategic-v26.
[Archived report summaries and hashes](historical-benchmark-summary.json) preserve
the local source records without modifying those pre-existing benchmark files.
The 1600 run had a brief startup overlap with a shutting-down prior process;
timings are hardware/load dependent.

## Neural experiments

| Configuration | Opponent | Games | W/D/L | Score |
|---|---|---:|---|---:|
| v8 with incremental inference, v16 screen | Optimized earlier heuristic | 20 | 10/2/8 | 55% |
| v35 current-input epoch40 | Strategic-v26 heuristic | 20 | 5/4/11 | 35% |

Both used 250 ms per move. Opponents and openings differ, so this is not a
controlled head-to-head ranking of the two neural configurations.
The v16 result is an exploratory correction-weight screen, not a confirmed
improvement. An earlier eight-game v8 pilot's 68.75% lead failed to reproduce in
larger old-model checks. The current app remains heuristic-only.

The v35 training experiment used 262 source games with whole-game/opening-family
splits, 179 training ranking pairs and 2,590 supporting score rows.
Learning training rankings did not generalize consistently to searched decisions
or equal-time games. Added relationship features did not establish an advantage.

- [v16 controlled weight experiment](../ml/RESULTS-v16.md)
- [v35 representation/training experiment](../ml/RESULTS-relationship-v35.md)
- [v35 full 20-game pilot](../ml/RESULTS-relationship-v35-user-pilot-20.md)
- [Loss diagnosis](../ml/RESULTS-relationship-v35-loss-diagnosis.md)
- [Comebacks and lost advantages](../ml/RESULTS-relationship-v35-comebacks.md)

## Final timeout fix

The model and evaluation remain unchanged. Search now retains the best fully
completed improving root alternative when its first iteration is interrupted.
It reports depth 0 and score null until the whole iteration completes.
It retains the last completed iteration on later interruptions.

Two known failure positions were replayed three times each at 250 ms:
the pre-fix search returned its recorded emergency blunder every time; the fixed
search avoided both recorded blunders in all six reruns. In game 11 it chose
Bxf3; in game 15 it chose Qf6 twice and g6 once. Stockfish 512k-node reviews
confirmed the replacement choices avoid the diagnosed large mistake/mate.

A separate screen contains 24 new positions from 12 fresh book-start games,
sampled after 8 and 16 plies of 8k-node Stockfish continuations. Fixed-depth-one
move/score results match exactly before and after. Timed choices agree on 23/24;
the remaining difference was 11 cp versus 5 cp move loss, with different completed
depths due to timing. No fresh-screen move newly allowed mate or incurred a
>=150 cp additional loss. This is a small diagnostic cohort, not a strength proof.

At the user's request, the game pilot was shortened from 20 to **five completed
games**: **1 win, 3 draws, 1 loss (50% score)**. Both sides used the same frozen
v35 checkpoint and 250 ms, comparing corrected search with archived pre-fix search.
There are two complete opening pairs plus one unmatched White game. A one-ply
sixth-game fragment was interrupted by the stop monitor, preserved in the raw
run and excluded from all reported scores. No further expansion was performed.

Mean completed depth was 2.128 fixed versus 2.111 pre-fix; mean elapsed time 249.3
versus 249.1 ms. The fixed side used partial-root results four times. Those counts
do not establish a depth/speed gain. Local UI/API smoke checks briefly overlapped
the exploratory pilot. Five games provide **no reliable strength estimate**.

- [Screen evidence](../ml/artifacts/timeout-wrapup/screen.json)
- [Five-game report](../ml/artifacts/timeout-wrapup/pilot-5/report.json)
- [Five-game PGNs](../ml/artifacts/timeout-wrapup/pilot-5/games.pgn)
- [Replay/hash audit](../ml/artifacts/timeout-wrapup/pilot-5/audit.json)
- [Original profiling](../ml/RESULTS-relationship-v35-fallback-profile.md)

## Verification

166 local automated tests pass with ML dependencies installed. Tests include
PVS re-search interruption, fail-low handling, preservation of completed
iterations, no-completed-root fallback, fixed-depth parity and history restoration.
The two recorded failures use deterministic node budgets in tests so host timing
does not decide whether the tests pass.

The browser was verified at 1360×900 and 390×844: legal move/engine response,
history, reset, playing Black, resignation and closing the victory dialog.
No JavaScript errors or horizontal mobile overflow were observed.
Screenshots show the actual heuristic app, not neural benchmark play.

An earlier neural-feature optimization reduced median standalone evaluation
cost from 0.1752 ms to 0.1137 ms (~35%) with feature parity. That is a historical
microbenchmark, not the latest model's latency or a 35% game-strength gain.
