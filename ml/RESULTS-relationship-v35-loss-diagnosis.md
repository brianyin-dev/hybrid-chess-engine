# Diagnosis of the v35 20-game pilot losses

Large mistakes, not exclusively gradual positional deterioration, are a major cause. All 11 lost games contain a confirmed selected move losing at least 150 cp relative to Stockfish's preferred continuation or allowing forced mate. Ten contain a selected move losing at least 300 cp or allowing mate; some occur after earlier damage, so these counts do not mean ten games were initially lost by a single 300 cp blunder. Both players repeatedly gave chances back.

All NN moves in the 11 losses were screened at 16,000 Stockfish nodes. Candidates were reviewed at 128,000 nodes. The first flagged competitive mistake and largest flagged competitive mistake per game were independently rechecked at 512,000 nodes, including alternative choices from fixed-depth and equal-time searches. Full move histories were preserved. Scores below are from the NN player's perspective; 100 cp is one pawn. -150 cp is a warning threshold, not proof of a forced loss.

| Game | Played move | Confirmed best → played | What happened |
|---|---|---|---|
| 7 | Black 15...Rd8 | +4.47 → -3.19 | Missed a winning knight tactic; the alternative Neg4 attacks/captures the bishop. |
| 11 | Black 11...Nf5 | +1.25 → -8.62 | Left the bishop available to hxg4; further captures in the reviewed line lose additional material. Depth-zero fallback. |
| 12 | White 22.Nc4 | +7.30 → mate in 3 against White | Missed a forcing rook invasion: Rb1+, followed by Rxc1+ and Rxe1#. Actual search completed depth one. |
| 13 | White 11.Qc2 | +7.04 → +0.27 | Missed a winning combination beginning Bf4 and Nc7+, rather than simply hanging material. |
| 15 | Black 17...Nxc3 | -0.18 → mate in 2 against Black | Captured a knight instead of defending: Qxf7+ Kh8 Qxg7#. Depth-zero fallback. |
| 16 | White 30.Qxb7 | +4.07 → -2.79 | Unsafe pawn grab permits Qa1+ and Re8+, exposing the king and bishop. |
| 17 | White 76.Re7+ | -0.07 → mate in 9 against White | A rook check permits Kf6 followed by Kxe7; the alternative Rb7 maintains defensive checking resources. |
| 19 | Black 34...Ra7 | -0.44 → -5.20 | Rook-endgame activity/king penetration error; no immediate material drop in the reviewed line. |
| 20 | White 39.Rd1 | 0.00 → -7.71 | Missed defensive checking resources and allowed Qxd4+ to win a bishop. |

The shorter principal variations recorded in loss-diagnosis.json are 128,000-node illustrative lines; the numbers in the table are from loss-confirmation.json at 512,000 nodes. They are engine analyses, not guaranteed unique continuations.

Smaller errors also occur. For example, game 6's initial flagged recapture changed from a 67 cp drop and a -150 crossing at 128k nodes to an 82 cp drop ending at -134 cp at 512k. It no longer crosses that threshold. A later Rc8 still loses about 254 cp, involving an unfavorable forcing exchange sequence. Game 19's early fxe5 costs 176 cp without an immediate net material loss. This is a mixture of tactical and positional weaknesses.

## Separating depth from evaluation

At the first flagged position in each of the 11 losses, the frozen NN's fixed-depth-three choice has lower confirmed move loss than the actual game move in **8/11** cases. Examples include the bishop error in game 11, mate-in-two error in game 15 and winning combination in game 13. This is a diagnostic comparison at selected failures, not an independent strength test.

The NN's recorded first flagged mistakes completed depth zero in two cases, depth one in three, and depth two in six. Depth zero means no full first iteration completed before fallback, not literally no evaluation or tactical work. The full match averaged completed depth 2.586 for NN versus 3.401 for heuristic.

At equal depth three, the NN is not universally worse: in game 11 it selects the Stockfish move while the heuristic still makes a large error. But in game 13 it sacrifices 219 cp of the available advantage where the heuristic selects the best move. Both evaluators retain major errors at other traced positions, including game 17's early Ne5 and game 16's later Qxb7. Completing depth three does not fix all errors.

At 250 ms, the heuristic avoids mate in two in game 15, chooses the safe queen exchange in game 12's first case and retains the advantage in game 20's first case while the NN fails. These probes are single reruns and may vary with timing; they do not prove a purely causal runtime explanation.

## Practical implication

Prioritize reliable completion of a tactically safe root decision, especially in positions with expensive quiescence, rather than another broad retraining pass alone. Inspect the depth-zero fallback path and expensive root searches on games 11 and 15. Separately retain equal-depth regression cases such as game 13 for evaluator training/debugging. There are shared search/evaluation errors as well as NN-specific regressions.

No engine or model was modified. These are post-match diagnosis positions, not fresh holdouts; any future training on them must not count their improvement as unseen progress.

Artifacts: ml/artifacts/relationship-v35/user-pilot-20-250ms/loss-diagnosis.json and loss-confirmation.json.
Scripts: python -m ml.diagnose_relationship_v35_losses and python -m ml.confirm_relationship_v35_losses.
