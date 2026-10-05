# Settled decision supervision and bounded output experiment, v33

The new tanh candidate slightly improves fresh development mean move loss, but adds a major timed error. The linear capped output does worse at equal time. Neither qualifies for test search or a game pilot; production remains the strategic-v26 heuristic. There is no demonstrated playing-strength improvement.

## What changed

The representation and network capacity remain 1420 inputs, 64/32 hidden units (93,057 parameters). Search, pruning, feature extraction, baseline, non-check gate, 25% blend and raw 250 cp correction bound are unchanged. Both candidates start with v32 epoch-8 hidden layers and a zero final head, with identical data, training seed, epochs and optimizer.

Two output formulations were tested: existing tanh and linear hard clipping to the same bound. The final evaluation correction is still about ±62.5 cp, with whole-score rounding. The linear formulation is a hypothesis, not an established improvement or a larger correction.

Ranking examples now require both a searched root decision and an appropriate search leaf:

1. Compare Stockfish's root candidate with the heuristic depth-2 choice and other Stockfish candidates.
2. Force each candidate through our full-window heuristic depth-2 trace and quiescence. Retain a nonterminal, non-check, score-bearing leaf whose heuristic evaluation matches the traced branch score. Full histories are preserved.
3. Require Stockfish to prefer the root move at 128k and 256k nodes, with at least 25 cp separation.
4. Require the resulting leaves to agree with that preference under static labels and 64k-node searched scores. Reject leaves whose best Stockfish continuation is a capture, promotion or check.
5. Remove prior aliases, duplicate endpoints and infeasible corrections. Keep one pair per root across middlegames/endgames and independent source games.

These filters reduce obvious target mismatches. They do not prove a leaf correction changes the whole minimax decision; the score-bearing path can change when the evaluator changes. Quiescence settling is also not a proof of tactical safety or perfect teacher labels. Direct terminal mates belong to search, rather than the static evaluator. The objectives do not apply searched root scores as static leaf targets.

Training uses a 20 cp ranking hinge, stronger protection for correct heuristic decisions, and supporting static residual targets capped at 50 cp in the final blend. A penalty discourages raw outputs outside that interior range. Both arms use the actual non-check gate and straight-through whole-score rounding. A post-fit capacity audit found one training pair can reach at most 18 cp against the 20 cp hinge target; its ranking is reversible, but the full target is unattainable. This creates a 2 cp residual and is recorded without retraining or tuning against exposed validation. Future training should cap target margins to each pair’s exact attainable margin. No additional layers or features were added.

## Data and independence

Weighted grandmaster-book starts were initially sampled at 10/12/14/16 plies; a prefit coverage supplement added unique 6/8/18/20-ply standard book prefixes in reserved families. Frozen final starts and original/coverage manifests are under `benchmarks/openings-decision-v33-data*`.

Initial 70 training, 30 validation and 30 test trajectories produced 94 training pairs and a validation screen capacity of only 22 positions across 11 families. Before any fitting or candidate predictions, a recorded coverage amendment added 26 training trajectories and six validation trajectories from three reserved families. Confirmation filters, minimum data requirements, architecture, output bounds and acceptance gate were not relaxed. Test sources were unchanged. Initial artifacts and amendment are preserved.

Final sources: 96 training, 36 validation, 30 test trajectories. They use Stockfish self-play or heuristic versus Stockfish, with truncated source trajectories allowed. They are data generation, not strength games.

| Split | Ranking pairs | Repair / retention | Contributing games | Contributing families |
|---|---:|---:|---:|---:|
| Training | 126 | 65 / 61 | 69 | 35 |
| Validation | 37 | 22 / 15 | 22 | 14 |
| Test | 47 | 24 / 23 | 22 | 13 |

Training also retains 2,000 old training-only static labels. New collection excludes canonical aliases of prior labels and the exposed v32 match. Whole six-ply canonical opening families own each split, retaining v32 assignments; current train/val/test family and position overlaps are zero. Legacy pretrained provenance is inherited: these are fresh positions/trajectories, not a claim of universally unseen historical opening families.

Training pairs include 32 advanced-pawn cases from 23 games, 66 pin cases from 48 games, 115 geometric king-pressure cases from 68 games, and two immediate-promotion cases. Tags overlap, and king pressure is a geometric feature, not proof of a decisive attack. Direct promotion supervision remains sparse; the dataset does not establish that this weakness is fixed.

## Training and saturation

One 24-epoch trajectory per arm, snapshots at 8/16/24. Development retention damage, ranking correctness, saturation, then earliest epoch select one snapshot per arm. Tanh selected epoch 16; linear clip selected epoch 24. No test predictions were used for selection.

| Validation metric | Heuristic | Old v32 | New tanh | New linear clip |
|---|---:|---:|---:|---:|
| Correct rankings / 37 | 15 | 18 | 18 | 17 |
| Repairs / 22 | 0 | 5 | 5 | 6 |
| Damaged correct rankings / 15 | 0 | 2 | 2 | 4 |
| Rounded corrections near bound / 74 endpoints | 0 | 22 | 0 | 13 |

Near bound means rounded correction magnitude >=60 cp. The tanh candidate removes near-bound outputs on these development endpoints, but does not improve total ranking correctness over old v32. Linear clipping has more saturation and damage than the retrained tanh control. Fewer saturated predictions are not themselves evidence of stronger chess.

At their selected snapshots, tanh learns 118/126 training rankings and linear clip 124/126, with no training retention regressions. The gap to development rankings remains large; training fit is not the primary success metric.

## Fresh searched comparison

Before fitting, freeze 24 validation positions from 24 distinct source games and at most two games per opening family, balancing colors/stages. Test has its own frozen 24 positions; no test model search was run after validation failed. Models are compared sequentially at exact depth 3 and 250 ms with common unchanged search. Stockfish reviews chosen moves at 256k nodes. Means use the same 23 non-mate-comparable positions for all models; error counts use all 24.

| Evaluator | Depth-3 mean move loss (cp) | 250-ms mean move loss (cp) | 250-ms major errors >=150 cp |
|---|---:|---:|---:|
| Heuristic | 32.48 | 75.48 | 3 |
| Old v32 | 40.78 | 65.52 | 3 |
| New tanh epoch 16 | 29.39 | 72.57 | 4 |
| New linear clip epoch 24 | 29.91 | 78.22 | 4 |

Lower move loss is better. This is a small development screen, not a game win rate or a statistical strength confirmation. The new tanh candidate improves means slightly relative to the heuristic, but is worse than old v32 at equal time and introduces an extra major error. Both fail the predeclared gate. No test search, new games, app adoption, or automatic extension followed.

At 250 ms, mean completed depths are heuristic 3.29, old v32 2.42, new tanh 2.38, and new clip 2.38. Mean times remain about 250 ms. This experiment does not optimize inference or prove runtime overhead is the sole cause of failures.

A concrete new timed failure occurs at `4r1k1/q1pb1pb1/2np2pp/1p6/3Pn3/1B2NN1P/1P3PP1/2BQR1K1 w - - 4 20`. The heuristic and old v32 complete depth 1 and choose `Nec2` (114 cp move loss). Both new candidates complete depth zero and use fallback `Qd3` (433 cp move loss). At depth 3 all four choose `Nec2`. This is evidence of a deadline/fallback failure in this case, not proof of a neural static error or intrinsically slower math. Evaluation changes can also change how much search work is done.

## Verification and next interpretation

151 unit tests pass. New tests cover both output rules/bounds, training/runtime parity, color symmetry, zero-head baseline equality, metadata rejection, check gate, and trace leaf score/history invariants. All trained snapshots pass inference checks on 120 legal positions with maximum Torch/runtime disagreement <=1 cp. Dataset audit replays full histories, verifies strategic scores, root/leaf ranking constraints and quiet settling rules, checks zero cross-split aliases/families, and confirms frozen engine/model hashes.

The meaningful result is a cleaner supervision experiment and a saturation ablation, not a stronger deployed NN. Saturation was reduced by the revised objective in the tanh arm without resolving generalization. The extra timed error gives a specific future search target: improve behavior when no full iteration completes, while preserving this experiment's models and untouched test set. Further promotion-specific data may also be needed, since immediate-promotion examples remain rare.

Artifacts: `ml/artifacts/decision-v33/`. Main runner: `python -m ml.run_decision_v33`; the recorded prefit extension is `python -m ml.extend_decision_v33_training`. Completed runs refuse overwrite. Read-only audit/description scripts are `ml.audit_decision_v33` and `ml.describe_decision_v33`. Final public report should not claim a neural win-rate improvement.
