# Evaluator calibration after the 500 ms losses

**Implemented and tested a conservative neural evaluator. It removes the old NN's fixed-depth regressions on the development screen, but does not beat the heuristic. It is not adopted.** Production remains strategic-v26, with no neural evaluator.

## Loss diagnosis

Reviewed the first Stockfish-confirmed competitive mistake in each of the earlier v32 500 ms match's nine losses. Screening uses 16,000 nodes; confirmation and move reviews use 128,000 nodes, full game histories, and the first competitive-error criteria previously used in v23. A -150 cp crossing is a diagnostic flag, not proof of a forced loss. These repeated match positions are diagnostic, not a holdout.

At fixed depth 3, the NN and heuristic choose the same move at five of these nine positions. The NN chooses a lower-regret move at one and a higher-regret move at three. This is mixed evidence: the evaluator causes some regressions, but not every loss. Score-bearing full-window depth-3/quiescence traces produced 40 distinct leaf positions; the old NN applied a correction of at least 60 cp in magnitude at 18 of them. Near-bound corrections are common here, not proven to cause every mistake.

## Concrete change

The v34 candidate retains the 1,420-input, 64/32 hidden representation from v32 epoch 8. It freezes every hidden weight and trains only the final 33 output parameters, starting that head at zero. It reduces the final correction bound from approximately +/-62.5 cp to +/-25 cp. The non-check gate, heuristic, search, features and runtime architecture remain unchanged.

Training uses 110 feasible Stockfish-confirmed ranking pairs from the existing v33 training split and 2,252 score-support positions, including 2,000 ordinary legacy training positions. Sixteen repair pairs cannot be reversed within the smaller bound and are excluded from ranking training; they remain in full training/validation metrics. Repair targets use a 10 cp margin capped by the attainable margin with a safety allowance. Correct heuristic decisions receive stronger retention protection. Supporting score residuals are clipped to +/-15 cp; penalties discourage large corrections.

Three correction penalties (0.1, 1, 10), 120 epochs each, snapshots every 20 epochs, and a zero-head control were frozen before fitting. Development static rankings selected penalty 1, epoch 20: 16/37 correct, versus the heuristic's 15/37, with one repaired ranking and no damage to its 15 correct rankings. Only 61/126 training rankings are correct for that snapshot; the retention-first selector deliberately preferred its development behavior. No loss-diagnosis positions were added to training. This is calibration on reused development data, not a new independent dataset.

## Searched decisions

Fixed pre-existing v33 development roots: 24 positions, with 23 common nonmate cases used for mean move loss. Major-error counts include all 24. Stockfish reviews use 256,000 nodes. Lower move loss is better; these figures are not game scores or Elo.

| Evaluator | Depth-3 mean move loss | 250 ms mean move loss | Major errors at 250 ms |
|---|---:|---:|---:|
| Unchanged heuristic | 32.48 cp | 75.48 cp | 3 |
| Earlier v32 NN | 40.78 cp | 65.52 cp | 3 |
| Calibrated v34 | 32.48 cp | 90.91 cp | 4 |

The candidate chooses exactly the heuristic's move at all 24 depth-3 roots. Thus it removes the old NN's regressions on this screen without adding useful move changes. Its maximum correction on the 40 diagnostic leaves is only 3 cp, far below its permitted bound; conservative calibration has effectively damped the correction. At 250 ms, mean completed depth is 2.375 for the candidate versus 3.292 for the heuristic (old NN 2.333). The small corrections still incur the full feature/inference cost. No playing-strength improvement is demonstrated.

A first searched screen overlapped unit-test activity; its results are preserved. A frozen clean rerun under the CPU lock, without concurrent tests, produced the same summary. No retraining or model selection changed between screens. The candidate failed the development gate, so the pre-existing untouched v33 test roots and game pilot were not run. Do not treat the better old-NN timed mean alone as demonstrated game strength: its direct matches remain poor.

## Verification and artifacts

All 154 unit tests pass when run without a benchmark. The initial suite had one benchmark-CLI timeout because it waited for the occupied CPU lock; the clean rerun passed. New tests cover frozen parameter integrity, attainable targets, training/runtime parity, +/-25 cp bounds, color symmetry, zero correction, check/terminal behavior and metadata rejection. Inference audits check all 19 checkpoints. Provenance audit verifies hidden weights remain bit-identical to v32, training histories and model/data/source hashes, and zero training-family or canonical-position overlap with validation/test.

Artifacts: `ml/artifacts/evaluator-v34/` contains loss traces, frozen protocol, training records/checkpoints, inference/provenance audits, original and clean searched screens, and adoption decision. Reproduction: `python -m ml.diagnose_evaluator_v34`, `python -m ml.run_evaluator_v34`, `python -m ml.rescreen_evaluator_v34`, `python -m ml.audit_evaluator_v34`. Existing outputs are protected against overwrite.

The result narrows the problem: suppressing large corrections can preserve heuristic decisions, but a useful residual still has to generalize. Repeating this head-only calibration with a looser bound is not yet justified by these data. The next evaluator experiment should improve the learned representation or supervision enough to produce useful unseen fixed-depth move changes, then satisfy the time-budget gate.
