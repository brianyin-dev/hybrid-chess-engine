# v31: square relationships, quiet-gate audit and retention protection

## Result

Implemented the representation experiment, trained one candidate trajectory and
selected epoch 16 on fresh searched validation. It failed the separate test, so
the conditional 20-game pilot did not run. The app remains strategic-v26
heuristic-only. No playing-strength gain, win percentage or Elo is claimed.

| Searched screen | Comparable roots | Heuristic move loss | Selected hybrid move loss |
| --- | ---: | ---: | ---: |
| Fresh validation, depth 3 | 24 | 48.50 cp | 41.08 cp |
| Fresh validation, 250 ms | 23 | 13.00 cp | 10.09 cp |
| Fresh test, depth 3 | 24 | 57.54 cp | 62.67 cp |
| Fresh test, 250 ms | 24 | 72.75 cp | 107.83 cp |

Lower is better. All screens contain 24 roots; a mate-scored validation root is
excluded from the timed means shared by all five evaluators. Both selected hybrid
and heuristic allowed a mate in one validation case. Error counts cover all roots.
The test at 250 ms has five heuristic errors of at least 150 cp versus seven
hybrid errors, and two versus four losing transitions; neither newly allows mate.
At depth 3 both have four large errors and two losing transitions, but the hybrid
has worse average move loss. Validation results did not repeat on independent
positions, and different opening sets prevent comparing these means directly to
v30's means as a strength trend.

The hybrid completed a shallower iteration in 10 of 24 timed test roots and the
same depth in 14, never a deeper iteration. Average completed depth is 2.83 versus
3.25. It chooses a different move on two depth-three roots and eight timed roots.
This is consistent with search-budget pressure, but equal-depth regression and
endpoint-ranking regression show that inference speed is not the only concern.

## What changed

Added an experimental 1,420-input network with unchanged 64/32 hidden layers:

- Existing 892 inputs remain intact.
- Four 64-square channels per color encode pin-aware attacker counts, defender
  counts, absolute pins, and attacks by cheaper pieces at each occupied square.
- Eight king features per color encode ring pressure by attacker type, attacks
  on the king and attacked/undefended ring squares.

Pinned attackers count only along their absolute pin ray. This is pin-aware
pseudo-attack information, not full legal capture analysis or static exchange
evaluation. Kings retain geometric attacks; en passant and discovered king
exposure are not simulated. The feature transform, tensor model and NumPy engine
adapter use the same color/rank mirror convention.

Warm-started v30 epoch 24. New first-layer input weights start at zero, initially
preserving previous predictions; the final audit confirms exact rounded parity
on 100 positions. Learned extra-input weight norms are nonzero at every snapshot.
The 25% quiet correction, raw 250 cp bound and integer rounding are unchanged.
No existing evaluator, search implementation or production app configuration was
modified. New checkpoint loading is separate from legacy checkpoint loading.

Retention loss protection increases from coefficient 4 to 10 and the protected
positive margin cap from 10 to 20 cp. Correction regularization increases from
.01 to .02, learning rate from .0001 to .0002. Repair ranking loss remains weighted
threefold. These changes and the representation are a combined experiment, not
an isolated causal feature ablation.

## Quiet-gate audit

Instrumented the actual baseline iterative depth-three LMR search on the exposed
24 v30 test roots, counting every static call including cache hits. Chosen moves
match uninstrumented search on all roots.

- 91,785 total static calls.
- 7,921 gate-active calls: **8.63%**.
- 83,864 blocked calls because the mover has a legal capture.
- No in-check static calls: quiescence searches evasions instead of standing pat.

These are static-call counts, not a fraction of all nodes or completed terminal
quiescence leaves. Static stand-pat scores can contribute before captures have
settled, including cutoff cases; those positions are often blocked by the gate.
The fraction is descriptive for this diagnostic set, not a universal activation
rate, and it is measured on baseline search rather than a hybrid-induced tree.

Traced full-window depth-three forced branches for the four exposed roots with
at least 150 cp heuristic timed error. Three have both score-bearing endpoints
blocked by legal captures; one has both active. Preference gaps are 33, 39, 118
and -1 cp, within the global 126 cp two-endpoint correction range. The traced
endpoints' scores cannot change when gated off, but other search leaves could
change a root choice. Forced traces disable TT/LMR and are not the same tree as
timed search. Therefore this does not prove three root moves are unrepairable or
that simply removing the gate will improve strength. Diagnostic teacher endpoint
reviews use 64,000 nodes. These exposed roots do not enter new fitting or selection.

## Training and fresh evaluation

Reused only v30 training labels: 300 pairs (114 repairs / 186 retention), 600
fresh endpoint labels sampled threefold plus 2,000 ordinary retained score labels
(3,800 score samples, not unique positions). V30 validation/test labels do not
enter fitting. Trained the exact rounded, quiet-gated hybrid for 24 epochs, with
only predeclared checkpoints 8, 16 and 24 eligible. Selection is by searched
validation, not endpoint metrics or training accuracy.

Generated 320 new weighted GM book starts, seed 310031. Excluding all known
v29/v30 source families leaves 70 starts across 48 new six-ply canonical families.
Teacher self-play uses 3,000 nodes per move, up to 120 additional plies. Fixed
snapshots at source plies 16/48/80/112 with alternating color offsets receive
64,000-node teacher reviews. Eligible roots have nonmate White scores within
800 cp and no aliases in known training/earlier screen positions.

Whole families own validation or test, and the roots are frozen before fitting.
Teacher-only coverage selection yields:

| Split | Roots / distinct games / families | Middlegame / endgame | White / Black |
| --- | --- | --- | --- |
| Validation | 24 / 24 / 24 | 13 / 11 | 13 / 11 |
| Test | 24 / 24 / 24 | 12 / 12 | 14 / 10 |

Search choices use full game histories, depth 3 or 250 ms/depth 64, unchanged LMR,
rotating evaluator order and exclusive CPU access. Distinct chosen moves get
256,000-node Stockfish best/forced reviews. Require at least 20 comparable roots,
nonworse equal-depth mean, strictly better equal-time mean and no additional
large errors, losing transitions or allowed mates. Epoch 16 passed validation;
epoch 24 did not. Only epoch 16 was examined on the fresh test, once, with no
post-test adjustment. Failed test prevents games and app adoption.

## Learning versus generalization

| Exposed endpoint set | Previous v30 correct | Selected epoch-16 correct | Previous repairs | New repairs | Previous / new retention regressions |
| --- | ---: | ---: | --- | --- | --- |
| Training, 300 | 230 | 254 | 47 / 114 | 68 / 114 | 3 / 0 |
| V30 validation, 60 | 41 | 35 | 13 / 24 | 7 / 24 | 8 / 8 |

Epoch 24 reached 265/300 training rankings, repairing 79/114 without training
retention regressions. Stronger protection worked on training examples, but did
not reduce the selected checkpoint's exposed held-out retention regressions.
Endpoint metrics were computed after searched selection/test and never selected
the model. V30 validation is exposed historical data, not this round's fresh
test evidence. The selected model still fails to generalize usefully.

## Verification and execution notes

All 141 tests pass, including pin-direction, spatial cheaper-attacker, mirror,
runtime/Torch parity, board preservation and checkpoint/blend compatibility.
The final audit validates all 70 legal source histories, zero cross-split alias
and family overlap, zero known alias/v29-v30 source-family overlap, frozen source
and previous checkpoint hashes, and exact engine/Torch rounded parity on 100
positions for each checkpoint. Legacy training-family provenance is incomplete;
global independence from every historical family is not claimed.

Two implementation issues were corrected without changing experiment thresholds:
the audit's fallback call lacked its required deadline argument, before collection
or training; later, a training-statistics mask accidentally used the final minibatch
instead of all pairs and interrupted epoch-8 reporting. The deterministic replay
matches every interrupted epoch-8 parameter bit-for-bit, then continues to epochs
16/24. No candidate outcomes were used to change learning settings. Prior protocols,
the interrupted checkpoint and the pre-fix runner are retained.

Reproduce with `python -m ml.run_threat_v31` and verify after completion with
`python -m ml.audit_threat_v31`, using the virtual environment. The full command
replays training/screens against frozen data, so is not a report-only operation.
Artifacts are in `ml/artifacts/threat-v31/`.

The next distinct experiment should address actual stand-pat/search-leaf training
and the gate together, with a fresh test set. An already quiet-trained network
should not simply be ungated and assumed stronger. This round's test is now exposed.
