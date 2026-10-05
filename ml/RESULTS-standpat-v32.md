# v32: actual static calls, static teacher targets and a non-check gate

## Outcome

Implemented and trained a new stand-pat hybrid. It improves static rankings and
average searched decisions at equal depth on these fresh positions, but does
not pass the equal-time strength gate. No game pilot ran and no model was adopted.
The app remains on the unchanged strategic-v26 heuristic. There is no new game
win rate or Elo estimate.

| Screen | Comparable positions | Heuristic move loss | Epoch-8 stand-pat hybrid |
| --- | ---: | ---: | ---: |
| Validation, depth 3 | 23 | 34.61 cp | 18.91 cp |
| Validation, 250 ms | 23 | 46.43 cp | 44.61 cp |
| Exploratory independent test, depth 3 | 24 | 31.29 cp | 23.58 cp |
| Exploratory independent test, 250 ms | 24 | 36.04 cp | 40.83 cp |

Lower is better. Screens each contain 24 roots. Means use the common nonmate
subset across all models; validation excludes one root, while error/mate counts
include every root. In timed validation, the heuristic makes two errors of at
least 150 cp and epoch 8 makes three. Both have two losing transitions and no
allowed mates. In the timed test, counts are one versus two large errors, one
losing transition each and no allowed mates. Thus the promising early checkpoint
does not meet the frozen acceptance criteria despite better fixed-depth means.

At 250 ms the hybrid completes a shallower iteration in 18/24 test positions,
the same depth in six and never a deeper iteration. Mean depth is 2.71 versus
3.54 for the heuristic. Broader activation adds inference cost. This supports
investigating inference efficiency, but does not isolate speed from all changes
to move ordering, pruning and evaluation. Small screens do not prove a general
playing-strength advantage even when average move loss improves.

## Gate and target redesign

The new gate permits correction at non-check static evaluations even when legal
captures exist. Search still handles terminal rules, checks and tactical
continuations. Public adapter calls also preserve terminal behavior. This does
not enable an old quiet-trained model outside its training domain: a separate
adapter requires matching non-check/static-target checkpoint metadata and rejects
quiet checkpoints. The old adapters and app configuration are unchanged.

Labels now come from Stockfish 19's `eval` debug command, specifically its
`Final evaluation ... (white side)` line. Local `evaluate.cpp::trace` confirms
that this is the normalized current-position static evaluation, with zero
optimism, rather than a searched score. Two printed pawn decimals correspond to
integer centipawns. The parser ignores raw/internal NNUE lines and rejects missing
or in-check values. Static networks can encode learned tactical expectations;
these labels are not guaranteed capture-free values. Crucially, the label request
does not itself search a capture continuation.

Deeper Stockfish analysis is kept separate: it selects usable source roots and
judges searched move choices. Those search scores never become evaluator score
targets. The candidate does not call Stockfish during inference.

Architecture remains 1,420 inputs and 64/32 hidden units, with the same raw
250 cp bound, 25% blend and whole-score integer rounding (up to approximately
62.5 cp correction). Hidden features warm-start from v31 epoch 16, but the final
layer is reset to zero so wider activation initially matches the heuristic.

## Actual-search dataset

Generated 960 fresh weighted GM book starts, seed 320032. Excluding known
v29/v30/v31 source families leaves 122 starts across 76 reserved six-ply canonical
families, allocated 2:1:1 to training/validation/test. Sources mix Stockfish
self-play at 3,000 nodes, heuristic self-play at 1,500 nodes and heuristic versus
Stockfish, with a 120-additional-ply cap. These are source trajectories, not
strength-test games.

Source snapshots at plies 16/48/80/112 with alternating color offsets receive
64,000-node teacher reviews; nonmate roots with absolute White score at most
800 cp are searched by our unchanged heuristic. The collector samples actual
static calls from iterative depth up to three with a 2,000-node limit. This
includes stand-pat calls before captures settle, cache hits and calls from a
partially completed iteration, rather than only completed quiet leaves.

Reservoirs retain up to 16 unique quiet and 16 unique capture positions per root.
Sampling does not filter by candidate errors or labels. Canonical aliases are
unique across folds and known previous labels/screens are excluded. This is
stratified coverage, not the natural frequency of evaluation calls.

| Split | Actual static labels | Quiet / capture | Contributing source families | Searched source roots | Static pairs, repair / retention |
| --- | ---: | --- | ---: | ---: | --- |
| Train | 4,317 | 2,157 / 2,160 | 35 | 137 | 480, 235 / 245 |
| Validation | 2,196 | 1,092 / 1,104 | 18 | 69 | 238, 120 / 118 |
| Test | 2,487 | 1,239 / 1,248 | 18 | 78 | 252, 120 / 132 |

There are **9,000 actual-call labels** from 123 source trajectories, plus 1,000
ordinary retained positions newly relabeled with the same static teacher. Only
4,317 fresh training labels plus those 1,000 retained labels enter fitting.
Whole current source games and families own one fold. Positions within a game
remain correlated. Legacy training-family provenance is incomplete, so absolute
independence from every historical family is not claimed.

Static pairs compare actual calls from the same root and search ply, reached
through different first root moves. Require at least 15 cp teacher static
separation, feasible independently bounded endpoint corrections and nontrivial
heuristic margins. Keep at most two repair/two retention pairs per root and never
reuse an endpoint. These are static-value rankings, not proof of the better
searched root move; the independent search screens judge that separately.

Across collection searches the old gate would activate on 57,418 of 356,878
static calls (**16.09%**); 299,460 calls have legal captures. The new gate can act
on every such non-check static call (**100%**). These numbers are not directly
comparable to v31's 8.63% because source positions/search limits differ.

## One fitted trajectory and held-out learning

Train once for 24 epochs, retaining only predeclared epochs 8/16/24. Use AdamW
at .0005, bounded deployed-residual SmoothL1 supervision, rounded hybrid score
proxy, repair ranking importance three, retention protection ten with a 20 cp
margin cap and .02 correction regularization. Gate, bound and rounding match
inference. No old searched-score targets are mixed into the new static target.

| Static ranking split | Heuristic correct | Epoch-8 correct | Repairs | Retention regressions |
| --- | ---: | ---: | --- | --- |
| Train, 480 pairs | 245 | 340 | 95 / 235 | 0 / 245 |
| Validation, 238 pairs | 118 | 135 | 28 / 120 | 11 / 118 |
| Test, 252 pairs | 132 | 145 | 22 / 120 | 9 / 132 |

Epoch 24 reaches 386/480 training rankings, with 141 repairs and no training
retention regressions, but is worse on searched validation. More training accuracy
does not determine selection. Epoch 8 reduces held-out static-score MAE from
213.44 to 208.79 cp on quiet test positions and 217.28 to 204.96 cp on capture
positions. Static metrics are descriptive, computed after selection/test, and
did not select any checkpoint. They support useful generalization in this target
domain but do not establish game strength.

## Fresh screens and recorded exploratory amendment

Freeze 24 roots from 24 distinct games in each held-out fold, max two/game-family,
no duplicate canonical roots and at least 12 families, before fitting. Both
actual screens span 18 families and 12 White/12 Black roots; validation has
14 middlegames/10 endgames, test 12/12. Source selection uses teacher scores and
coverage only. Models search depth 3 or 250 ms/depth 64 with unchanged LMR,
rotating order, full histories and exclusive CPU access. Distinct choices get
256,000-node best/forced Stockfish review.

The original gate requires at least 20 comparable roots, nonworse equal-depth
mean, strictly better timed mean and no additional large errors, losing
transitions or allowed mates. **No snapshot passed original validation.** Its
selection/decision remain unchanged in the artifacts.

Epoch 8 improved both averages but its extra large-error case was already lost:
Black's teacher best continuation is -638 cp. The heuristic loses another 97 cp
with `Kd7`; epoch 8 loses 376 cp with `Ke7`. It is still an error and is retained
in all reported counts. Because this candidate uniquely improved both means
without extra losing transitions/mates, a recorded amendment after validation
but before any test result allowed one exploratory independent test. The
checkpoint, roots, code hashes and exception rationale were frozen before it.
No model was retrained or selected on the test. The original strict criteria,
including all large errors, remained in force for that test and failed. No
further amendment, weight tuning or game pilot followed. The test is now exposed.

## Verification and implementation notes

All **146 tests** pass. Audits replay all source and sampled histories, check
static label parsing and repeat teacher queries, validate pair provenance and
feasibility, and find zero fresh cross-split aliases/families, zero known prior
alias/source-family overlap, and zero ordinary-retention/held-out overlap.
Six complete instrumented trees match uninstrumented scores, moves, nodes,
quiescence nodes, cache hits and repetition counters. Checkpoint/source/binary
hashes match their frozen protocols.

Torch and optimized inference agree exactly on 120 audited actual positions for
epoch 8, epoch 16 and the initial zero model; epoch 24 differs by at most one cp
at a floating-point rounding boundary. Every model preserves color symmetry and
boards. The initial adapter matches the heuristic on all 120 positions.

Before fitting, an exact shared-attack-mask encoder replaced repeated pin/attacker
computations. It preserves feature arrays byte-for-byte in random/special-position
tests and on 512 actual sampled positions. Alternating-order component timing
reduces median feature encoding from 130.72 to 98.20 microseconds. This excludes
baseline evaluation, other features, the network and search; it is not a full
latency or playing-strength claim.

Three prefit corrections/amendments are preserved: the collector accidentally
overwrote the inherited repetition counter with a reservoir counter, so four
preliminary sources were discarded and collection restarted after a regression
test; inference feature reuse was added before fitting, preserving completed
collection; initial test coverage permitted only 23 distinct games under the
family cap, so one teacher-play trajectory from a reserved missing test family
added four roots/128 labels. The 24-game/max-two-family restrictions were kept.
No model outcomes informed these collection or encoder changes.

Artifacts are in `ml/artifacts/standpat-v32/`. Use the virtual environment with
`python -m ml.audit_standpat_v32` and `python -m ml.static_metrics_v32` for checks.
`python -m ml.run_standpat_v32` replays training and screens, not just reports;
preserve existing outcomes before rerunning. The separate exploratory script
refuses to repeat an existing test. Do not treat reused test roots as fresh data.

The next useful experiment is to preserve this early model's predictions while
reducing full inference cost, then measure on new equal-time positions. Native
feature/inference execution or a carefully verified cheaper representation is
more motivated here than another epoch grid or adding layers. Adoption still
requires a fresh equal-time pass and a subsequent game pilot.
