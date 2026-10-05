# Broad move-ranking experiment (v30)

## Outcome

The unchanged network learned from a substantially broader ranking dataset and
passed searched validation, but failed the independent equal-time test. No game
pilot ran and no model was promoted. The app continues to use the strategic-v26
heuristic alone. This is not a demonstrated playing-strength improvement.

| Searched screen | Positions with nonmate scores | Heuristic mean move loss | Epoch-24 hybrid mean move loss |
| --- | ---: | ---: | ---: |
| Validation, depth 3 | 24 | 55.00 cp | 41.83 cp |
| Validation, 250 ms | 24 | 87.08 cp | 59.38 cp |
| Test, depth 3 | 23 | 114.70 cp | 104.83 cp |
| Test, 250 ms | 23 | 77.57 cp | 81.52 cp |

Lower move loss is better. Each screen contains 24 roots. Test means exclude
one root with a mate-valued review; error counts cover all roots. Both test
evaluators had four errors of at least 150 cp, four losing transitions and zero
newly allowed mates in each search mode. The candidate failed because its
equal-time mean was worse, despite improving at equal depth. These small screens
are selection evidence, not an Elo estimate or a game win rate.

## Collection and supervision

Generated 179 legal source trajectories from fresh weighted book starts:
60 Stockfish self-play at 3,000 nodes per move, 60 new-heuristic self-play at
1,500 nodes, and 59 heuristic-versus-Stockfish trajectories (2,000 opponent
nodes). Trajectories are capped at 160 additional plies and can be truncated;
they are not completed strength-test games.

| Split | Distinct pairs | Repair / retention | Contributing games | Generated opening families | Middlegame / endgame | White / Black roots |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Train | 300 | 114 / 186 | 103 | 61 | 87 / 213 | 150 / 150 |
| Validation | 60 | 24 / 36 | 20 | 10 | 15 / 45 | 35 / 25 |
| Test | 60 | 19 / 41 | 22 | 12 | 10 / 50 | 28 / 32 |

There are 420 unique ordered endpoint pairs and 840 unreused endpoints, drawn
from 145 contributing games. Repair pairs have a nonpositive heuristic endpoint
margin; retention pairs preserve a positive one. These are endpoint rankings,
not necessarily mistakes the heuristic makes after search.

Canonical color/transposition-equivalent first-six-ply opening families and
whole games belong to one split. Pair spacing is at least eight plies, with one
accepted pair per 24-ply window and at most five per game. Pairs within a game
remain correlated; 420 distinct pairs does not mean 420 statistically independent
observations. The dataset is endgame-heavy.

Proposals use Stockfish MultiPV and heuristic/previous-NN disagreements throughout
games. Preliminary root reviews use 16,000 nodes; confirmation uses 128,000 nodes,
or 256,000 for consequential student disagreements. Both alternatives are walked
through teacher continuations until strict quiet endpoints, with up to 12 steps
at 8,000 nodes and final endpoint labels at 64,000 nodes. Root and endpoint gaps
must each be at least 15 cp, with nonmate bounded scores. Teacher-settled endpoints
are not guaranteed to be the engine's actual quiescence leaves.

The actual 25% quiet correction, integer rounding and raw 250 cp bound are used
to check whether independent endpoint corrections could reverse a ranking.
The maximum endpoint correction is 62.5 cp. This capacity check is necessary
supervision filtering, not proof that the network or root search can repair it.

## One frozen training round

Warm-started the previous v29 epoch-16 model. Kept the 892-input, 64/32-hidden
architecture, quiet gate and 25% bounded blend unchanged. Trained once for
24 epochs; only predeclared epochs 8, 16 and 24 were candidates. Supervision
includes 300 new pairs, repair loss weighted threefold, retention protection,
and 2,000 ordinary retained score labels. Sampling the 600 fresh training endpoint
labels threefold gives 3,800 score samples per epoch, not 3,800 unique positions.

Before fitting, froze 24 searched validation roots from 23 games and 24 test
roots from 24 games. Both have 12 White and 12 Black roots; validation has
15 middlegames / 9 endgames, test 14 / 10. Root pools come from fixed trajectory
snapshots, filtered only by teacher nonmate scores within 800 cp and coverage,
not by candidate results. Searched choices are reviewed at 256,000 teacher nodes.
Validation selects epoch 24; test is consulted once. There is no post-test tuning.

Two documented collection amendments occurred before any fitting: archived seven
preliminary sources/six pairs to fix even-ply color bias and broaden generation;
then expanded screen snapshot coverage and actual warm-start alias exclusions
after 290 training pairs, before collecting held-out data. Those 290 training
pairs were retained. Protocols and prior collector versions are preserved.

## What it learned and what still fails

| Endpoint ranking set | Heuristic correct | Previous NN correct | Epoch-24 correct | Epoch-24 repairs | Epoch-24 retention regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train, 300 | 186 | 188 | 230 | 47 / 114 | 3 / 186 |
| Validation, 60 | 36 | 39 | 41 | 13 / 24 | 8 / 36 |
| Test, 60 | 41 | Not evaluated | 39 | 8 / 19 | 10 / 41 |

Endpoint metrics are descriptive and did not select the checkpoint. Training
improved substantially, but unseen retention damage offsets repairs. The test
does not establish that inference speed is the cause of the equal-time regression;
it also shows a static generalization problem. More layers or another tiny
targeted retraining are not justified by this result. The next distinct experiment
should examine representation or quiet-gate/settled-leaf alignment, with a fresh
test set. These test roots must now be treated as exposed.

## Verification and artifacts

All 135 unit tests pass. The provenance audit validates legal histories, teacher
label consistency, feasibility, spacing, unique endpoints, frozen source hashes
and zero fresh cross-split position/family overlap. Held-out positions also have
zero canonical alias overlap with retained and known warm-start training data.
Legacy training opening-family provenance is incomplete, so historical family
independence is not claimed. Each checkpoint matches engine inference exactly
on 100 audited positions, preserves color symmetry and leaves boards unchanged.

The first full-suite attempt overlapped the exclusive audit lock, causing two
benchmark CLI timeout errors. After the audit released the lock, the complete
135-test suite passed; no timeout thresholds or implementation were changed.

Reproduce collection/training/screens with `python -m ml.run_ranking_v30` (reruns
training and screens against the frozen data), verify with
`python -m ml.audit_ranking_v30`, and generate descriptive endpoint metrics with
`python -m ml.ranking_metrics_v30`. Use the project's virtual environment.
Run timed experiments without competing CPU workloads. Collection checkpoints
resume source generation, but the full command is not a report-only operation.

Detailed data, protocols, checkpoints, audits and searched decisions are under
`ml/artifacts/ranking-v30/`; the initial discarded collection is under
`ml/artifacts/ranking-v30-pre-collection-amendment/`.
