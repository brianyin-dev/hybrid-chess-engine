# v28: new app heuristic and rebased neural corrections

The app now uses the selected v26 strategic heuristic (weights 1,1,0.25,1,1,1). The original `engine.evaluation` is preserved for benchmarks and old NN checkpoints. API position scores use the same strategic baseline; terminal mate/draw semantics are preserved, and each search has its own mutable evaluator cache. The user authorized adoption based on the modest prior game evidence; this change does not claim a proved Elo gain or a completed 750ms game confirmation.

The interrupted v27 experiment is retained separately: 132 completed games, 66 wins, 16 draws and 50 losses, 56.1% score at250ms. The user stopped it; one interrupted game is excluded, and its planned app-budget games did not run.

## Neural training

Reused existing historical labels/rankings from balanced-v19, plus quiet search labels from v26. Recomputed targets as label minus the new strategic score without rerunning Stockfish labeling. Historical labels retain their original analysis budgets/semantics; reuse is not a claim of newly stronger teacher analysis.

After quiet/terminal filtering, training uses 9,919 score rows and 4,186 ranking pairs; validation uses 68 rows and 45 pairs. Existing whole-game splits are preserved with cross-split canonical aliases/game IDs removed. Earlier opening-family provenance is incomplete, so globally independent opening families are not claimed.

Trained separate 10% and 25% candidates from zero final-layer output, rather than transferring old corrections unchanged. Both retain the compact 892→64→32 architecture, color consistency, and a raw250cp correction bound (maximum deployed changes25cp/62.5cp). The loss uses the actual quiet gate, full hybrid integer rounding through a straight-through gradient, teacher score proxy, move-ranking supervision and protection for rankings the new heuristic already gets right. This proxy is not a calibrated win probability. Epochs are selected on existing validation rankings/score proxy; no test-driven epoch selection.

A new adapter checks the checkpoint baseline identity/weights and trained blend. Optimized inference is compared with Torch predictions, and terminal, gate, symmetry and wrong-baseline/weight rejection are tested. Original NN inference paths remain unchanged.

## Searched development decisions

Fresh book starts were frozen before training. Whole first-four-ply families are separate between this round's searched development/test sets; roots exclude reused training/ranking aliases. Inexpensive2000-node Stockfish continuations generate positions without filtering on candidate outcomes. Historical family independence is not claimed. Development has19 roots; the unsearched test has20 roots. Both include full histories.

Each model is compared at depth3 and250ms (depth ceiling64), unchanged LMR, rotated model order, exclusive CPU. Distinct chosen moves are reviewed at256,000 Stockfish nodes. Mean errors use common nonmate-score positions (all19 here).

| Mode | New heuristic | 10% NN | 25% NN |
|---|---:|---:|---:|
| Depth3 mean regret |81.53cp|87.42cp|114.84cp|
| 250ms mean regret |116.63cp|122.53cp|160.26cp|
| Depth3 errors >=150cp |5|5|6|
| 250ms errors >=150cp |7|7|8|

Losing transitions were2 each at depth3 and3 each at250ms; no model newly allowed a mate. Neither candidate passed development: both increased mean regret at equal depth and equal time. This indicates evaluation/decision weaknesses beyond inference cost alone, though a fixed-depth search comparison does not isolate static evaluation causality.

No blend was selected, the untouched searched test was not run, and no game pilot was run. The app uses the new heuristic alone. If a future candidate passes development and untouched test, the implemented protocol runs20 fresh paired games against the new heuristic and expands to40 only for a >=60% pilot score. No NN strength claim follows from these rejected candidates.
