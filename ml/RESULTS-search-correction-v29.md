# v29: new-engine leaves, consequential regressions and searched selection

The app's strategic heuristic, search, inference implementation and NN architecture are unchanged. This experiment changes data and checkpoint selection. No neural evaluator was promoted.

## Diagnosis and correction capacity

Reviewed all19 v28 searched development roots, retaining their256k-node teacher move reviews. There were six consequential NN disagreements (at least50cp worse than the heuristic or an added losing/mate transition). Full-window depth-three traces provide additional leaf/horizon context, but use a different diagnostic tree from LMR search and do not establish static-evaluation causality.

Stockfish settled better and worse alternatives through tactical continuations (up to12steps at8000nodes/step), then labeled quiet endpoints at64,000nodes. Three positive-ranking rows resulted, representing **two distinct endpoint pairs**. Both10% and25% blends can theoretically correct those endpoint rankings within their respective25cp/62.5cp per-endpoint limits. Endpoint capacity permits a ranking correction; it does not prove a root search repair. Tactical positions with captures/checks get zero correction.

The heuristic already ranks all three rows correctly. The old25% NN ranks only one correctly; the selected retrained NN restores all three. These are training examples, not evidence of unseen improvement.

## Broader actual-search distribution

Frozen80 fresh book starts before the experiment. Generate40 fixed-node source trajectories with the new heuristic (24train,8validation,8test), up to80 self-play plies after the book prefix. These are truncated source trajectories, not completed strength-test games. Search1500nodes/source move; sample roots at four prescribed stages, collecting quiet leaf reservoirs with depth3/node4000searches.

Canonical position after the first six plies owns one fresh split, including transpositions/color aliases. Diagnosed families and the reserved v28 searched-test families are excluded from fresh validation/test generation. An initial move-sequence grouping was found to admit transposition-equivalent families; its preliminary collection was stopped before fitting or selection and preserved separately. Canonical grouping was applied before the final collection. A subsequent implementation error occurred before collection/training and was repaired; it yielded no candidate results.

Final root counts:83training,16validation,16test, from16/6/6fresh opening families with zero cross-split overlap. Deeply label consequential disagreements at256k nodes; ordinary quiet leaf labels use64k. The collection yielded310new leaf/disagreement label rows and no additional confirmed positive ranking pairs beyond the diagnostic cases. Combined with diagnostic endpoints, there are313unique new score positions.

Training mixes2000ordinary retained rows with the new labels sampled fourfold (3264score samples total),1500retention ranking pairs and the three feasible focused ranking rows sampled eightfold (1524ranking samples total). Retention protects decisions the new heuristic gets right. Current held-out root aliases are removed from score/ranking training. Historical opening-family provenance is incomplete, so global family independence is not claimed. Three focused rows/two distinct pairs remain a small limitation; this is not a large new consequential-ranking dataset.

## Training and selection

Warm-start the rebased v28 checkpoints for their matching10%/25% blends, keep892→64→32architecture, color consistency, raw250cp bound, quiet gate and integer rounding. Use teacher score proxy plus settled ranking supervision and correct-ranking protection. The proxy is not calibrated WDL. Save six predetermined candidates: two weights at epochs8/16/24. Select by actual searched depth3/250ms choices on fresh validation, not training accuracy.

Stockfish reviews distinct chosen moves at256k nodes, with full histories. Timed probes precede teacher queries; model order rotates and CPU workloads are serialized. Means use the same16nonmate-score positions for every model. The25% epoch16candidate passed validation and was frozen before the untouched test; no retuning followed test results.

| Mean move regret | Heuristic | Selected25% NN |
|---|---:|---:|
| Validation depth3 |74.19cp|48.75cp|
| Validation250ms |65.38cp|62.44cp|
| Untouched test depth3 |69.50cp|81.69cp|
| Untouched test250ms |99.44cp|134.94cp|

On test, >=150cp errors increased1→2 atdepth3 and3→4 at250ms. Losing transitions increased0→1 at250ms; no model newly allowed a mate. All10% candidates failed validation. The selected25% candidate failed test, so no game pilot ran, no strength/Elo claim is made, and the app remains on the heuristic alone.

## Verification and implications

All133tests pass, including transposition-family grouping and correction-capacity/gate tests. Audit verifies40legal source trajectories, zero fresh family overlaps, zero training aliases of held-out roots, frozen source hashes, and exact optimized/Torch rounded inference agreement plus color symmetry for all six checkpoints on100positions each.

The confirmed training regressions were repaired and searched validation improved, but the gain did not generalize to untouched decisions. Failure at equal depth means inference speed alone cannot explain it. This experiment does not prove that architecture or correction limits are the cause. More independent consequential rankings remain a practical next data need; two distinct focused pairs are insufficient evidence of broad repair learning.
