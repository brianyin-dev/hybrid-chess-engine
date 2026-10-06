# v35: search-derived supervision and relationship-input ablation

**Implemented all four steps and trained both models. There are useful unseen corrections, particularly at equal depth on Black positions, but no consistent equal-time advantage or demonstrated game-strength improvement. Neither model is adopted.** Production remains the unchanged strategic-v26 heuristic. No game pilot or new win rate.

## Data and supervision

Added 100 standard-book source games in 100 new six-ply canonical opening families: 60 training, 20 validation, 20 test. Reused 162 existing v33 game histories to save compute; no historical NN weights are reused. Sources mix Stockfish self-play and heuristic-versus-Stockfish, with book moves only as prefixes. All games and opening families remain confined to one split.

Collected new roots throughout middlegames/endgames and traced the heuristic's actual depth-2/3 decisions through full-window branches and quiescence. Repair pairs require an actual heuristic choice at least 25 cp worse than a Stockfish-confirmed alternative. Retention pairs require the actual heuristic decision to remain within 25 cp of Stockfish's best move under deeper analysis. Root alternatives are confirmed at 128,000 nodes; new leaf endpoints receive 32,000-node analysis, must retain the preferred ordering by at least 15 cp, and must have a best reply that is neither a capture, promotion nor check. These finite searches are quality filters, not proof that all tactics are exhausted.

Static Stockfish labels support score fitting, but static endpoint preference is not required to agree with the confirmed searched decision. This avoids forcing the ranking objective to follow a conflicting static preference. Impossible bounded repairs and duplicate/canonical-mirror endpoints are filtered. One pair is retained per sampled root, separated by 24 plies along each new sampling trajectory. Reused prior pairs can share nearby positions in those same trajectories; the pairs are not all independent observations. Independence is enforced at the game/family split level. Prior pairs are reused only after verifying that they meet the actual-decision criteria; their original family splits remain unchanged.

| Split | Confirmed pairs | New / reused | Repair / retention | Contributing games | Contributing families |
|---|---:|---:|---:|---:|---:|
| Training | 179 | 86 / 93 | 98 / 81 | 98 | 65 |
| Validation | 71 | 42 / 29 | 36 / 35 | 38 | 24 |
| Test | 73 | 41 / 32 | 37 / 36 | 34 | 25 |

Training also uses 2,232 ordinary positions sampled from actual static calls, for 2,590 score-support rows including pair endpoints. Whole source collection: 262 games, split 156/56/50, with 97/34/34 source families. Training root-ranking pairs include 137 White and 42 Black cases; its pair endpoints include 188 White-to-move and 170 Black-to-move positions. Root-color coverage remains uneven and should be balanced in subsequent collections.

## Controlled models and training

Both models have 64/32 hidden units and a 25% correction with a raw +/-250 cp bound (about +/-62.5 cp final), integer rounding and the existing non-check gate. The current-input arm has 1,420 inputs and 93,057 parameters. The relationship arm adds 396 inputs and has 118,401 parameters; equal hidden widths do not mean equal parameter counts.

Added shared relationship summaries:

- Piece-type distance and direction histograms relative to each side's king, in the piece owner's orientation.
- Pawn distance to promotion, passed-pawn and blocked-forward-square histograms.
- Attacked-piece value, undefended value, cheaper-attacker value margin, least-defender value and pinned attacked-piece value, grouped by piece type.

These are descriptive, pin-aware geometric features, not complete capture-legality/SEE calculations or guaranteed tactical detectors. Color-mirror and horizontal-translation tests check that the same geometry is represented consistently.

Both models train every layer from scratch, with identical initial weights for shared inputs/layers, a zero output head and zero weights for the added inputs. They use identical data and batch-order seeds, 48 epochs, AdamW at 0.0005, weight decay 0.001, feasible softplus ranking targets, repair weighting and retention protection. Static supporting residuals are clipped to +/-45 cp with a small loss weight. There is no global penalty rewarding a correction of zero.

Snapshots at epochs 8/16/24/32/40/48 are selected by searched depth-3 development decisions, rather than training or static-ranking accuracy. Every hidden layer changes; final training snapshots learn all 179 rankings. That success does not carry over fully to held-out rankings. Selected checkpoints: current-input epoch 40 and relationship-input epoch 16. Their held-out static ranking counts are 39/73 and 35/73 respectively, versus the heuristic's 36/73. Test predictions never select epochs or training parameters.

## Original searched comparison

All models use unchanged search, LMR and fixed depth 3 first; timing tests then use 250 ms per move under the CPU lock without concurrent test activity. Stockfish move reviews use 128,000 nodes. All means below have 24 nonmate comparable positions. Lower move loss is better; these are not game scores or Elo.

| Evaluator | Development depth 3 | Original test depth 3 | Development 250 ms |
|---|---:|---:|---:|
| Heuristic | 35.58 cp | 19.71 cp | 32.25 cp |
| Old v32 NN | Not used for epoch selection | 28.58 cp | 37.75 cp |
| Current-input v35 | 25.71 cp | 21.54 cp | 30.00 cp |
| Relationship-input v35 | 27.67 cp | 25.33 cp | 32.92 cp |

The current-input model removes the heuristic's one >=150 cp error in the original test: a 251 cp move loss becomes zero. However, it improves two positions and worsens five, leaving average loss 1.83 cp higher. The relationship model also falls short of the heuristic's test average. Therefore both fail the frozen original acceptance criteria. Original White test timing and game pilots are not run; a better timed development average alone does not override failed held-out depth quality.

## Supplemental Black coverage

Final audit found that the newly sampled development/original searched-test roots were all White to move, despite training containing both colors. This limits the original comparison. A separately recorded, post-selection diagnostic adds 24 actual Black-to-move roots from reserved test games, with the same teacher-only prefilter and unchanged checkpoints. No fitting, model selection, gate adjustment or adoption decision follows this supplement.

| Evaluator | Black depth-3 mean move loss | Major errors at depth 3 | Black 250 ms mean move loss | Major errors at 250 ms |
|---|---:|---:|---:|---:|
| Heuristic | 38.13 cp | 3 | 49.08 cp | 3 |
| Old v32 NN | 45.96 cp | 3 | 34.71 cp | 3 |
| Current-input v35 | 20.71 cp | 1 | 51.04 cp | 2 |
| Relationship-input v35 | 22.38 cp | 1 | 47.13 cp | 2 |

Black timed mean completed depths: heuristic 3.875, old NN 3.083, current-input 2.958, relationship-input 2.875. Current-input depth-3 corrections help substantially here, but their benefit is not consistent at equal time. The extra relationship inputs do not beat the current representation at equal depth in either test cohort; small timed differences do not establish strength.

Each cohort has 24 distinct source games and 24 families. Together they contain 48 positions from 31 source games and 27 families, so the cohorts are related and must not be treated as 48 independent games. The supplement is additional coverage, not a newly declared balanced primary benchmark or permission to change failed gates. None of its families overlaps training.

## Verification, decision and reproduction

All 160 unit tests pass. New tests cover mirror/translation geometry, promotion distance, attacker/defender values, batched canonicalization, inference/training parity, correction bounds, zero behavior, terminal/check handling, checkpoint metadata and trace history/score perspective. Inference audit checks every snapshot and both zero controls. Data audit verifies full histories, static baselines, actual-decision confirmation, legal/quiet endpoint replies, and zero cross-split family or canonical-mirror position overlap. Final audit verifies frozen source/data/root hashes, shared initial weights, zero added columns, every hidden layer updating, and no test use for checkpoint selection.

The controlled experiment demonstrates changes worth investigating, but does not establish a repeatable NN advantage over the heuristic. The current-input candidate is more promising than the added-feature candidate in these fixed-depth comparisons. Original adoption criteria stay unchanged; production remains heuristic-only. No games ran and no rating estimate is updated. Changed data/objective/initialization together distinguish v35 from v32, so any improvement over v32 cannot be attributed solely to more data.

Artifacts: `ml/artifacts/relationship-v35/`, including original collector data, reuse/retention-validation records, split/provenance audits, training checkpoints/history, frozen selection, original searched screens, and the separate Black coverage protocol/results.

Reproduction, in order: `python -m ml.collect_relationship_v35`, `python -m ml.merge_relationship_v35`, `python -m ml.validate_relationship_v35`, unit tests without a benchmark running, `python -m ml.train_relationship_v35`, `python -m ml.check_black_relationship_v35`, `python -m ml.audit_relationship_v35`. Recorded outputs are protected against overwrite. Collection can resume the identical frozen collector after interruption. External book and Stockfish provenance are documented in their existing repository instructions.

## Later user-requested exploratory match

After the original experiment, the user explicitly requested ten games. The frozen current-input epoch 40 model scored 4 wins, 3 draws, 3 losses (55% score) against strategic-v26 at 250 ms on five fresh color-swapped starts. This does not retroactively change the original gates or establish a repeatable advantage. See `ml/RESULTS-relationship-v35-user-pilot.md` for the audited match. Production remains heuristic-only.

## User-requested extension to 20 games

The same frozen current-input epoch40 candidate finished 5W/4D/11L against strategic-v26 at 250 ms: **35% score**. The additional ten games were 1W/1D/8L. The original ten are preserved, and the extension was requested after their results were seen. All games and paired starts passed the audit. This does not establish an NN advantage; production remains heuristic-only. See RESULTS-relationship-v35-user-pilot-20.md.
