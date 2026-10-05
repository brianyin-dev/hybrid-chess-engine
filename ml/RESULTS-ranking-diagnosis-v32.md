# Exposed v32 ranking failure diagnosis

Post-test diagnosis only. No fitting, parameter changes, selection, or games. The app remains the strategic-v26 heuristic. These are 107 failed static leaf rankings, from 26 source games, not 107 actual played blunders or independent games.

Against the stored static labels: 51 correction differences worsen the ranking; 42 improve it without reversing it; 14 leave it unchanged. There are 98 unrepaired examples and nine retention regressions, and two final ties. All 107 are theoretically reversible within the existing endpoint bounds; that does not establish that one shared model can repair them all simultaneously.

In 38 failed pairs at least one rounded endpoint correction has magnitude at least 60 cp. In 12 of the 14 unchanged pairs both endpoints receive the identical near-bound correction. Thus saturation can remove discrimination. Simply widening the bound can also amplify wrong-direction corrections.

Each endpoint was reconstructed with full source history, then independently analyzed by single-thread Stockfish 19 with 64,000 and 256,000 nodes, fresh game/hash state. At 256,000 nodes, the searched order agrees with the original static order in 54 pairs, reverses in 48, and ties in five. Mate scores are ordered ahead of finite scores. Search budgets are finite and the counts changed from the 64,000-node pass: this is evidence that static-label accuracy is not synonymous with searched decision quality, not proof that static labels are erroneous. Sampling paired branches at one root and ply does not settle tactical continuations or define final root decisions.

Concrete examples whose preferred order survives both analysis budgets:

- Promotion/endgame: `8/8/1R6/4k3/6K1/8/4p3/8 b - - 0 67` contains immediate `e2e1q`. Stockfish searched White score is -265 cp versus 0 in the paired drawing position. The hybrid prefers the Black-winning position for White; its correction worsens the margin from -63 to -181 cp.
- Pinned bishop: `r3rk2/ppp2ppp/8/3p1b2/1P3P2/4B1P1/PPP1K1P1/R2B4 b - - 0 17`. White bishop e3 is pinned to king e2 by rook e8. Stockfish starts `d5d4` and evaluates White at -808 cp, versus -259 in the paired position. Material is identical. The heuristic barely prefers the safer position (+2 cp margin), while the hybrid reverses it (-26), a retention regression.
- Forced mate: `8/pp1k3p/6p1/n7/P4Q2/2P3Pb/5P1P/4rBK1 b - - 11 33` has `e1f1` mate in one. The paired alternative favors White at +478 cp after search. The hybrid correction improves the static ranking by 53 cp but leaves a -37 cp wrong margin.
- Queen-endgame mating geometry: `8/4Q1p1/8/5p1p/7P/5qP1/7K/5k2 w - - 26 65` is mate in four against White; the paired alternative is drawn. The hybrid changes the ranking by only 3 cp; its endpoint corrections are +59 and +62 cp.

These demonstrate evaluator ordering errors, not that the complete engine necessarily plays each mating or promotion mistake: legal move search, terminal handling, and quiescence may resolve them. No defensible exhaustive hanging-piece/positional taxonomy was inferred from geometric attacks alone.

Artifacts: `ml/artifacts/standpat-v32/ranking-failure-diagnosis.json`. Reproduce with `python -m ml.diagnose_rankings_v32`, followed by `python -m ml.deepen_rankings_v32`, using the project virtual environment. Neither script trains or modifies the engine. Future selection must use fresh unexposed positions.
