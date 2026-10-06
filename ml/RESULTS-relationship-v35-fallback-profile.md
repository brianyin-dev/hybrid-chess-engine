# Why games 11 and 15 returned depth-zero blunders

The failure reproduces in all three normal 250 ms NN runs for each position, with the same recorded fallback moves. The heuristic completes at least depth one in every comparison. No engine/model changes were made.

| Measured traced 250 ms search | Game 11 | Game 15 |
|---|---:|---:|
| Emergency fallback pass | 2.1 ms | 2.3 ms |
| Static evaluation including cache lookups | 218.1 ms | 196.5 ms |
| Fraction of total budget spent in static evaluation | ~87% | ~79% |
| Deepest visited ply | 16 | 15 |
| Best completed root candidate before interruption | Bxf3 | Qf6 |
| Returned fallback | Nf5 | Nxc3 |

Depth one still includes extensive quiescence. Completed traced depth-one searches visited 983 nodes (982 quiescence) in game 11 and 1,766 nodes (1,765 quiescence) in game 15. Capture sequences and mandatory check evasions extend to 16 and 15 plies. This is a finite expensive search, not an infinite loop, and no ply-limit termination occurred.

The expensive root branches include Bxf3 (~86 ms) and Nf5 (~58 ms) in game 11, and Qf6 (~80 ms, 856 quiescence visits) in game 15. These figures include evaluation and instrumentation. The emergency fallback considers all 41/40 legal moves and costs around 2 ms; it is not what burns the budget.

## A timeout discards information the search already found

search() initializes its returned move with an emergency pass, then only replaces it after an entire iteration completes. The root worker updates root_candidate as individual root moves finish, but when depth one times out the outer function returns the old emergency fallback. This is conservative last-completed-iteration behavior with an unsafe consequence when no iteration completed.

In game 11 the traced partial candidate Bxf3 is Stockfish's best move (+125 cp). The returned Nf5 instead evaluates to -862 cp, a 987 cp loss at512k nodes.

In game 15 the traced partial candidate Qf6 avoids mate (approximately -16 cp). The returned Nxc3 allows mate in two. Separate full-window forced depth-one/quiescence probes assign Nxc3 a mate-loss score (-99996), while Qf6 scores -185 cp in our NN evaluator. Thus quiescence can already detect this mate; it is not inherently beyond the nominal depth-one search horizon. The problematic Nxc3 root branch completed before the timeout in the traced search.

This does not prove that always trusting a partial iteration is a safe general fix: unsearched moves, score bounds and changed root ordering matter. Partial-root handling should be explicitly designed and tested, rather than changing the meaning of completed depth.

## Feature extraction dominates

Separate cProfile depth-one runs attribute about 44–45% of evaluator cumulative time to threat feature extraction, about 14–16% to incremental encoding, and about 35–36% to the handcrafted baseline. Remaining evaluator overhead, including dense network calculation, is much smaller. These are profiled inclusive timings, not additive independent wall-clock benchmarks. Matrix multiplication is not the principal bottleneck.

A completed traced depth-one NN run takes approximately272 ms in game11 and268 ms in game15, versus128/107 ms for the heuristic. Additional uninstrumented NN runs range267–315 ms and256–400 ms, showing host variability. All untimed runs complete depth one. Model initialization is outside the measured search time. Profiler overhead inflates cProfile timings, so those are used only for attribution.

More time is not a complete cure: completing depth one selects g6 and avoids mate in game15, but selects Bd7 in game11, still a502 cp Stockfish regret. At depth three, prior diagnosis found Bxf3 in game11. Both timeout handling and evaluation/search quality remain relevant.

## Next concrete change to test

Handle interruption before the first completed iteration using trustworthy completed root work, with clear score-bound/completion tracking and preserved legal fallback behavior if no root branch finishes. Add regression tests for interrupted search, board restoration and tactical safety on these failures. Separately reduce duplicated board-feature computation while maintaining exact evaluator parity. Keep the checkpoint and playing budget fixed; verify broader fresh positions and games before claiming strength improvement.

Artifacts: ml/artifacts/relationship-v35/user-pilot-20-250ms/fallback-profile/report.json, partial-candidate-check.json, forced-depth1-values.json, and the two cProfile text reports.
Runner: python -m ml.profile_relationship_v35_fallbacks. Existing output is protected against overwrite.
