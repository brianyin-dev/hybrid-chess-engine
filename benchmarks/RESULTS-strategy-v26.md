# v26: general strategic evaluation and representative validation

This round builds a general evaluation benchmark and six reusable positional relationships, rather than a rule for one previously observed move. The original heuristic and v24 LMR are the baseline. No NN is trained; later neural training can reuse the search-leaf data and benchmark.

## Representative positions

Generated 120 independent source games from fresh weighted book starts, using single-threaded Stockfish at 2,000 nodes per source move, up to 200 plies. This inexpensive source budget provides positions, not final labels. Categorized positions geometrically as tactical opportunities/checks, king pressure, ordinary middlegames, advanced passed-pawn races and other endgames. These are sampling strata, not expert-certified tactical puzzles. Both sides to move are included.

Frozen 120 training roots (24/category), 40 development roots (8/category) and 40 test roots (8/category). Whole canonical first-four-ply opening families own one split. Root subsets use 23 training, six development and six test families. Prior labeled/diagnostic root aliases are excluded. This is current-study independence, not a claim that no earlier project ever analyzed those opening families.

Coverage was repaired before labels or fitting: the initial hash partition supplied only five test games and insufficient king-pressure coverage, and the initial every-other-ply rule sampled White only. The saved games were replayed on all plies and whole families assigned by descending size to balance 72/24/24 source games. No teacher/candidate score determined those assignments. All amendments and initial source splits are retained.

Collected 467 quiet static-evaluation calls from our own bounded searches of the training roots, retaining full history. Stockfish labels use 64,000 nodes. Each record includes the fixed base score, the six feature values, source category/family/game, root and history. Root/test aliases are excluded across current splits; no global claim that every search leaf is absent from all earlier experiments.

## Coherent evaluator

Material values and the existing piece-square tables remain fixed. The added relationships are:

- Mobility on squares not attacked by enemy pawns.
- King pressure from cooperating attackers and missing pawn shelter, reduced in endings and without enemy queens.
- Pressure on attacked pieces, considering cheaper attackers and whether the piece is defended.
- Advanced passed-pawn danger, blockers, support, enemy control and king distance; a larger race bonus is limited to cases without enemy non-pawn pieces.
- Isolated and doubled pawns.
- Bishop-pair value.

These are symmetric, bounded heuristic estimates. Pinned attackers, tactical compensation and forced sequences can still defeat their assumptions. They do not prove a pawn promotes or a king attack wins.

Fitted all six weights jointly on category-balanced quiet search labels using clipped residual Huber loss and regularization toward conservative priors. Development compared four predetermined options: original baseline, exact zero-weight reuse, conservative priors and fitted weights. The fitted weights overemphasized king danger and performed worse at equal depth, so they were rejected. The selected general weights are **[1, 1, 0.25, 1, 1, 1]**, the conservative prior configuration. They are not a learned neural model or a successful fitted-weight result.

## Profiling and cache correction

On five category-spanning development positions, original evaluation occupied about half the profiled search time. The initial piece-block/shared-attack reuse implementation was 18.8% slower at fixed depth. Its static cache unnecessarily distinguished reversible clocks despite the score depending only on the position. An explicit `position_only` trait fixes that for this evaluator; terminal and repetition checks remain unchanged, while clock-aware NN caches retain their original keys. The first partial development run and old profile are preserved; the correction occurred before model selection and final test.

After the correction, zero-weight reuse was still about 8% slower on the summed fixed-depth median timings. Exact move, score and node parity hold. No speedup is claimed. General features also reduce completed depth slightly; acceptance depends on better decisions under the clock. A compiled core remains a possible later step, supported by the profile, rather than an implementation claimed here.

## Searched decisions

The final candidate was frozen by development results. Stockfish confirms distinct chosen moves at 256,000 nodes, preserving full history; all timed probes precede teacher queries. Search has unchanged v24 LMR, depth 3 for equal-depth tests and 750ms/depth ceiling 64 for equal-time tests, with rotated model order and exclusive CPU workloads.

An analytic pawn-race check found a tempo error in the first prototype: giving the opposing king the first move incorrectly increased the promotion bonus. The formula was corrected, the same 467 labels were re-featurized, and development was repeated. The initial prototype games were aborted and retained separately; they do not count toward confirmation. A new 40-position test and 50 new confirmation starts were frozen. The replacement positions are unseen, but reuse the same held-out source families and some source games.

| Corrected frozen test screen | Original mean regret | Candidate mean regret | Original >=150cp mistakes | Candidate |
| --- | ---: | ---: | ---: | ---: |
| Depth 3, 36 common nonmate-score positions | 56.58cp | 49.03cp | 4 | 4 |
| 750ms, 36 common nonmate-score positions | 59.78cp | 48.64cp | 5 | 4 |

Mistake counts include all 40 positions; neither model newly allowed a mate. Avoidable losing transitions remained four each at depth 3 and fell from five to four at 750ms. Average completed depth at 750ms was 4.775 versus 4.5. Common-position means prevent unequal mate-score denominators from making the comparison misleading. Eight cases/category cannot establish comprehensive chess strength.

The predefined searched-choice gates passed: no regression in equal-depth mean regret, strictly better timed mean regret, and no additional large mistakes, avoidable losing transitions or newly allowed mates. This qualified the candidate for a 100-game confirmation on 50 fresh, frozen book starts, both colors, 250ms/search move budget and depth ceiling 64. Opening starts do not overlap the new training labels or roots. Opening families may overlap; this is fresh-game evidence, not globally unseen-family evidence. No book is used after the prefix, no NN, no pondering, no concurrent benchmark/training. Move setup is outside the search budget; the policy measures equal search time rather than total request latency.

## Confirmation and verification

Finished all 100 games on 50 paired openings: **51 wins, 8 draws, 41 losses, 55% score**. There were no unfinished games, interruptions or engine errors. This passes the predetermined exploratory screen (>50%) but does not demonstrate a repeatable advantage. An opening-pair bootstrap (10,000 resamples) gives an approximate 95% interval of 45.5%–64.5%, including 50%. The evaluator remains experimental; the app retains the original evaluator. No Elo claim follows from this comparison.

Mean search elapsed was 242.6ms for baseline and 243.5ms for candidate. Some moves exceeded the nominal budget (maximum overruns 92ms and 192ms); equal nominal budgets are not exact equal elapsed time. Evaluator initialization is outside the timer. These limits further discourage a strong strength claim.

Final audit verifies 100 legal PGNs, zero current-study cross-split position/family overlaps, original-baseline source identity and zero-weight parity/color symmetry across 667 positions. The 126-test suite passes. Saved prototype results and protocol amendments make the rejected branches reviewable.

The concrete result is a reusable general evaluator and independent validation pipeline, with better searched choices on the frozen position screen and a modest game lead. The weight fit did not improve play, the caching work did not produce a speedup, and no neural model was promoted. A separate confirmation is needed before claiming repeatable playing-strength improvement.
