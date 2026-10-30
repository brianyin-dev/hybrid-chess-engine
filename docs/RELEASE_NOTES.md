# v0.1.0 — Local chess application and experimental neural evaluation

## Included

- Browser chess app, Flask API, Python search and strategic-v26 heuristic default.
- Optional Polyglot opening book with external data and checksum verification.
- Experimental neural training/inference and documented controlled tests.
- Corrected first-iteration timeout handling and API used_partial_root diagnostics.
- 166 local automated tests, GitHub Actions CI and desktop/mobile screenshots.
- CI now installs CPU-only PyTorch and NumPy so neural test imports succeed.

## Timeout behavior

Previously, interrupted depth-one search always returned the original emergency
fallback, even after fully evaluating safer alternatives. Search now retains
completed improving root work, publishing it only after child search and any
required PVS full-window verification return normally.

Partial work does not become a completed iteration: depth remains 0 and score
remains null. With no completed root alternative, the legal static fallback is
retained. At depth 2 or later, timeouts retain the last fully completed iteration.
This change does not prune additional branches or alter fixed-depth scores.

Six known-case reruns avoid the two diagnosed fallback blunders. The final
user-limited five-game smoke test scores 1W/3D/1L against the frozen previous
search using the same NN. This is behavioral verification, not demonstrated
strength improvement. See [results](RESULTS.md).

## Reproduce locally

~~~sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ml.run_timeout_wrapup_pilot --output /tmp/chess-timeout-five-new
~~~

The game runner requires optional ML dependencies and the checked-in v35
checkpoint. It plays exactly five games with the frozen openings and refuses to
overwrite output. It does not retrain, expand the match or enable the NN in the
app. The comparator is preserved in benchmarks/pre_timeout/search.py.

The fresh-position screen is recorded by ml.screen_timeout_wrapup; existing
output is protected against overwrite. The extraction/replay/hash audit is in
ml.audit_timeout_wrapup. The original stopped run and interrupted fragment remain
under ml/artifacts/timeout-wrapup/pilot-20/; this is not a completed 20-game test.

## Known limits

- No demonstrated repeatable neural advantage; app remains heuristic-only.
- Historical Stockfish limited-strength results are not human Elo ratings.
- Python evaluation/features limit depth; completing depth one can still choose
  a strategically or tactically inferior move.
- No active hosted demo, online multiplayer, accounts or underpromotion picker.
- No further feature-sharing optimization or architecture/training changes are
  included; scope was intentionally closed after the timeout fix.
