## Objective
What problem does this PR solve?

## Scope and files changed
-

## Design and compatibility
- Existing behavior preserved:
- New interfaces/configuration:
- ML/data-boundary impact:

## Validation performed
| Command | Result |
|---|---|
| `python -m ruff check .` | PASS / FAIL / NOT RUN (reason) |
| `python -m mypy` | PASS / FAIL / NOT RUN (reason) |
| `python -m pytest -q` | PASS / FAIL / NOT RUN (reason) |

## ML leakage checklist
Tick all that apply, or write N/A when the PR does not touch data, features, models, or evaluation.
- [ ] Holdout data, labels, and metrics are not used to choose models, hyperparameters, features, or thresholds.
- [ ] Learned transforms are fitted inside training folds only.
- [ ] Model, tuning, and threshold choices use training data only.
- [ ] The served model, `models/current.json`, the inference contract, and committed baseline reports are unchanged (or the change is explained and approved).
- [ ] Reported numbers come from generated artifacts, not hand edits.
- [ ] Tests cover the changed behavior with deterministic seeds.

## Security / responsible use
- [ ] No real employee data, secrets, or credentials are included.
- [ ] The change does not enable individual employment decisions.

## Risks and rollback
-

## Deferred work / owner decisions
-
