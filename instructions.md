# IBM HR Attrition — Repository Improvement & AutoML Execution Plan

> **Audience:** Claude Code or OpenAI Codex working in `hansdarmawann/ibmhratt`  
> **Repository:** <https://github.com/hansdarmawann/ibmhratt>  
> **Plan date:** 2026-10-13  
> **Status:** Proposed work — **not** evidence that any listed PR has been implemented or merged.  
> **Operating mode:** Implement **one scoped pull request per session** on its own local branch, validate it, and stop for human review. **The repository owner manually stages, commits, pushes, opens PRs, and merges.**

## 0. Mission and non-negotiable rules

Improve documentation, governance, package portability, evaluation integrity, tests, API security, and releases. Add **optional, reproducible AutoML with Optuna** while preserving the existing scikit-learn baseline, FastAPI and Streamlit interfaces, model artifact format, and responsible-use restrictions.

**Never:**

1. Use reserved holdout metrics, data, or labels to optimize models, hyperparameters, features, or decision thresholds. The published holdout has already been examined; do not treat it as a fresh independent benchmark for repeated AutoML iterations.
2. Change the active/served model, `models/current.json`, public inference contract, existing baseline reports, or reproducibility settings without an explicit PR requirement, appropriate tests, and human approval.
3. Convert this educational demonstration into a tool for hiring, firing, promotion, disciplinary action, or other individual employment decisions; never introduce real personnel data.
4. Rewrite unrelated modules, weaken existing security checks, silently loosen CI, remove existing tests, or claim a test passed without running it.
5. Choose a source-code license, assert dataset redistribution rights, publish a Docker image/release, change GitHub settings, merge a PR, or deploy a new model without repository-owner approval.
6. Run `git add`, `git commit`, `git push`, `git merge`, `git rebase`, `git reset`, `git cherry-pick`, `gh pr create`, or any equivalent staging, commit, publishing, or history-changing command. **All Git commits are performed manually by the repository owner.** Agents may inspect Git state/diffs and create the agreed local feature branch only if the working tree is safe. Never force-push or run destructive Git commands.

**Engineering defaults:** Python 3.12; respect existing `pyproject.toml`, requirements/lock-file strategy, Ruff, mypy, pytest, Linux/Windows CI, and existing documentation. Prefer small changes, explicit config, deterministic tests, and backward compatibility. Use Conventional Commits.

### Agent instructions — read before modifying anything

1. Inspect `README.md`, `pyproject.toml`, `environment.yml`, `Makefile`, `.github/workflows/`, `docs/`, `src/attrition/config.py`, `src/attrition/models/train.py`, `src/attrition/models/`, `src/attrition/artifacts.py` (or actual equivalent), `app/`, `tests/`, and relevant requirements. **Verify paths instead of assuming all suggested files exist.**
2. Print the current branch, working-tree status, default-branch HEAD, existing tests, and baseline behavior. Preserve any existing uncommitted user changes; never overwrite them.
3. Check whether the selected PR's tasks are already implemented, partially implemented, or obsolete. Adapt to actual source; do not create duplicate files or features.
4. State a concise plan for the **selected PR only**, identify impacted files and risks, implement, add tests, verify, and summarize. Do not execute subsequent PRs automatically.
5. If repository access, network, dependencies, dataset rights, tests, or GitHub privileges are missing: finish all safe local work; report precisely what could not be checked. Do not fabricate results or silently skip acceptance criteria.
6. Never modify unrelated PR branches. If parallel work is required, use separate Git worktrees rather than sharing a checkout.
7. **Do not stage, commit, push, publish a PR, or merge changes.** Leave changes unstaged on the selected local feature branch. Supply a suggested Conventional Commit message and a ready-to-paste PR description; the owner performs all Git write/publishing operations manually.

## 1. How to run this plan with Claude Code or Codex

Place this file at the **repository root** as `instructions.md`. Codex typically discovers `AGENTS.md`, while Claude Code typically discovers `CLAUDE.md`; **this `instructions.md` is not assumed to auto-load**. Explicitly tell the agent to read it. Optionally create a tiny `AGENTS.md` or `CLAUDE.md` pointing to this file, only if desired and approved.

**Starter prompt (copy/paste):**

```text
Read instructions.md completely and inspect the current repository before changing code.
Execute PR 01 only: docs/repository-governance.
Follow all guardrails, update documentation and tests as needed, and work only on the specified
local feature branch. Do not stage, commit, push, create a PR, merge, publish, or start PR 02.
I will run git add, git commit, git push, and open the PR manually.
Report changed files, commands actually run, test results, risks, a suggested commit message,
and a ready-to-paste PR description.
```

For subsequent runs, replace `PR 01` and the branch name with the next row of the execution matrix. Prefer a new context/session for each PR to reduce context drift. The agent must **never** create a commit or remote PR; it prepares only local file changes, verification evidence, a suggested commit message, and a ready-to-paste PR summary.

### Branch and PR lifecycle — human-controlled commits

**Agent responsibilities (local only):** inspect the working tree, confirm prerequisites are merged, create or switch to the requested **local feature branch** if safe, edit files, run tests, and show `git status`, `git diff --check`, and `git diff --stat`. **Leave all edits unstaged and uncommitted.** If there are pre-existing changes or a branch conflict, do not discard/rebase them; report the blocker.

**Repository owner responsibilities (manual commands):** after reviewing the agent's work, run the following personally in a terminal. These commands are examples **for the owner, not instructions for the coding agent to execute**:

```bash
# Run by the repository owner ONLY, after reviewing files/tests.
git status --short --branch
git diff --check
git diff
# Review and stage only files belonging to this PR:
git add <specific-reviewed-files>
git diff --cached
# Example for PR 01; change message and branch for other PRs:
git commit -m "docs: add contribution and security guidance"
git push -u origin docs/repository-governance
# Open a PR against main manually in the GitHub UI (or run gh yourself).
```

- Use **one feature branch per PR**. The agent can create the local branch after checking the tree is clean/safe; the owner is free to create it in advance instead.
- Create independent branches from the latest reviewed `main`. For dependent PRs, wait for prerequisite PRs to merge; then start from updated `main`.
- If stacked branches are required, obtain a specific owner decision and document the base; **the agent still must not rebase, commit, or push**.
- Keep unrelated work out of scope. Do not run multiple agents on the same checkout.
- PR labels here are **plan identifiers**, not pre-existing GitHub PR numbers.

### Model and thinking-effort recommendations per PR

Choose **either** the GPT/Codex or Claude Code option for each PR. These are suggested starting settings based on task complexity, not performance guarantees. **High** is a sound default; use **Extra High (`xhigh`)** for the most reasoning-sensitive work. Availability depends on account, client, and model selection.

| PR | GPT / Codex alternative | GPT effort | Claude Code alternative | Claude effort | Preferred starting choice |
|---|---|---|---|---|---|
| **01** Governance, README | GPT-6.1 Sol | Medium | Claude Sonnet 5.5 | Medium | **GPT-6.1 Sol · Medium** |
| **02** Holdout / leakage guardrails | GPT-6 Astra | Extra High | Claude Opus 5.5 | Extra High | **GPT-6 Astra · Extra High** |
| **03** Paths and wheel packaging | GPT-6.1 Sol | High | Claude Opus 5.5 | High | **GPT-6.1 Sol · High** |
| **04** Optuna AutoML core | GPT-6 Astra | High | Claude Opus 5.5 | High | **GPT-6 Astra · High** |
| **05** Nested CV / AutoML evaluation | GPT-6 Astra | Extra High | Claude Opus 5.5 | Extra High | **GPT-6 Astra · Extra High** |
| **06** AutoML experiment tracking | GPT-6.1 Sol | High | Claude Sonnet 5.5 | High | **GPT-6.1 Sol · High** |
| **07** Testing and coverage | GPT-6.1 Sol | High | Claude Sonnet 5.5 | High | **Claude Sonnet 5.5 · High** |
| **08** API/container security | GPT-6 Astra | High | Claude Opus 5.5 | High | **GPT-6 Astra · High** |
| **09** Model card | GPT-6.1 Sol | High | Claude Sonnet 5.5 | High | **Claude Sonnet 5.5 · High** |
| **10** Release management | GPT-6.1 Sol | Medium | Claude Sonnet 5.5 | Medium | **GPT-6.1 Sol · Medium** |
| **11** Optional model promotion | GPT-6 Astra | Extra High | Claude Opus 5.5 | Extra High | **GPT-6 Astra · Extra High** |
| **12** Optional FLAML benchmark | GPT-6 Astra | High | Claude Opus 5.5 | High | **Claude Opus 5.5 · High** |

**Model selection notes:**

- **GPT-6.1 Sol** is the economical default for scoped coding, tests, and documentation. **GPT-6 Astra** is appropriate for complex ML evaluation, trusted model promotion, and security-sensitive architecture.
- **Claude Sonnet 5.5** is the efficient Claude Code option for everyday changes; **Claude Opus 5.5** is better suited to complex or long-running coding tasks.
- `Extra High` is sometimes labeled `xhigh` in tooling. If an exact model or effort option is unavailable, use the strongest available equivalent; do not assume aliases never change.
- For each PR: use **one** primary implementation agent and optionally a **different** model for a read-only final review. Do not have both editing the same checkout simultaneously.
- Agent choice does **not** change Git permissions: **the owner alone runs `git add`, `git commit`, `git push`, and PR creation/merging.**
- Official references (checked 2026-10-13): [OpenAI models](https://developers.openai.com/api/docs/models), [OpenAI model selection](https://developers.openai.com/api/docs/guides/model-selection), [Claude models](https://platform.claude.com/docs/en/models/overview), [Claude model selection](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model), [Claude Code model configuration](https://code.claude.com/docs/en/model-config).

## 2. Execution matrix and dependencies

| Plan ID | Branch | Scope | Prerequisites | Priority |
|---|---|---|---|---|
| PR 01 | `docs/repository-governance` | README, contribution/security guidance, issue/PR templates, licensing assessment | None | High |
| PR 02 | `fix/evaluation-guardrails` | Holdout claims, split/selection tests, methodology text | None | High |
| PR 03 | `refactor/configurable-paths` | Configurable paths, packaging/wheel checks | None | High |
| PR 04 | `feat/automl-optuna-core` | Optional Optuna search engine, CLI, dependency, smoke tests | PR 02 and PR 03 merged | High |
| PR 05 | `feat/automl-evaluation` | Nested evaluation, OOF threshold rules, leakage checks | PR 04 merged | High |
| PR 06 | `feat/automl-experiments` | Trial recording, reproducibility metadata, comparison reports | PR 05 merged | Medium |
| PR 07 | `test/coverage-hardening` | Test-only coverage reporting and behavioral regression tests | None; branch from updated main | Medium |
| PR 08 | `security/api-hardening` | Serving validation, authentication tests, deployment boundaries | None; branch from updated main | Medium |
| PR 09 | `docs/model-card` | Model card and scientific/responsible-use documentation | PR 05 if discussing AutoML outputs | Medium |
| PR 10 | `chore/release-management` | Release process, versioning, changelog | None; branch from updated main | Low |
| PR 11 | `feat/automl-promotion` | Controlled, explicit promotion of approved AutoML models | PR 06 merged; owner approval | Optional |
| PR 12 | `experiment/flaml-benchmark` | Separate FLAML baseline experiment (not default serving) | PR 06 merged | Optional |

**Suggested execution:** PR 01 → PR 02 → PR 03 → PR 04 → PR 05 → PR 06 → PR 07/08 → PR 09 → PR 10. PR 11/12 only when there is a concrete need. PR 07/08 may be handled earlier since largely independent.

## 3. Detailed acceptance criteria by PR

### PR 01 — Documentation and governance

**Targets:** `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`, relevant `docs/`.

- [ ] Keep the README a concise landing page with architecture, prerequisites, quickstart, entry points for FastAPI/Streamlit, accurate evaluation interpretation, and links to detailed guides.
- [ ] Add a working request sample using `examples/employee.json`; check endpoint and authentication assumptions against actual API code. Explain the response shape without inventing probability values.
- [ ] Explain current selected-model metrics and small-sample limitations. Synchronize any published numbers with generated artifacts; do **not** manually replace computed report sections.
- [ ] Add development setup, lint/type/test commands, PR guidelines, ML evaluation rules, and the prohibition on real employee data to `CONTRIBUTING.md`.
- [ ] Add a safe vulnerability disclosure path in `SECURITY.md`; link to GitHub private reporting **only if enabled**. Never expose security reports in public issues by default.
- [ ] Provide reproducible bug/feature templates and a PR template with an ML leakage checklist.
- [ ] **Investigate** code and bundled dataset licenses. Do not insert a guessed `LICENSE`; raise an explicit owner-decision item if rights are unverified.
- [ ] Verify internal links, documented commands, and markdown integrity.

**Done when:** New contributors can set up the repo and know how to propose changes, with no fabricated licensing or security claims.

### PR 02 — Holdout and selection guardrails

**Targets:** `src/attrition/models/train.py`, associated evaluation modules/tests, `README.md`, `docs/`.

- [ ] Trace stratified split creation, preprocessing fitting, CV model comparison, frozen model choice, out-of-fold (OOF) threshold choice, and holdout evaluation.
- [ ] Confirm **all selections occur using training data only**; check any tuning diagnostics, engineered-feature ablations, and calibration steps.
- [ ] The repo currently reports holdout metrics for multiple candidate models; do not describe the holdout as evaluated strictly once if code runs multiple evaluations. Label non-selected comparisons **post-selection descriptive diagnostics** or restrict presentation after reviewing downstream report dependencies.
- [ ] Establish regression tests proving selector/tuner routines cannot obtain holdout labels or metrics; tests should fail when that boundary is violated.
- [ ] Preserve selected baseline metrics and artifacts absent a proven correction; explain any necessary changes.
- [ ] Explicitly distinguish ordinary CV, nested CV, OOF threshold diagnostics, and final held-out evaluation.

**Done when:** The procedural boundary is enforced by tests and stated accurately in published documentation.

### PR 03 — Configurable paths and package portability

**Targets:** `src/attrition/config.py`, path consumers, `pyproject.toml`, packaging tests, setup docs.

- [ ] Inspect existing checkout-derived `ROOT` and paths for dataset, model bundles, `models/current.json`, metrics, and figures.
- [ ] Add environment overrides or a small validated configuration object; preserve existing checkout defaults and compatibility.
- [ ] Do not allow untrusted API requests to specify arbitrary model files, pickle/joblib paths, or write destinations.
- [ ] Normalize/validate external path handling for Linux and Windows; distinguish read-only checks from directory creation.
- [ ] Build a wheel; install it in a clean environment outside the checkout; verify imports and at least a supported CLI path with explicitly supplied data/artifact paths.
- [ ] Review `pyproject.toml`'s dynamic dependencies and requirements synchronization without introducing redundant dependency lists.
- [ ] Add path override and wheel-installation tests, update setup documentation, and disclose remaining packaging limits.

**Done when:** External artifact paths work without editing code and supported wheel-based operations do not require repository-relative imports.

### PR 04 — Optional Optuna AutoML core

**Targets:** new `src/attrition/models/automl.py` (or cohesive package), optional dependency declaration, minimal unit tests, short usage docs.

**Design constraints:**

- [ ] Integrate Optuna with **existing sklearn preprocessing pipelines**, applying all learned transforms within each CV training fold. Reuse existing feature schema/validation and seed conventions.
- [ ] Search **logistic regression**, **random forest**, and **histogram gradient boosting** initially; treat a dummy classifier as a non-tuned reference if appropriate.
- [ ] Use *conditional* parameter spaces appropriate to each estimator. Avoid invalid solver/penalty combinations and overly broad ranges on the small dataset.
- [ ] Default primary optimization score: **average precision (AP)**, estimated solely by stratified training-fold CV.
- [ ] Start with 5 folds, seeded sampler (e.g. Optuna `TPESampler(seed=...)`), serial execution (`n_jobs=1`) for reproducibility, and configurable trial/time budget; recommend about 30 trials/900 seconds for manual experiments, not CI.
- [ ] Provide an explicit opt-in command such as `python -m attrition.models.automl --n-trials 30 --timeout 900 --seed 42`. This is a **target interface**, not assumed to exist yet.
- [ ] Keep baseline `python -m attrition.models.train` unchanged; add no implicit AutoML or retraining during imports or app startup.
- [ ] Never read or touch holdout labels, generate final holdout scores, update `models/current.json`, or publish model bundles as a side effect of a search trial.
- [ ] Add unit tests with tiny synthetic data and a 1–2 trial deterministic smoke test. Bound trial runtime and resources.
- [ ] Make Optuna optional (`requirements-automl.txt`, appropriate extras, or another repo-consistent mechanism). If a lock file is authoritative, update it consistently. Do not add Optuna to serving images unnecessarily.

**Done when:** A controlled search produces a reproducible **candidate configuration and training-CV score**, without changing the served system.

### PR 05 — AutoML evaluation integrity

**Targets:** AutoML evaluation/tuning modules and tests, methodology docs.

- [ ] Implement nested CV around the **entire** AutoML selection procedure for less biased development estimates: outer folds for assessment, inner folds for Optuna search. All nesting stays inside the *original training partition*.
- [ ] Prevent reuse of outer validation data for inner trial selection. Fit preprocessing within appropriate folds.
- [ ] Apply an explicitly specified, training-only method for threshold selection (e.g. OOF predictions and F2), only after model/hyperparameter choices are made inside their allowed partition.
- [ ] Compare the existing selection procedure and AutoML on aligned partitions/metrics. Explain uncertainty and avoid treating an apparent gain on a small fictional dataset as evidence of generalization.
- [ ] Keep the previously published holdout separate from iterative AutoML comparisons. If a new, independent test set is unavailable, describe comparisons as development estimates, not a new unbiased final test.
- [ ] Add tests preventing data-flow leakage and preventing threshold tuning on outer-fold validation labels; use synthetic fixtures and appropriate mocking/instrumentation.
- [ ] Limit nested-CV CI smoke tests to very small trial counts and/or marked opt-in tests to avoid excessive CI duration.

**Done when:** Model-family and hyperparameter search, threshold choice, and performance estimation have explicit, tested boundaries.

### PR 06 — Experiment tracking and reporting

**Targets:** AutoML reporting code, `reports/automl/` if appropriate, documentation, tests.

- [ ] Log trials, estimator family, sampled hyperparameters, CV results, failures, elapsed time, objective name, seed, folds, dependency versions, and git revision when available.
- [ ] Use reproducible serialization (JSON/CSV) for lightweight summaries; optionally provide SQLite storage for Optuna study continuation, **not committed as a mutable database**.
- [ ] Save enough metadata to reconstruct the chosen candidate without making untrusted joblib files loadable from arbitrary paths.
- [ ] Produce a comparison table/plot against baseline development-CV metrics. Keep historical reports isolated from the baseline auto-generated report contract.
- [ ] Ensure repeated runs cannot silently overwrite committed baseline results; choose explicit output directories/run IDs.
- [ ] Add tests for metadata schema, trial failures, repeatable summarization, and backwards-compatible report paths.
- [ ] Document reproducibility limitations (versions, platform, multithreading, stochastic estimators, dataset size).

**Done when:** Trials are inspectable and reproducible without affecting the canonical baseline reports or serving state.

### PR 07 — Behavioral tests and coverage

- [ ] Preserve the existing combined-workflow coverage gate; add distinct **pytest-only** coverage reporting so executed training lines are not mistaken for assertion coverage.
- [ ] Add focused tests for schema errors, boundary values, batch size, authentication, missing/corrupt artifacts, incompatible metadata, external paths, and cross-platform assumptions as applicable.
- [ ] Do not raise coverage only by adding non-asserting smoke tests; justify new thresholds against the measured baseline.
- [ ] Ensure Linux/Windows and existing notebooks/CI still pass, or give exact reasons when not executable locally.

**Done when:** Critical observable behaviors have regression coverage and coverage reports are accurately labeled.

### PR 08 — API and container security

- [ ] Review actual `app/api.py` contracts. Preserve local demo usability but describe **required** gateway authentication, TLS, rate/body limits, trusted artifact provisioning, and monitoring for any public exposure.
- [ ] Test missing/incorrect API keys, malformed payloads, unexpected fields, batch limits, and safe error responses.
- [ ] Keep integrity checks for artifacts, narrow allowed paths, and avoid logging secrets or sensitive payloads.
- [ ] Preserve pinned Actions, least-privilege workflow permissions, pip-audit/Trivy, and container smoke tests.
- [ ] Check whether the Docker runtime can run non-root without breaking volume access; test before changing.
- [ ] Do not represent API-key authentication as sufficient production security.

**Done when:** Public-deployment boundaries are explicit and verified defensive behavior is better, without breaking the local demo.

### PR 09 — Model card and research transparency

- [ ] Add `docs/model-card.md` and link it from README.
- [ ] Document dataset provenance, fictional nature, permitted/prohibited use, splits, baseline/AutoML distinction, metric definitions, uncertainty, calibration, subgroup limitations, feature attribution limitations, small-sample caveats, and reproducibility commands.
- [ ] Separate actual measured results from proposed or hypothetical AutoML outcomes; derive numerical results from artifacts.
- [ ] Avoid implying predictions are causal or suitable for individual employment decisions.

**Done when:** A reviewer can trace every stated model result to its method and artifact.

### PR 10 — Releases and changelog

- [ ] Introduce `CHANGELOG.md` and document a consistent version/tagging strategy after reviewing existing version metadata.
- [ ] Document backward compatibility for API endpoints, artifact schema, and command-line interfaces; explain when a breaking change warrants a major version.
- [ ] Verify publication workflows are appropriately gated and do not release artifacts from arbitrary unreviewed branches.
- [ ] Do not publish releases or change branch protection without owner approval; report required manual GitHub settings separately.

**Done when:** Maintainers can reproduce a reviewed release with clear change notes and version compatibility.

### PR 11 — OPTIONAL: Approved AutoML candidate promotion

Only begin with explicit owner approval and after PR 06. Keep a **manual promotion gate**: documented candidate review, provenance, data-separation checks, artifact-schema tests, API/dashboard smoke tests, and rollback of atomic current-run pointer. Do not simply replace the served model with the highest CV AP. Protect baseline reports and prior bundles.

### PR 12 — OPTIONAL: Independent FLAML benchmark

Implement FLAML as a separate, optional environment/CLI and compare against the same **training-only evaluation protocol**; use explicit resource/time limits. Do not mix framework selection with holdout scores and do not add FLAML to the default serving dependency list. Report added installation/CI cost before merging.

## 4. Required verification gates

Discover actual commands and environment before running. Depending on installed tools, representative checks are:

```bash
python --version                     # target: Python 3.12
python -m pytest -q
python -m ruff check .
python -m mypy src app scripts
python -m build                     # when build tooling is installed; packaging-related PRs
```

Use repository-provided `Makefile` targets or official CI equivalents when they differ. Also run the project-specific training/report-reproducibility checks, notebook checks, Windows validation, and Docker/security tests **for PRs touching those systems**. Running full training may rewrite tracked reports or bundles; inspect `git status` before and after and do not commit regenerated assets unintentionally.

For AutoML:

- [ ] Tiny deterministic smoke test: 1–2 trials, small synthetic classification data, bounded runtime.
- [ ] End-to-end trial demonstrates feature preprocessing stays within CV folds.
- [ ] Holdout labels/results cannot reach optimizer or threshold selector.
- [ ] Baseline CLI outputs and served pointer remain identical unless explicitly approved.
- [ ] Trial/report metadata captures seed and fold configuration.
- [ ] Large experiments are opt-in (manual or scheduled workflow) rather than per-PR mandatory.

### Required PR description

```markdown
## Objective
What problem does this PR solve?

## Scope and files changed
- ...

## Design and compatibility
- Existing behavior preserved:
- New interfaces/configuration:
- ML/data-boundary impact:

## Validation performed
| Command | Result |
|---|---|
| `...` | PASS / FAIL / NOT RUN (reason) |

## Security / responsible use
- ...

## Risks and rollback
- ...

## Deferred work / owner decisions
- ...
```

Do not mark a verification item `PASS` unless the command executed successfully. Distinguish `NOT RUN`, `BLOCKED`, and `FAIL`.

## 5. Session stop condition

At the end of **each PR session**:

1. Show the local feature branch, `git status`, diff summary, and changed files. Confirm **no files were staged, no commits were created, and nothing was pushed or published by the agent**. Do not invent commit hashes or remote PR URLs.
2. Report precise test commands and results, any changed reports/artifacts, compatibility impact, security implications, unresolved owner decisions, **a suggested Conventional Commit message**, and **a ready-to-paste PR description**.
3. Identify the next eligible PR **but do not start it**. Remind the owner to review, stage, commit, push, and open the PR manually.
4. Leave `main`, production artifacts, GitHub settings, and deployments untouched unless that PR was explicitly approved to modify them. No agent-run `git add`, `git commit`, `git push`, merge, or PR creation under any circumstances.

## 6. Definition of done for the roadmap

The roadmap is complete when all approved non-optional PRs are merged with passing applicable CI, methods and documentation accurately reflect the code, tests protect training/holdout boundaries, AutoML runs independently with bounded/reproducible experiments, and the active demo behavior has not silently changed. Optional promotion and FLAML benchmarking remain separate owner decisions.

**Core philosophy:** Correctness, reproducibility, safety boundaries, and reviewable history are more important than maximizing an AutoML score or increasing the number of features.
