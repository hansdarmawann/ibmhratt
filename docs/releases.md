# Releases

[Back to README](../README.md)

Releases package this fictional-data educational demonstration. They do not
approve employment decisions, promote an AutoML candidate, or deploy a service.
The owner reviews, commits, tags, pushes, and publishes. Agents leave changes
unstaged and never perform those actions.

## Versions and tags

`src/attrition/__init__.py` is the single version source (`__version__`, currently
`1.1.0`). Setuptools reads it through the dynamic version in `pyproject.toml`;
FastAPI uses it for its service version and `/model`'s `api_version`. Do not add a
second hard-coded package version. Update version-specific wheel examples when
bumping it, and verify built wheel metadata with the packaging tests.

Use stable semantic versions `MAJOR.MINOR.PATCH` and annotated Git tags
`vMAJOR.MINOR.PATCH`, exactly matching the source version. Leading zeroes,
prerelease suffixes, and build suffixes are not supported by the publication
checker. There is no automated version bump, tag creation, GitHub Release, or
PyPI upload. The existing `1.1.0` declaration is not evidence of a published
release. Record new work under `[Unreleased]` in [CHANGELOG.md](../CHANGELOG.md).
Before tagging, move the reviewed notes into a section such as
`## [1.1.1] - 2026-10-13` using the actual release date and chosen version;
retain an empty `[Unreleased]` section for future changes.

- **PATCH:** compatible fixes or documentation/security corrections that retain
  the supported contracts. A stricter validator that rejects formerly documented
  valid inputs is a breaking change, even if motivated by security.
- **MINOR:** compatible capabilities, optional flags, or additive response fields.
  Clients must tolerate unknown response fields; requests still forbid extras.
- **MAJOR:** removing/renaming an endpoint or supported flag, changing a response
  field's type/meaning, requiring new input fields, or removing supported artifact
  loading without a compatible migration path. Default or authentication changes
  that break documented usage also require a major version and migration notes.

Keep package version, bundle schema version, and model run identity separate.
A package bump does not imply a schema bump or a new model. Document deprecations
before removal and include upgrade, migration, and rollback instructions for
breaking changes. Security fixes may need an expedited review but still need
explicit compatibility notes.

## Compatibility contract

| Surface | Current supported contract | Release review |
|---|---|---|
| HTTP | `GET /`, `/health`, `/model`; `POST /predict`, `/predict/batch`; OpenAPI/docs | Preserve fields, types, status meanings, input validation, ordering, and `X-API-Key` behavior. `/health` is 200 when ready and 503 otherwise; model endpoints return 503 when unavailable. |
| Prediction | `prediction`, `attrition_probability`, `threshold`, nullable `run_id`; batch `count` and ordered `predictions` | Probability is an estimate; the label uses the stored threshold. New response fields can be additive, but consumers must ignore unknown fields. |
| Bundle schema | Schema 2: `current.json`, UUID run directory, checksummed manifest, pipeline and matching metadata/report | Keep all integrity checks. Unsupported schemas fail; do not silently load them. Test any schema migration and retain a reviewed previous pointer for rollback. |
| Legacy artifact | Explicit trusted standalone schema-1 `.joblib` loading; schema-2 pipelines require verified bundles | Removing this legacy path is breaking. Pickle/joblib is trusted-operator input only; matching schema numbers do not promise cross-library binary compatibility. |
| CLI | `python -m attrition.models.train` with `--with-shap`, `--data`, `--output-root`, `--models-dir`; `predict` with JSON input and `--model`; `prune` with `--keep`, `--dry-run`; drift with `--bundle`, `--output`, `--fail-on-shift` | Preserve documented flags, defaults, JSON output meaning and exit behavior. Repository `scripts/` and `app/` are checkout tools, not included in the wheel. |
| Configuration | `ATTRITION_ROOT`, `ATTRITION_DATA_PATH`, `ATTRITION_MODELS_DIR`, `ATTRITION_API_KEY` | Preserve path precedence and safe artifact provisioning. Do not expose operator paths through requests. |

Scikit-learn artifacts are tied to their recorded Python/dependency versions;
use `requirements-lock.txt` and fail on incompatible loading. A release must
not change the selected model, threshold, data split, or published metrics
implicitly. Docker builds train a bundle from the locked baseline; release tests
must verify report reproducibility, and production promotion remains a separate
owner decision.

## Publication gates

[ci-cd.yml](../.github/workflows/ci-cd.yml) keeps Linux/Windows validation,
combined coverage, reproducibility/notebook/live-service checks, pip-audit,
container smoke tests, Trivy, and pinned actions/images. CI evidence uploads are
short-lived workflow artifacts, not releases. PRs, manual dispatches, schedules,
and forks cannot publish a container. The container job has read-only repository
permissions. Only the separate `publish` job has `packages: write`.

Publication requires all validation, audit, and container jobs to succeed, a
non-fork push to `main` or a `v*` tag, and repository variable
`ENABLE_GHCR_PUBLISH` equal to the literal string `true`. The job references
`ghcr-publication` for owner-configured approval. Before login, the read-only
`scripts.check_release` command checks that HEAD equals the publication ref and
is an ancestor of `origin/main`. Tags must be stable, match `__version__`, and
have a dated changelog section. Missing refs, malformed tags, mismatches, and
unmerged feature-branch tags fail closed. Full checkout history is required.

The job downloads `tested-container-<run attempt>` from its own workflow run
and attempt with GitHub CLI (read-only Actions access),
loads the saved image, and pushes that same tested image without rebuilding.
The image artifact is retained for one day; if approval is delayed past expiry,
rerun the complete validation workflow to produce fresh evidence and an image.
Main publishes mutable `latest` and `sha-<full commit SHA>` tags. A version push
publishes `vX.Y.Z` and its SHA tag, leaving `latest` unchanged. Run IDs identify
bundles, not release versions. Record the resulting registry digest; use digest
pins for repeatable deployments instead of mutable `latest`.

### Required manual GitHub settings

Before enabling publication, the owner must:

1. Protect `main` with reviewed pull requests and required CI checks for both
   validation platforms, audit, and container validation; restrict bypasses.
2. Create `ghcr-publication`, configure required reviewers, prevent self-review
   and administrator bypass where supported, and permit only `main` branches
   and `v*` tags using separate environment ref rules.
3. Restrict creation/update/deletion of `v*` tags to authorized release owners
   through a tag ruleset. Never move or reuse a published release tag.
4. Confirm GitHub Actions/package permissions and GHCR repository access. Decide
   package visibility only after resolving code licensing and bundled dataset
   redistribution rights; see [Licensing status](../README.md#licensing-status).
5. Set `ENABLE_GHCR_PUBLISH=true` only after the protections and rights review are
   complete. Keep it unset/false when publication is not approved.

An environment name alone does **not** configure reviewers. GitHub can create a
missing environment without protection rules, so the opt-in variable must remain
disabled until the owner has configured and verified it. Protection availability
depends on repository visibility and GitHub plan; see
[GitHub environment documentation](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).
If required protection is unavailable, leave automated publication disabled.
These remote settings were not inspected or changed as part of this PR.

## Owner release procedure

1. Review scope and compatibility, resolve publication rights, choose the version,
   update its single source, finalize dated change notes, and review the diff.
2. In a clean Python 3.12 environment use `python -m pip install -r
   requirements-dev.txt -c requirements-lock.txt`. Run `python -m pip check`,
   `python -m scripts.check` (regenerates outputs),
   `python -m pytest -q tests/test_packaging.py tests/test_releases.py`, and
   `python -m pip wheel . --no-deps --wheel-dir dist`. Check metadata and exclude
   unintended generated assets. The wheel includes only `attrition`, not apps or
   dataset; no wheel is automatically published.
3. Commit and merge reviewed changes manually. Wait for the merged commit's
   required CI, security, reproducibility and container checks. Update the owner
   checkout of `main` and its remote refs manually. Do not tag a feature branch.
4. The owner creates the annotated tag on the reviewed commit. Before pushing it,
   run the following read-only checks from that exact checked-out commit (example
   version only):

   ```bash
   python -m scripts.check_release --ref refs/heads/main
   python -m scripts.check_release --ref refs/tags/v1.1.1
   git status --short --branch
   git show --no-patch v1.1.1
   ```

   The tag must already exist locally for its check to pass. The checker performs
   no network fetch and uses the current local `origin/main`; keep it current.
5. Push the reviewed tag manually. The tag workflow reruns every required check;
   inspect the exact commit and environment approval request before approving.
   Record the run URL, tag/commit, image digest, test/audit evidence, and release
   notes. The owner may create a GitHub Release manually using those notes after
   successful publication. Do not attach mutable model bundles as a release
   shortcut or claim that local tests are remote CI evidence.
6. Confirm the image digest and run a deployment smoke test before any separate
   deployment approval. A code/image release does not update a running API's
   model pointer, and replacing an image is not artifact-schema migration.

## Failures and rollback

If a check fails, withhold publication, correct it in a reviewed PR, and use a
new version/tag if a release was already published. Disable `ENABLE_GHCR_PUBLISH`
to stop future automated publication; cancel a pending run separately if needed.
Never bypass failing checks, retag an existing release, or delete another run's
artifacts. Deployment rollback uses a recorded previous image digest and trusted
compatible bundle/pointer after smoke tests. Keep prior bundles and release
notes; follow [Serving and operations](serving.md) for the active-run mechanism.
