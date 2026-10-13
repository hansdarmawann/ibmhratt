# Changelog

Changes are recorded here for maintainers and users. Versioning and release
instructions are in [Releases](docs/releases.md). Only dated version sections
represent release notes prepared for an owner-created tag; an entry does not
prove that an image or GitHub Release has been published.

## [Unreleased]

### Added

- A release guide defining semantic versioning, compatibility commitments,
  owner release steps, validation evidence, and rollback.
- A read-only release eligibility command checking the publication ref, package
  version, reviewed-main ancestry, and dated notes for version tags.

### Changed

- GHCR publication is separated from container validation and disabled unless
  the owner enables `ENABLE_GHCR_PUBLISH`. Publication uses the
  `ghcr-publication` environment and the image tested and scanned in the same workflow run and attempt.
- Version-tag publication requires a stable tag matching the package version
  and a commit contained in reviewed `main`.

## Existing source baseline

The current source declares `1.1.0` in `src/attrition/__init__.py`. No release
history or dates are inferred from that declaration; no local version tags
were present when this changelog was introduced. Previously implemented
capabilities include training-only selection boundaries, verified schema-2
bundles with explicit legacy schema-1 loading, and configurable artifact paths.
These are baseline context, not claims of releases published by this PR.
