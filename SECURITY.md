# Security policy

## Scope and support

This is an educational demonstration, not a production service. Only the latest commit on `main` is maintained. The API's optional shared key is a basic demo guard, not production authentication; see [Serving and operations](docs/serving.md#api-usage) for deployment limits.

## Reporting a vulnerability

**Please do not open a public issue, pull request, or discussion for a suspected vulnerability.**

Contact the repository owner privately instead:

- Use the repository's **Security** tab → **Report a vulnerability** if it is available to you. This repository's private vulnerability reporting setting has not been verified, so the button may be absent.
- If it is not available, contact the owner through the profile of [@hansdarmawann](https://github.com/hansdarmawann) and ask for a private channel. Do not include exploit details in a public message.

Please include affected files or endpoints, steps to reproduce, impact, and the commit or image tag you tested. Use only synthetic data in reports.

Reports are handled on a best-effort basis; no response time or bounty is promised.

## Known sensitive areas

- **Model artifacts** are loaded with joblib (pickle). Loading an untrusted file can execute arbitrary code. Only load artifacts you produced or can verify; checksums detect corruption, not malice.
- **API exposure.** For anything beyond local use, put the service behind proper authentication, TLS, and rate and body-size limits.
- **Data.** Never submit real employee data to this project or its issue tracker.

## Dependencies

CI runs `pip-audit` on the pinned requirements and Trivy on the container image, and Dependabot proposes updates. Findings that CI misses can still be reported through the private channel above.
