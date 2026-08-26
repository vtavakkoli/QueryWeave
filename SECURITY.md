# Security policy

## Supported versions

QueryWeave is currently pre-1.0. Security fixes are applied to the latest code on `main` and, when practical, to the latest tagged release.

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public GitHub issue, discussion, pull request, benchmark report, or social-media post before a fix is available.

Use GitHub's private vulnerability reporting / Security Advisory flow for this repository when available. Include:

- affected component and version/commit;
- reproduction steps or a minimal proof of concept;
- expected and observed behavior;
- security impact;
- deployment assumptions required for exploitation;
- any proposed mitigation.

Reports that concern a dependency should identify the dependency and advisory/CVE when known.

## Security scope

QueryWeave processes untrusted query and document content, so deployments should assume request data can be adversarial. Production operators should:

- keep request/body and concurrency limits enabled;
- place public deployments behind appropriate authentication/authorization and network controls;
- avoid exposing destructive administration endpoints such as index reset to untrusted callers;
- run the container as the provided unprivileged user;
- keep Rust, Python, container base images, Tantivy, USearch, and other dependencies updated;
- treat external embedders, rerankers, and remote model services as separate trust boundaries;
- avoid logging sensitive document/query content unless explicitly required.

The bundled HTTP service does not currently implement authentication or tenant isolation. Those controls must be supplied by a trusted gateway/service mesh or application layer when required.
