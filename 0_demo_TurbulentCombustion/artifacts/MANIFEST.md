# GL_rbf_CQ release artifacts

The portable inference checkpoint is generated from the immutable
`gl-rbf-cq-v0.9.0-rc1` research checkpoint. Large `.pt` files are intentionally
not committed to ordinary Git history.

## Portable RC1 checkpoint

- Path: `ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_ema_resolved_portable.pt`
- SHA256: `2516ffeb45775d4e6b8d88b4b24d927aac28665a2a90102583e07deaca78f64d`
- Size: 22,023,920 bytes
- Resolved tensor-state SHA256: `f1c92d4bcf1b9e0ac90ad20b3b3468764f6cca200dde1575b914d9ab68d7b99f`
- Source research checkpoint SHA256: `31e59110258d4cc4715e13a5c92efb01d1eec72bdac7da3a3cec384da6f2042a`
- Source: tag `gl-rbf-cq-v0.9.0-rc1`, commit `1b9a6d47f6c248364df6ba54155b5eac3d5e6e67`
- Selection: corrected legacy EMA shadow plus live frozen parameters/buffers

The payload contains one strict-loadable `model` state, normalization mean/std,
field order, public portable config, epoch/global step, and provenance. It does
not contain optimizer, scheduler, live alternate weights, or resume-only EMA
state. Keep the research checkpoint for training resume and audit.

Generate and verify locally:

```bash
python scripts/export_gl_rbf_cq_release.py
python scripts/verify_release_artifacts.py
```

The machine-readable companion is
`artifacts/GL_rbf_CQ_v0.9.0-rc1_portable.json`.

