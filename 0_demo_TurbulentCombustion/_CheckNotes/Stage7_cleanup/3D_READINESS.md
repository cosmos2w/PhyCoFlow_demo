# 3-D code-readiness audit

## Finding

The GL-RBF/CQ point-cloud path is code-ready for three varying coordinate
channels. This is a software/equivalence result, not evidence of trained 3-D
scientific quality. The turbulent-combustion dataset remains a 2-D plane stored
as `(x, y, z=0.5)`.

## Verified dimension-generic paths

- Public config/factory carries `coord_dim` into RFF prior, coordinate Fourier
  encoding, sensor/query encoders, GL-RBF core, and CQ decoder.
- Top-K distances and KeOps/pure-Torch neighbor expressions reduce over the
  final coordinate axis; the persistent-cache schema stores shapes/dtype/device
  rather than assuming x/y.
- Sparse-condition construction derives `coord_dim` from `coords_full.shape`.
- Raw measurement/support uses the same dimension-independent Top-K
  indices/distances and adds only field value/support scalars.
- Query microbatch and reconstruction chunking slice the query axis and retain
  all coordinate channels.

`tests/test_cleanup_public_api.py` uses nonconstant x/y/z coordinates and checks
CQ forward, latent widths, persistent geometry construction, and monolithic vs
query-microbatch loss. The full suite also verifies cached/fresh outputs and
zero post-build KNN calls.

## Deliberately 2-D-only paths

- `helpers.validate_regular_grid_compatibility` and the FNO baseline use
  `Num_x × Num_y`, x/y uniqueness, and 2-D grid reshapes.
- `helpers.visualize_reconstruction` and triangulation/scatter utilities plot
  coordinate channels 0 and 1 only.
- The FNO path in `Model.py` explicitly uses `coords[..., :2]`.

These are baseline/visualization constraints, not GL_rbf_CQ model constraints.
They were documented rather than silently generalized because doing so would
change baseline and plotting semantics.

## Remaining 3-D work outside this cleanup

- train/validate on a volumetric dataset with nonconstant z;
- add volumetric visualization/export;
- define 3-D field normalization and scientific metrics; and
- benchmark KeOps/cache memory with production-scale volumetric geometry.

