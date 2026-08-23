# Stage 6 formal-baseline figure contract

Core conclusion: Increasing effective query supervision from 4,096 to 16,384
does not materially improve the current decoder at one training seed and raises
training cost substantially, so F0 remains the formal baseline for Stage 6.

Figure archetype: quantitative grid plus a matched reconstruction image plate.

Target/output: double-column scientific figure, 183 mm wide, with editable SVG
as the primary output and PDF, TIFF, and PNG companions.

Backend: Python only. Model inference produces numeric arrays without plotting;
all figure generation and export runs in the project figure environment.

Panel map:

- Decision figure a: 200-epoch training and matched validation trajectories.
- Decision figure b: paired F1-minus-F0 RF-loss differences across 64 fixed
  validation layouts after averaging three controlled RF draws per layout.
- Decision figure c: per-field matched reconstruction-error differences across
  Euler NFE 1, 2, and 4.
- Decision figure d: normalized epoch-time, training-step, and sampled reserved
  GPU-memory costs.
- Reconstruction plate: truth, F0 best, and F1 best for all five fields under
  the same snapshot, sensors, RF draw, solver, and NFE 1.

Evidence hierarchy:

- Hero evidence: fixed-manifest paired RF loss and its layout-level 95% CI.
- Validation evidence: matched convergence and best-checkpoint reconstruction.
- Cost evidence: epoch time, diagnostic step time, and sampled reserved memory.

Statistics: one training seed per protocol; 64 fixed validation layouts; three
controlled RF draws per layout treated as technical repeats and averaged before
the layout-level paired interval/test. No multi-seed architecture inference is
claimed.

Image integrity: matched reconstruction arrays use one unchanged physical
snapshot and identical sensor/sample seeds. Truth and predictions share the
same per-field color limits. No local contrast changes, cropping, stitching, or
synthetic data are used.

Reviewer risks: the comparison has one training seed, the two best checkpoints
occur at different epochs, the reconstruction plate uses one snapshot, and the
batch size was reduced equally to 64 after F1 could not fit at 144 or 96 on a
48 GiB GPU.

