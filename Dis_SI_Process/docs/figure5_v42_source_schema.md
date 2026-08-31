# Figure 5 V4.2 source contract

V4.2 is additive and does not overwrite V4 or V4.1. Panels a, b, c, and e retain the exact V4.1 evidence and visual grammar. Panel d restores canonical training update time (`ms/update`) as its x coordinate and uses logarithmic x and y axes.

The six single-stage coordinates from `training_replay_formal_v4r2` must remain bit-for-bit unchanged. Latent FM remains unavailable because its two required, architecturally different stages do not define a single update-time scalar. Geo-FNO is added only from `geofno_ddp_timing_formal_v42r2`: two clean physical GPUs, DDP, global batch 192 (96 per rank), 20 warmups, 10 blocks × 10 measured updates, synchronized max-rank wall time, and a preloaded training batch.

The timed boundary includes device-side conditioning/grid preparation, forward pass, loss, backward pass, DDP gradient communication, gradient clipping, and optimizer step. It excludes dataset I/O, data-loader work, host transfer, validation, logging, and checkpointing. GPU-ms and peak memory are provenance fields, not panel-d x coordinates.
