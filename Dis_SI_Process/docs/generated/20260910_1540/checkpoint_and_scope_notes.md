# Checkpoints and comparison scope

Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.

Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.

| display_label | policy | checkpoint_epoch | global_step | parameter_elements_with_optimizer_state | scheduler_t_max | checkpoint_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| Full model | last | 7520 | 533920 | 6507151 | 10000 | 71a9e1004010558f7137f4140eb976d51d367d39e9bc3473334912257cbaa541 |
| No sensor feedback | last | 6000 | 426000 | 5716879 | 10000 | 5037246ef415d5d0b78d253b5a4ecdde76b549f2436541f32dd5d1aa54e401cf |
| No local conditioning | last | 6000 | 426000 | 5650572 | 10000 | 15de1f603be5580e51379675296dfb218e7a34f983272f081abd442510683cda |
| Local-only conditioning | last | 6000 | 426000 | 895118 | 10000 | 0e6184a6b9c5634d79005529744f98ebc0f3d016a084ec1133987b987c1ea2a3 |
| IID Gaussian prior | last | 7310 | 519010 | 6507151 | 10000 | 21a57fbc86dc7eef7b498c8a1110b1ac9c1a679f73dd5f344f8d446c4de454a9 |
| Deterministic regression | last | 7180 | 509780 | 6507151 | 10000 | f5fb90759ae30eebcfcaebac7f12ce38d52b4e85bac13adde1b864bec3314ecc |
| Full model | best | 7095 | 503745 | 6507151 | 10000 | 634b839b1be4abdb976fe848524bb81708254df65b1e8cd29b3111a9f63ec654 |
| No sensor feedback | best | 5365 | 380915 | 5716879 | 10000 | d311ef1f17aa9ad7bba72ace94560402743e02644a0ab91deb466587b658fb3e |
| No local conditioning | best | 5615 | 398665 | 5650572 | 10000 | 7db5a8b4c7afd11ec9954b7f2891f81b83b6f18b4134596a4dca37fd9817abf1 |
| Local-only conditioning | best | 5300 | 376300 | 895118 | 10000 | b8549181d245b0e0d8ddf57aac4bb9a0246aebe1b0523bb265b1491b00976d0f |
| IID Gaussian prior | best | 7155 | 508005 | 6507151 | 10000 | 7e50d6260677b8f725f6354f8d24c481877f49e7a813ff6660554fa7c275c1ae |
| Deterministic regression | best | 7140 | 506940 | 6507151 | 10000 | 73275866949b4975719afb9545e005746db87890d015b2fad47c4202479895dd |

```yaml
deterministic_objective_control_retained_in_si: true
generative_draws_per_ablation_state: 1
independent_untouched_test_set: false
matched_backbone_initialization_verified: false
matched_evaluation_states: true
matched_measurements: true
matched_training_budget: false
multiple_training_seeds: false
new_ablation_reference_separate_from_benchmark: true
primary_checkpoint_policy: last
sensitivity_checkpoint_policy: best
type: saved_checkpoint_comparison
```
