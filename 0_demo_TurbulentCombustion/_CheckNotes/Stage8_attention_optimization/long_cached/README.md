# GL_rbf_CQ cached/full long validation

The local training case completed epoch 650 and was intentionally stopped. Its
formal matched endpoint is epoch 600. The versioned release evidence is in
`evaluation_0600/`; large research checkpoints and reconstruction PNGs remain
local and are not release-source artifacts.

Final execution selection:

```yaml
condition_attention_execution: cached_kv
sensor_attention_padding_mode: full
```

`legacy_mha + full` remains supported for compatibility and debugging. Static
bucketing and dynamic trimming are not promoted. The Stage-7 epoch-1000
EMA-resolved portable checkpoint remains the release model artifact.
