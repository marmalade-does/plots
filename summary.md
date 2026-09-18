# Optuna study `mae_3d_deepz`

- objective: `value` (maximize)
- trials: 64 COMPLETE 64

## Best trial: #63 — +0.2289

Reproduce it with:

```bash
scripts/train_mae.py --config configs/mae_3D_data_grid_deepz_tune.yaml \
    --override tuning.enabled=false data.batch_size=16 model.dim=768 model.heads=4 model.masking.block_scale=2.317008968884105 model.masking.disk_scale=0.8519428550093127 model.masking.level_allocation=uniform model.masking.spatial_pattern=isotropic_disk model.masking_ratio=0.687622203128051 model.num_layers=9 optim.gradient_clip_val=0.831669639448592 optim.lr=0.0008672087775136818 optim.warmup_epochs=8
```

| param | value |
|---|---|
| `data.batch_size` | 16 |
| `model.dim` | 768 |
| `model.heads` | 4 |
| `model.masking.block_scale` | 2.317008968884105 |
| `model.masking.disk_scale` | 0.8519428550093127 |
| `model.masking.level_allocation` | uniform |
| `model.masking.spatial_pattern` | isotropic_disk |
| `model.masking_ratio` | 0.687622203128051 |
| `model.num_layers` | 9 |
| `optim.gradient_clip_val` | 0.831669639448592 |
| `optim.lr` | 0.0008672087775136818 |
| `optim.warmup_epochs` | 8 |

## Leaderboard (top 10)

| trial | value | batch_size | dim | heads | block_scale | disk_scale | level_allocation | spatial_pattern | masking_ratio | num_layers | gradient_clip_val | lr | warmup_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 63 | +0.2289 | 16 | 768 | 4 | 2.317 | 0.8519 | uniform | isotropic_disk | 0.6876 | 9 | 0.8317 | 0.0008672 | 8 |
| 54 | +0.2246 | 8 | 768 | 2 | 2.391 | 0.9261 | uniform | isotropic_disk | 0.6949 | 10 | 0.108 | 0.0001517 | 5 |
| 28 | +0.2241 | 16 | 512 | 2 | 2.284 | 0.8121 | uniform | isotropic_disk | 0.6839 | 12 | 0.2791 | 0.000502 | 7 |
| 49 | +0.2235 | 16 | 512 | 2 | 2.375 | 0.8708 | uniform | isotropic_disk | 0.6941 | 11 | 0.4971 | 0.0006512 | 6 |
| 31 | +0.2234 | 16 | 512 | 2 | 2.293 | 0.882 | uniform | isotropic_disk | 0.7028 | 12 | 0.2461 | 0.0005093 | 7 |
| 29 | +0.2229 | 16 | 512 | 2 | 2.26 | 0.8038 | uniform | isotropic_disk | 0.7028 | 11 | 0.2314 | 0.0004895 | 7 |
| 50 | +0.2228 | 16 | 768 | 2 | 2.371 | 0.8664 | uniform | isotropic_disk | 0.6919 | 11 | 0.1048 | 0.0006613 | 6 |
| 5 | +0.2219 | 16 | 256 | 4 | 2.23 | 0.8326 | redundancy | isotropic_disk | 0.737 | 10 | 0.8414 | 0.0005656 | 1 |
| 30 | +0.2212 | 16 | 512 | 2 | 2.288 | 0.8851 | uniform | isotropic_disk | 0.6947 | 11 | 0.2298 | 0.0005001 | 7 |
| 58 | +0.2210 | 16 | 768 | 2 | 2.352 | 0.8414 | uniform | isotropic_disk | 0.6804 | 10 | 0.3664 | 0.0006384 | 5 |

## Per-knob breakdown

### `data.batch_size`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 16 | 37 | +0.2152 | -0.1304 | +0.2289 | 34.5 |
| 8 | 15 | +0.2020 | +0.0638 | +0.2246 | 33.0 |
| 32 | 12 | +0.1489 | +0.0433 | +0.2185 | 20.3 |

### `model.dim`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 512 | 26 | +0.2152 | +0.1135 | +0.2241 | 34.3 |
| 768 | 16 | +0.2150 | -0.1304 | +0.2289 | 44.4 |
| 384 | 10 | +0.1787 | +0.0190 | +0.2185 | 18.2 |
| 256 | 12 | +0.1540 | +0.0433 | +0.2219 | 19.2 |

### `model.heads`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 2 | 37 | +0.2157 | -0.1304 | +0.2246 | 40.0 |
| 4 | 15 | +0.1695 | +0.0190 | +0.2289 | 24.9 |
| 8 | 12 | +0.1345 | +0.0638 | +0.2185 | 13.4 |

### `model.masking.block_scale`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 2.284 – 2.482 | 22 | +0.2183 | +0.0190 | +0.2289 | 42.7 |
| 2.023 – 2.281 | 21 | +0.1883 | +0.0433 | +0.2229 | 30.2 |
| 1.013 – 2.004 | 21 | +0.1844 | -0.1304 | +0.2186 | 21.0 |

### `model.masking.disk_scale`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 0.8224 – 0.8691 | 21 | +0.2123 | +0.0638 | +0.2289 | 29.7 |
| 0.8695 – 0.9458 | 22 | +0.2098 | -0.1304 | +0.2246 | 37.8 |
| 0.751 – 0.8206 | 21 | +0.1849 | +0.0433 | +0.2241 | 26.7 |

### `model.masking.level_allocation`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| uniform | 40 | +0.2157 | +0.0190 | +0.2289 | 37.0 |
| redundancy | 15 | +0.1945 | +0.0638 | +0.2219 | 23.9 |
| dirichlet | 9 | +0.1317 | -0.1304 | +0.1604 | 19.8 |

### `model.masking.spatial_pattern`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| isotropic_disk | 40 | +0.2157 | -0.1304 | +0.2289 | 36.9 |
| anisotropic_disk | 8 | +0.2010 | +0.1537 | +0.2152 | 32.0 |
| anisotropic_block | 7 | +0.1755 | +0.0190 | +0.2020 | 20.0 |
| iid | 9 | +0.1604 | +0.1317 | +0.2185 | 16.2 |

### `model.masking_ratio`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 0.6804 – 0.7028 | 21 | +0.2192 | +0.1187 | +0.2289 | 43.0 |
| 0.7051 – 0.737 | 21 | +0.2137 | +0.1135 | +0.2219 | 31.0 |
| 0.7372 – 0.8428 | 22 | +0.1551 | -0.1304 | +0.2185 | 21.0 |

### `model.num_layers`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 10 – 12 | 21 | +0.2181 | +0.1679 | +0.2246 | 41.2 |
| 12 – 16 | 22 | +0.2093 | +0.0433 | +0.2186 | 31.3 |
| 6 – 10 | 21 | +0.1627 | -0.1304 | +0.2289 | 22.0 |

### `optim.gradient_clip_val`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 0.2461 – 0.4736 | 21 | +0.2157 | +0.0638 | +0.2241 | 33.3 |
| 0.1005 – 0.2443 | 21 | +0.2100 | +0.1334 | +0.2246 | 39.5 |
| 0.4971 – 0.9883 | 22 | +0.1615 | -0.1304 | +0.2289 | 22.2 |

### `optim.lr`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 0.000256 – 0.0005068 | 21 | +0.2158 | +0.1187 | +0.2241 | 34.6 |
| 0.0005093 – 0.000926 | 22 | +0.2130 | -0.1304 | +0.2289 | 37.2 |
| 1.226e-05 – 0.0002493 | 21 | +0.1537 | +0.0190 | +0.2246 | 22.4 |

### `optim.warmup_epochs`

| value | n | median | min | max | mean trial# |
|---|---|---|---|---|---|
| 5 – 7 | 21 | +0.2182 | +0.1355 | +0.2246 | 37.7 |
| 7 – 10 | 22 | +0.2097 | +0.0638 | +0.2289 | 32.3 |
| 1 – 5 | 21 | +0.1566 | -0.1304 | +0.2219 | 24.5 |

## Parameter importance (fANOVA)

| param | share |
|---|---|
| `model.masking.level_allocation` | 25.4% |
| `model.masking_ratio` | 18.8% |
| `optim.lr` | 12.2% |
| `optim.gradient_clip_val` | 11.9% |
| `model.heads` | 10.3% |
| `model.masking.block_scale` | 6.5% |
| `optim.warmup_epochs` | 4.0% |
| `model.masking.disk_scale` | 3.8% |
| `model.masking.spatial_pattern` | 2.1% |
| `model.dim` | 2.0% |
| `model.num_layers` | 1.5% |
| `data.batch_size` | 1.5% |

