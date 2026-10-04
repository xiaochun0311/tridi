# Small-batch pilot: three-direction TriDi prototype

Run on the server's RTX 3090 Ti with GRAB training data only. Starting checkpoint: `assets/gb_main.pth`. The successful pilot used batch size 4, 30 optimizer steps, a fresh AdamW/cosine schedule (no warmup), and saved `experiments/045_tri3_pilot_20261004b/checkpoints/checkpoint-step-0000030.pth`. The first attempt stopped with a NaN loss. A small-batch guard was added for geometry losses when no sample passes the low-noise mask; the retry finished, but the exact cause of the first NaN was not independently established.

Evaluation used the first test frame of banana, binoculars, camera, and cup; `sample.mode=101`, 100 denoising steps, and paired seeds 1000–1003. Both checkpoints used the same original contact guidance (`cg_apply=true`, `cg_scale=2`). `landing_only` uses the original checkpoint plus the new constraint step. Trained variants use the 30-step checkpoint; `trained_projection` additionally enables manifold projection strength 0.05.

| Variant | Mean predicted contact points | Mean distance of predicted contacts (mm) | Empty contact masks | Mean nearby vertices | Mean penetrating nearby vertices | Mean penetration-depth sum (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 16.75 | 44.05 | 0/4 | 14.50 | 4.00 | 0.0673 |
| Landing only | 16.75 | 43.88 | 0/4 | 14.25 | 5.00 | 0.0660 |
| Trained, no landing or projection | 4.75 | 66.62 | 1/4 | 3.50 | 2.00 | 0.0188 |
| Trained, landing, no projection | 4.75 | 66.19 | 1/4 | 3.50 | 2.00 | 0.0189 |
| Trained, landing and projection | 13.75 | 46.18 | 1/4 | 10.75 | 3.25 | 0.0315 |

The 30-step checkpoint does **not** show an overall interaction improvement: fewer apparent penetrations often coincide with lost contact or hands farther from the object. The contact-distance mean excludes the empty-mask case, so it must be read with the mask count. The penetration metric is a nearest-sampled-point/normal proxy within 5 cm, not an exact mesh-intersection test. Four frames and 30 steps are only a smoke/pilot experiment; no statistical or publication claim is supported.

Raw measurements are on the server at `experiments/045_tri3_pilot_20261004b/pilot_metrics.json` and `pilot_ablations.json`. Next useful experiment: train longer with contact-mask preservation monitored, then evaluate more subjects/objects and exact mesh penetration before tuning landing or manifold projection.
