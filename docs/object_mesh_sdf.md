# Mesh + SDF object geometry

The object geometry wrapper now owns both the original canonical mesh and a
mesh-backed signed distance field. SDF values are negative inside, positive
outside, in the same units as the queried points. This is not an output filter:
the field is queried during diffusion guidance and in optional training losses.

The denoiser still predicts the existing 9-dimensional rigid object pose.
PointNeXt conditioning and checkpoint tensor shapes are unchanged. No SDF
feature encoder or new object-shape generation network is introduced.

## Enable

Overlay `scenarios/object_mesh_sdf.yaml` after the base scenario:

```bash
python main.py -c config/env.yaml scenarios/gb_main.yaml scenarios/object_mesh_sdf.yaml -- \
  run.name=mesh_sdf_sample run.job=sample \
  resume.checkpoint=/home/yanghc/tridi/assets/gb_main.pth \
  env.datasets_folder=/home/yanghc/tridi/data/preprocessed \
  env.smpl_folder=/home/yanghc/tridi/data/smplx_models \
  env.assets_folder=/home/yanghc/tridi/assets \
  dataloader.batch_size=8 sample.dataset=normal sample.mode=100 sample.contacts_mode=heatmap
```

Use an existing full dataset containing all configured objects. Restricting class
lists has pre-existing assumptions in the legacy mesh/keypoint loader. The normal
sampler avoids needing a CLIP download. The sampling entry point uses CUDA.

The overlay enables a 64-cubed field and guidance after diffusion timestep 200.
`model.cg_scale=0.01` and loss weights are starting values, not tuned results.
The SDF correction supports `ddpm_guided`, `ddpm`, and `ddim` through the existing
predicted-clean-sample interface. Conditioned variables are kept fixed and the
SDF guidance does not change body shape parameters to escape collision.

For training, the same overlay merges these optional losses into the base losses:

- `obj_sdf_penetration`: squared negative SDF over all human vertices.
- `obj_sdf_contact`: squared SDF only at ground-truth sampled human vertices
  within 5 mm of the object surface. It does not pull all vertices to the object.

Both losses apply only when both human and object diffusion timesteps are below
`train.loss_t_stamp_threshold`. They backpropagate through human geometry and
the object rigid pose to the denoiser. Empty eligible batches/contact sets return
a differentiable zero. Training uses `batch.scale` consistently for predicted
vertices and distances; ground-truth contact selection uses unscaled geometry.
Use a new run name and resume weights without the old optimizer/scheduler/state
when starting a new fine-tuning experiment. No full training is launched by the
smoke test below.

Set `model.object_sdf_resolution=0` and `model.cg_apply=false`, and remove the two
SDF training losses, to use the original behavior. With SDF guidance disabled,
the same weights and random seed produce identical samples in the smoke test.

## Geometry and limitations

Each field is lazily built with the existing libigl dependency and cached below
`env.datasets_folder/object_sdf/`. Cache keys include canonical mesh content,
resolution, padding and sign mode. GPU/CPU trilinear interpolation preserves
gradients through points, object translation and rotation. Queries transform
points into canonical object coordinates before sampling, with the same rigid
transform and common scale as the mesh wrapper.

Grid bounds use per-axis padding so thin axes retain their sampling resolution.
Outside the grid, a positive distance extension supplies a nonzero gradient;
this extension is an approximation, not an exact mesh distance. Fine structures
smaller than grid spacing can still be missed. SDF does not by itself guarantee
a stable grasp, prevent hand self-intersection, or enforce temporal consistency.

Strict mode (`model.object_sdf_allow_open=false`, the default) requires closed,
consistently oriented volumes after merging duplicate vertices and fixing face
orientation on a copy. It does not fill holes or convexify the source geometry.
The server's existing 41 canonical meshes included 21 non-closed meshes after
this cleanup (18 BEHAVE and GRAB camera/headphones/phone). Therefore the example
overlay **explicitly** enables `object_sdf_allow_open=true`: open meshes use
generalized winding-number signs and emit a warning. Those signs are approximate,
especially near holes; repair the meshes or use strict mode for reliable signed
collision evaluation. Rendering/export continues to use the original mesh.

## Checks

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_object_sdf.py -v
PYTHONPATH=. python tests/smoke_object_sdf.py --source /home/yanghc/tridi
```

The unit tests cover signed values, gradients, rigid transforms, common scale,
mixed object batches, caching, open meshes, thin geometry and in-loop guidance.
The CPU smoke test strictly loads the official checkpoint and contact encoder/
decoder, uses SMPL+H and a real GRAB frame, exercises generated/conditioned
partitions with DDIM and guided DDPM, and runs an SDF-loss backward pass through
the denoiser. For a guaranteed nonzero collision gradient the backward check
explicitly moves the object into one predicted vertex; this is a controlled
test, not a naturally occurring failure or a reported quality improvement.
Reports are written to `outputs/object_mesh_sdf_smoke.json`.

Server validation (2026-10-22): 21 unit tests passed. The real frame
`s9/banana_eat_1/4` ran all six combinations of generation modes `100`, `010`,
`111` with DDIM and guided DDPM. Every guided run queried SDF twice; all
conditioned partitions stayed unchanged and outputs remained finite. Disabled
SDF runs matched the original path bit-for-bit. The `100` runs had no correction
on this frame, while `010`/`111` produced nonzero corrections. The controlled
training collision produced finite nonzero denoiser gradients; real-frame GT
contact loss was zero (no sampled GT vertex within 5 mm). Unit tests separately
exercise nonzero contact gradients. An open GRAB camera mesh was also queried
successfully with the explicit approximate-sign mode and finite gradients.
These checks establish implementation behavior, not a benchmark improvement;
full training and hand-contact quality evaluation remain to be done.

## Bounded arm/hand refinement

`scenarios/object_mesh_sdf_hand.yaml` enables the optional second version. Load
it after `scenarios/gb_main.yaml` and `scenarios/object_mesh_sdf.yaml`:

```bash
cd /home/yanghc/tridi-object-mesh-sdf
/home/yanghc/miniconda3/envs/tridi-offline/bin/python main.py \
  -c config/env.yaml scenarios/gb_main.yaml scenarios/object_mesh_sdf.yaml scenarios/object_mesh_sdf_hand.yaml -- \
  run.name=mesh_sdf_hand_sample run.job=sample \
  resume.checkpoint=/home/yanghc/tridi/assets/gb_main.pth \
  env.datasets_folder=/home/yanghc/tridi/data/preprocessed \
  env.smpl_folder=/home/yanghc/tridi/data/smplx_models \
  env.assets_folder=/home/yanghc/tridi/assets \
  dataloader.batch_size=8 sample.dataset=normal sample.mode=100 sample.contacts_mode=heatmap
```

No retraining or new weights are needed. The sampler now forwards the configured
scheduler and inference-step count; the example uses 100 guided-DDPM steps.

- In **heatmap-conditioned** sampling, the sampler explicitly stores the input
  binary 5 cm heatmap in `batch.sbj_contact_target`. Geometry refinement can use
  these original target bits instead of recovering them from a lossy latent.
  This is additional retained input information compared with the first version;
  its benefit must not be attributed to SDF alone. No ground-truth pose is read
  by the refiner. A custom caller must provide this field only for a declared
  input heatmap, not derive it from withheld evaluation annotations.
- CLIP-conditioned sampling leaves that field unset. Generated-contact modes
  ignore it. Both use the existing decoder, so missing decoded hand contacts
  remain a limitation for those modes.
- Contact loss is normalized by active hand targets. Hand and non-hand collision
  losses are separately normalized by active penetrating points, avoiding dilution
  of a few finger collisions by all 6890 body vertices. Surface tolerance is 2 mm;
  collision tolerance is 1 mm. These tolerances are in the canonical metre units
  used by the existing sampling path.
- Each guided clean prediction takes five gradient steps, with per-coordinate
  changes bounded by `sdf_refine_max_delta=0.1`. Only parameters `106:322`
  (shoulders through fingers) are directly changed. Shape, root orientation,
  torso/legs, root translation, object pose and contacts are directly frozen.
  Diffusion coupling can still change their final generated values indirectly;
  evaluate final drift rather than claiming exact equality to unguided output.
- Per sample, only an improved candidate whose contact loss, hand/non-hand
  collision loss, fixed-denominator total collision loss and maximum depth do
  not worsen is returned. The unchanged clean prediction is the fallback. These
  checks use interpolated SDF values, not exact physical guarantees: open meshes,
  sub-voxel details, self-intersection and grasp stability still require separate
  evaluation. In object-only generation the original SDF guidance remains active.

The original path remains available with `model.sdf_hand_refinement=false`.
This option changes sampling only; existing SDF training losses are unchanged.
Tests are in `tests/test_sdf_refinement.py`, including empty targets, source
isolation, bounded updates, collision regression and sampler settings.

### Second-version validation (2026-10-28)

**Historical results with inconsistent point inputs:** the later audit below
found that server-generated NPZ keypoints covered only small surface patches.
These numbers preserve that experiment; they do not establish performance with
the bundled canonical inputs or successful grasps.

Fresh sequence pairs excluded all sequences used in the first evaluation.
Calibration used 12 conditions and selected step scale 2 with penetration weight
10. Test used 33 conditions, 19 object types and two new noise seeds (6792/6793),
100 guided-DDPM steps, the same frozen official weights, and four variants.
There were 264 generated samples; no finetuning was performed.

Closed-volume analysis covers 29 conditions / 16 object types. Exact original
triangle queries, independent of the SDF interpolation grid, gave these means:

| Variant | Hand penetration mm | Full-body penetration mm | GT20mm target gap mm |
| --- | ---: | ---: | ---: |
| Old SDF, scale 2 | 0.1141 | 0.1878 | 53.6344 |
| Bounded refinement, decoded targets | 0.0328 | 0.1685 | 49.8110 |
| Bounded refinement, original binary heatmap | 0.0575 | 0.1726 | 40.1132 |

Decoded-target refinement reduced hand penetration relative to old SDF in 18/29
conditions, worsened 6 and tied 5; the object-cluster bootstrap interval for mean
change was [-0.1574, -0.0199] mm. Its gap change interval crossed zero. Retaining
the original heatmap reduced gap in 19/29 conditions and worsened 10, with mean
change interval [-24.8325, -3.1681] mm; its hand-penetration change interval crossed
zero. Input-heatmap 5 mm touch rate did not improve over old SDF (8.05% vs 8.57%).
These are partial aggregate improvements, not guarantees for each grasp.

Per-step local updates produced final root drift of 0.18/0.23 mm relative to
unguided generation, versus 26.51 mm for old SDF. The same previously explored
test subjects s9/s10 were used, so these are fresh sequences, not blind subject
generalization. Open meshes, self-collision, force/friction and temporal stability
remain unverified limitations. Thirty-two tests and independent recomputation of
12 stored metric rows passed; all outputs and fixed conditions were verified.

Server artifacts and reproducible scripts:
`outputs/sdf-hand-20261028-r2/`, especially `report_v2.md`, `summary_v2.json`,
`verification.json` and `scripts/`. Earlier pilot validation is preserved in
`outputs/sdf-hand-20261028/`; it predates the added total-penetration safeguard.

### Canonical point input repair and failure audit (2026-10-28)

The user-reported flute, mug and doorknob examples were contact failures: their
right hands had no vertices within 5 mm of the exact object surface. The server's
preprocessed flute points covered only 13 mm of its 210 mm length; mug points
covered only its bottom face, and doorknob points only a base corner. Bundled
`assets/object_keypoints/{grab,behave}.pkl` references cover the complete shapes.
For these three objects, reference point-to-mesh median errors were at most
0.038 mm and maximum errors at most 0.425 mm, consistent with their mesh frames.

Both normal and random dataset loaders now prefer these bundled references and
warn if regenerated NPZ points differ. Shared preprocessed data are never
overwritten. A custom dataset without a bundled reference still loads its NPZ
files; an incomplete or malformed present reference fails clearly. PointNeXt
features and learned weights are unchanged.

The regression reran all 33 previous conditions, seeds 6792/6793, batch size 8,
100 steps and all four variants, without retuning (264 generations). These are
reused diagnostic cases, not a new held-out evaluation. Physical input parameters
and GT geometry remained identical; only canonical point inputs and their supplied
5 cm contact heatmaps/latents changed. Input repair must be assessed separately
from SDF guidance. Original assets, NPZ files, source and checkpoint hashes were
verified unchanged during the run. All 42 unit tests passed.

For the displayed first-seed right hands, nearest exact surface gaps changed:
flute 22.18 -> 0.04 mm, mug 28.44 -> 9.30 mm, doorknob 23.83 -> 102.03 mm.
Flute now has contact but differs strongly in hand orientation from the reference;
this alone does not establish a correct playing grasp. Mug and doorknob still
have zero near-surface hand vertices. SDF distance/collision objectives do not
encode anatomical palm contact, task-appropriate wrist orientation or a grasp
prior. This repair does not solve those grasp failures.

Artifacts: `outputs/sdf-correct-points-20261028/`, including
`report_input_repair.md`, `input_repair_verification.json`, `test/` and `visual/`.
