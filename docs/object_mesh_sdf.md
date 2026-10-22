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
