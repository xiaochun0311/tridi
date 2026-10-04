# Three-direction TriDi prototype

This branch contains three **experimental adaptations**, not reproductions of the cited methods:

- [MaMi-HOI](https://proceedings.mlr.press/v306/wang26id.html): 32 canonical object surface samples are attended to after each denoising Transformer layer. A second residual attention update lets the body tokens respond to object/contact tokens. Both gates start at zero, so an old checkpoint initially retains its denoiser behavior.
- [Landing for constrained diffusion](https://proceedings.mlr.press/v306/jeon26d.html): after a reverse diffusion step, one clipped gradient step reduces contact distance and the existing normal-based nonpenetration penalty. This is a first-order landing-inspired correction, **not** the paper's constrained SDE or feasibility guarantee.
- Joint manifold: a bottleneck autoencoder learns from concatenated SMPL+H parameters and the 128-D contact code. Reconstruction and projection losses train it jointly with TriDi. Optional late-step projection moves sampled human/contact values toward this learned space. This is a simple learned manifold baseline, not a validated topology model.

The old configuration remains unchanged. `scenarios/gb_three_directions.yaml` enables geometry training, the two manifold losses, and a small landing step; manifold projection stays off until a newly trained checkpoint exists.

On the server, start a short continuation from an existing checkpoint:

```bash
cd /home/yanghc/tridi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled \
taskset -c 16-23 python main.py \
  -c config/env.yaml scenarios/gb_main.yaml scenarios/gb_three_directions.yaml -- \
  run.job=train run.name=three_directions_smoke \
  resume.checkpoint=./assets/gb_main.pth resume.training=false \
  dataloader.batch_size=2 dataloader.workers=0 train.max_steps=100
```

After training, sample from the new checkpoint with `model.manifold_projection_strength=0.05`. Compare equal seeds/checkpoints and report contact distance, penetrated vertices and depth, joint quality, and diversity for at least GRAB and BEHAVE. Ablate `model_denoising.params.use_geometry`, `model.landing_step`, and the manifold losses/projection separately. The old checkpoint can load with new modules, but its geometry adapter and autoencoder have not been trained; quality claims require a new checkpoint and a dataset-level evaluation.
