# Contact loss repair and checks (2026-10-11)

Branch: codex/debugger.

- Contact geometry loss now checks both subject and object timesteps <= 250.
- Empty geometric selections produce differentiable zero losses instead of NaN.
- Training and validation losses are recorded in local logs even with W&B disabled.
- Regression check: PYTHONPATH=. python tests/test_contact_loss.py (2 tests passed).
- Real-frame check: PYTHONPATH=. python scripts/check_stool_reconstruction.py --checkpoint <pth> --output <folder>.
  Uses two held-out BEHAVE stool-sitting frames, seed 43 and identical noise for each model.
  This is a one-step denoising probe at t=0/100/250, not a full reconstruction benchmark.
  Rotation round-trip maximum error: 1.788e-7.
- Resumed original step 123000 for 1000 updates; finite logged losses and gradients.
  New checkpoint: experiments/071_debug_contactloss_fix_124000/checkpoints/checkpoint-step-0124000.pth.
  Original checkpoint and original training log preserved. New experiment config uses its own output path.
- Seed43, 2 samples, 1000 DDPM steps:
  old stool contact count 0; new stool contact count 47.
  New samples: experiments/072_debug_fixed124000_seed43.
  New pose resembles sitting, but foot support is still unnatural.
  Continuing training and code repair happened together, so this does not isolate the repair's causal effect.
- Two-frame mean subject vertex error (cm): old vs repaired/short-trained
  t=0: 2.96 vs 2.31; t=100: 9.54 vs 11.09; t=250: 17.64 vs 18.70.
  These results do not establish general quality improvement or convergence.
- Added Trae profiles for fixed124000 sampling, real-frame checks and resuming training.
  No long training is currently running. No self-collision constraint was added.
