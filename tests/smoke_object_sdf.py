"""CPU smoke test with existing TriDi weights, SMPL+H and one real GRAB frame.

Run from repository root: PYTHONPATH=. python tests/smoke_object_sdf.py --source /home/yanghc/tridi
This does not update a checkpoint or run a training experiment.
"""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf

from config.config import ProjectConfig
from tridi.core.trainer import Trainer
from tridi.data.batch_data import BatchData
from tridi.model import get_model
from tridi.model.conditioning.contact_ae_clip import ContactEnc, ContactDec
from tridi.model.wrappers.contact import ContactModel
from tridi.model.wrappers.mesh import MeshModel
from tridi.utils.geometry import matrix_to_rotation_6d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/object_mesh_sdf_smoke.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = OmegaConf.merge(OmegaConf.structured(ProjectConfig()), OmegaConf.load('scenarios/gb_main.yaml'))
    cfg.model.object_sdf_resolution = 32
    cfg.model.cg_scale = 0.01
    model = get_model(cfg).cpu().eval()
    checkpoint = torch.load(args.source / 'assets/gb_main.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    contact_checkpoint = torch.load(args.source / 'assets/gb_contacts.pth', map_location='cpu', weights_only=False)
    encoder, decoder = ContactEnc(690, 256, 128), ContactDec(690, 256, 128)
    encoder.load_state_dict(contact_checkpoint['model_enc'])
    decoder.load_state_dict(contact_checkpoint['model_dec'])
    encoder.eval()
    decoder.eval()
    contact = ContactModel('encoder_decimated_clip', device='cpu')
    contact.contact_dec = decoder
    model.set_contact_model(contact)

    root = args.source / 'data/preprocessed/grab_smplh'
    object_id = int(cfg.grab.obj2classid['banana'])
    meshes, keypoints = {}, {}
    for name, class_id in cfg.grab.obj2classid.items():
        meshes[class_id] = trimesh.load(root / 'object_meshes' / f'{name}.ply', process=False)
        keypoints[class_id] = dict(np.load(root / 'object_keypoints' / f'{name}.npz'))
    # Existing mesh wrapper indexes its legacy keypoint stack by global class id.
    for name, class_id in cfg.behave.obj2classid.items():
        behave = args.source / 'data/preprocessed/behave_smplh'
        meshes[class_id] = trimesh.load(behave / 'object_meshes' / f'{name}.ply', process=False)
        keypoints[class_id] = dict(np.load(behave / 'object_keypoints' / f'{name}.npz'))
    mesh = MeshModel(str(args.source / 'data/smplx_models'), 1, meshes, keypoints,
                     object_sdf_resolution=32, object_sdf_cache_dir=Path('outputs/object_sdf_cache'))
    model.set_mesh_model(mesh)
    indexes = torch.from_numpy(np.load(args.source / 'assets/smpl_template_decimated_idxs.npy')).long()
    with h5py.File(root / 'dataset_test_1fps.hdf5') as data:
        subject = 's9'
        name = next(name for name in sorted(data[subject]) if data[subject][name].attrs['obj'] == 'banana')
        seq = data[subject][name]
        frame = int(seq.attrs['T']) // 2
        rotations = np.concatenate([seq['sbj_smpl_global'][frame].reshape(1, 3, 3),
                                    seq['sbj_smpl_body'][frame].reshape(21, 3, 3),
                                    seq['sbj_smpl_lh'][frame].reshape(15, 3, 3),
                                    seq['sbj_smpl_rh'][frame].reshape(15, 3, 3)])
        rotations = matrix_to_rotation_6d(torch.tensor(rotations)).float()
        batch = BatchData(
            sbj=[subject], sbj_shape=torch.tensor(seq['sbj_smpl_betas'][frame])[None].float(),
            sbj_pose=rotations[1:].reshape(1, -1), sbj_global=rotations[:1],
            sbj_c=torch.tensor(seq['sbj_smpl_transl'][frame])[None].float(),
            obj_R=matrix_to_rotation_6d(torch.tensor(seq['obj_R'][frame].reshape(1, 3, 3))).float(),
            obj_c=torch.tensor(seq['obj_c'][frame])[None].float(),
            obj_class=torch.tensor([object_id]), obj_group=torch.tensor([cfg.grab.obj2groupid['banana']]),
            scale=torch.tensor([float(seq['prep_s'][frame])]), sbj_contact_indexes=indexes[None],
            sbj_gender=torch.tensor([seq.attrs['gender'] == 'male']),
        )
    with open(root / 'object_pointnext.pkl', 'rb') as fp:
        batch.obj_pointnext = torch.tensor(pickle.load(fp)['banana']).reshape(1, -1).float()
    with torch.no_grad():
        batch.sbj_vertices, batch.sbj_joints = mesh.get_smpl_th(batch)
        obj_points = mesh.canonical_obj_keypoints_th[batch.obj_class]
        from tridi.utils.geometry import rotation_6d_to_matrix
        batch.obj_keypoints = obj_points @ rotation_6d_to_matrix(batch.obj_R).transpose(1, 2) + batch.obj_c[:, None]
        distances = torch.cdist(batch.sbj_vertices[:, indexes], batch.obj_keypoints).min(-1).values
        batch.sbj_contacts = encoder((distances < 0.05).float())
        batch.sbj_contacts_full = distances

    report = {'frame': f'{subject}/{name}/{frame}', 'checkpoint_strict_load': True}
    model.requires_grad_(False)
    for mode in ('100', '010', '111'):
        for scheduler in ('ddim', 'ddpm_guided'):
            model.cg_apply = False
            model.object_sdf_resolution = 0
            torch.manual_seed(6790)
            baseline = model.forward_sample(mode, batch, scheduler=scheduler, num_inference_steps=6, disable_tqdm=True)
            model.object_sdf_resolution = 32
            torch.manual_seed(6790)
            disabled = model.forward_sample(mode, batch, scheduler=scheduler, num_inference_steps=6, disable_tqdm=True)
            assert torch.equal(baseline, disabled), 'SDF disabled changed the baseline'
            model.cg_apply = True
            from unittest.mock import patch
            torch.manual_seed(6790)
            with patch.object(mesh, 'query_object_sdf', wraps=mesh.query_object_sdf) as query:
                guided = model.forward_sample(mode, batch, scheduler=scheduler, num_inference_steps=6, disable_tqdm=True)
                assert query.call_count > 0, 'Sampling did not query the object SDF'
                query_count = query.call_count
            assert torch.isfinite(guided).all()
            if mode[0] == '0':
                assert torch.equal(guided[:, :325], model.merge_input_sbj(batch))
            if mode[1] == '0':
                assert torch.equal(guided[:, 325:334], model.merge_input_obj(batch))
            if mode[2] == '0':
                assert torch.equal(guided[:, 334:], batch.sbj_contacts)
            report[f'{mode}_{scheduler}'] = dict(sdf_queries=query_count,
                max_guidance_change=float((guided - baseline).abs().max()),
                finite=True, conditions_preserved=True, disabled_matches_baseline=True)
    assert all(p.grad is None for p in decoder.parameters())

    # A real forward/backward through the denoiser and the new training geometry losses.
    model.requires_grad_(True)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = SimpleNamespace(train=SimpleNamespace(loss_t_stamp_threshold=250,
        losses={'obj_sdf_penetration': 20.0, 'obj_sdf_contact': 20.0}))
    trainer.model, trainer.mesh_model = model, mesh
    torch.manual_seed(42)
    losses, aux = model(batch, 'train', return_intermediate_steps=True)
    output = model.split_output(aux[3], aux)
    output.sbj_vertices, _ = mesh.get_meshes_wkpts_th(
        output, batch.obj_class, scale=batch.scale, sbj_gender=batch.sbj_gender
    )
    # Inject one controlled collision to require a nonzero geometry gradient,
    # even when this real frame has no GT contact within 5 mm.
    grid, lower, upper = mesh.object_sdf._field(object_id)
    z, y, x = np.unravel_index(int(grid.argmin()), grid.shape[-3:])
    interior = lower + torch.tensor([x, y, z]) / (grid.shape[-1] - 1) * (upper - lower)
    anchor = output.sbj_vertices[:, indexes[0]].detach() / batch.scale[:, None]
    target_center = anchor - interior[None, None] @ rotation_6d_to_matrix(output.obj_R.detach()).transpose(1, 2)
    output.obj_c = output.obj_c + (target_center.reshape(1, 3) - output.obj_c).detach()
    output.timesteps_sbj = output.timesteps_obj = torch.zeros(1, dtype=torch.long)
    loss, logged = trainer.compute_loss(batch, output, losses)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    report['training_sdf_losses'] = logged
    report['training_controlled_collision'] = 'object moved to overlap one predicted vertex; gradient check only'
    report['training_backward'] = 'finite nonzero denoiser gradients'
    report['cached_objects'] = sorted(mesh.object_sdf.fields)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
