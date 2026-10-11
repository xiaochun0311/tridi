"""Check two real held-out stool frames; report parameter round-trip and one-step denoising."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tridi.data import get_eval_dataloader
from tridi.data.batch_data import BatchData
from tridi.model import get_model
from tridi.model.wrappers.contact import ContactModel
from tridi.model.wrappers.mesh import MeshModel
from tridi.utils.training import set_seed
from tridi.utils.geometry import rotation_6d_to_matrix

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
cfg = OmegaConf.load('experiments/018_gb_train_20260926_183823/config.yaml')
cfg.run.datasets = ['behave']
cfg.run.job = 'sample'
cfg.dataloader.batch_size = 2
cfg.dataloader.workers = 0
cfg.model.cg_apply = False
set_seed(43)
loaders, meshes, keypoints = get_eval_dataloader(cfg)
ds = loaders[0].dataset
indices = [i for i, s in enumerate(ds.data) if s.object == 'stool']
assert len(indices) >= 2, 'Need at least two held-out stool frames'
chosen = [indices[len(indices)//3], indices[2*len(indices)//3]]
batch = BatchData.collate([ds[i] for i in chosen]).cuda()
mesh = MeshModel(cfg.env.smpl_folder, 2, meshes, keypoints, 'cuda')
contact = ContactModel(cfg.model_conditioning.use_contacts, device='cuda')
contact.set_contact_model(cfg.model_conditioning.use_contacts, 'model_enc', 'assets/gb_contacts.pth')
contact.set_contact_model(cfg.model_conditioning.use_contacts, 'model_dec', 'assets/gb_contacts.pth')
outdir = Path(args.output)
outdir.mkdir(parents=True, exist_ok=True)
report = {'checkpoint': args.checkpoint, 'frames': [str(ds.data[i]) for i in chosen], 'results': []}
with torch.no_grad():
    gt, joints = mesh.get_smpl_th(batch)
    batch.sbj_vertices, batch.sbj_joints = gt, joints
    batch.sbj_contacts, batch.sbj_contacts_full, _ = contact.encode_contacts(
        gt, batch.sbj_contact_indexes, batch.obj_keypoints, batch.obj_class, batch.obj_R, enc_type='model_enc')
    report['gt_contact_vertices'] = (batch.sbj_contacts_full < .05).sum(-1).tolist()
    rotations = torch.stack([torch.tensor(ds.get_sequence(ds.data[i].subject, 'stool', ds.data[i].action)['sbj_smpl_body'][ds.data[i].t_stamp], device='cuda') for i in chosen]).reshape(2, 21, 3, 3)
    reconstructed_rotations = rotation_6d_to_matrix(batch.sbj_pose.reshape(2, 51, 6))[:, :21]
    report['rotation_roundtrip_max_abs'] = float((rotations - reconstructed_rotations).abs().max())
    assert report['rotation_roundtrip_max_abs'] < 1e-5
    report['male'] = batch.sbj_gender.tolist()
    model = get_model(cfg).cuda().eval()
    state = torch.load(args.checkpoint, map_location='cuda', weights_only=False)['model']
    model.load_state_dict(state, strict=True)
    model.set_mesh_model(mesh)
    sbj, obj = model.merge_input(batch)
    x0 = torch.cat([sbj, obj, batch.sbj_contacts], dim=1)
    noise = torch.randn_like(x0)
    gt_output = model.split_output(x0)
    gt_meshes, gt_objects = mesh.get_meshes(gt_output, batch.obj_class, batch.scale, batch.sbj_gender)
    for i in range(2):
        gt_meshes[i].export(outdir / f'frame{i}_subject_GT.ply')
        gt_objects[i].export(outdir / f'frame{i}_object_GT.ply')
    # This is a one-step denoising probe, not an unconditional sample or full reconstruction benchmark.
    for t in [0, 100, 250]:
        times = torch.full((2,), t, dtype=torch.long, device='cuda')
        xt = model.scheduler.add_noise(x0, noise, times)
        inp = model.get_input_with_conditioning(xt[:, :334], t=times, t_aux=times,
            obj_group=batch.obj_group, obj_pointnext=batch.obj_pointnext, contact_map=xt[:, 334:])
        pred = model.split_output(model.denoising_model(inp, times, times, times))
        vertices, points, pred_joints = mesh.get_meshes_wkpts_th(pred, batch.obj_class, batch.scale, return_joints=True)
        row = {'timestep': t,
               'subject_mean_vertex_error_cm': (torch.linalg.vector_norm(vertices-gt, dim=-1).mean(-1)*100).tolist(),
               'joint_error_cm': (torch.linalg.vector_norm(pred_joints-joints, dim=-1).mean(-1)*100).tolist(),
               'object_keypoint_error_cm': (torch.linalg.vector_norm(points-batch.obj_keypoints, dim=-1).mean(-1)*100).tolist(),
               'predicted_contact_vertices': (contact.contact_dec(pred.contacts)>.4).sum(-1).tolist()}
        assert all(torch.isfinite(torch.tensor(row[k])).all() for k in ['subject_mean_vertex_error_cm', 'joint_error_cm', 'object_keypoint_error_cm'])
        report['results'].append(row)
        if t == 100:
            pm, po = mesh.get_meshes(pred, batch.obj_class, batch.scale, batch.sbj_gender)
            for i in range(2):
                pm[i].export(outdir/f'frame{i}_subject_denoised_t100.ply')
                po[i].export(outdir/f'frame{i}_object_denoised_t100.ply')
(outdir/'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
