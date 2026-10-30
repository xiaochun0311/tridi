"""Joint S1/S2 scene denoising conditioned on S0 and two ordered action labels.

Action IDs and object references are learned conditioning, never pose-edit rules.
This architecture needs recorded two-task-object S0/S1/S2 supervision to train.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn

from tridi.model.denoising.transformer_uni_3 import Projection, TransformertUni3WayModel


class OrderedSceneDenoiser(nn.Module):
    def __init__(self, num_actions, descriptor_dim=12, checkpoint="assets/gb_main.pth"):
        super().__init__()
        if num_actions < 1:
            raise ValueError("A training-defined action vocabulary is required")
        self.num_actions, self.descriptor_dim = num_actions, descriptor_dim
        self.core = TransformertUni3WayModel(
            dim_sbj=325, dim_obj=9, dim_contact=128, dim_cond=1064,
            dim_hidden=312, dim_timestep_embed=128, dim_output=462,
            num_layers=8, nhead=8, dim_feedforward=2048,
        )
        self.initialization = dict(checkpoint=None, sha256=None, strictly_loaded_tensors=0)
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            prefix = "denoising_model.model."
            state = {k[len(prefix):]: v for k, v in saved.items() if k.startswith(prefix)}
            self.core.load_state_dict(state, strict=True)
            self.initialization = dict(checkpoint=str(Path(checkpoint).resolve()),
                sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                strictly_loaded_tensors=len(state))

        # Keep the unused legacy condition/contact-AE modules frozen.
        self.core.requires_grad_(False)
        for name in ["transformer_encoder", "layernorm", "projection_T",
                     "projection_S_shape", "projection_S_orient", "projection_S_pose", "projection_S_transl",
                     "projection_O_orient", "projection_O_transl", "decoder_S_shape", "decoder_S_orient",
                     "decoder_S_pose", "decoder_S_transl", "decoder_O_orient", "decoder_O_transl"]:
            getattr(self.core, name).requires_grad_(True)
        for name in ["sbj_embedding", "obj_embedding", "cnt_embedding", "cond_embedding",
                     "sbj_pose_embedding", "sbj_shape_embedding", "global_orient_embedding", "global_transl_embedding"]:
            getattr(self.core, name).requires_grad_(True)
        self.contact_projection = Projection(690, 312)
        self.contact_head = Projection(312, 690)
        self.descriptor_projection = Projection(descriptor_dim, 312)
        self.action_embedding = nn.Embedding(num_actions, 312)
        self.step_embedding = nn.Embedding(2, 312)
        self.source_projection = nn.Linear(312, 312)
        self.target_projection = nn.Linear(312, 312)
        self.no_target = nn.Parameter(torch.zeros(312))

    def forward(self, shape, human, object_pose, contact, initial_object_pose,
                descriptors, action_ids, action_objects, action_targets, t):
        batch = len(shape)
        expected = [(shape, (batch, 10)), (human, (batch, 2, 315)),
                    (object_pose, (batch, 2, 2, 9)), (contact, (batch, 2, 2, 690)),
                    (initial_object_pose, (batch, 2, 9)),
                    (descriptors, (batch, 2, self.descriptor_dim)),
                    (action_ids, (batch, 2)), (action_objects, (batch, 2)),
                    (action_targets, (batch, 2))]
        if any(tuple(value.shape) != size for value, size in expected) or tuple(t.shape) not in {(batch,), (batch, 3)}:
            raise ValueError("Expected S0 conditions and two noisy future scene states")
        times = t[:, None].expand(-1, 3) if t.ndim == 1 else t
        core = self.core
        step = self.step_embedding(torch.arange(2, device=shape.device))
        geometry = self.descriptor_projection(descriptors)
        initial = core.prepare_obj(initial_object_pose.reshape(batch * 2, 9)).reshape(batch, 2, 2, 312)
        initial = initial + geometry[:, :, None] + core.cond_embedding
        identity = initial.mean(2)
        index = torch.arange(batch, device=shape.device)[:, None]
        source = identity[index, action_objects]
        target = identity[index, action_targets.clamp(max=1)]
        target = torch.where((action_targets == 2)[..., None], self.no_target, target)
        actions = (self.action_embedding(action_ids) + step[None]
                   + self.source_projection(source) + self.target_projection(target) + core.cond_embedding)

        shared = (core.projection_S_shape(shape) + core.sbj_embedding + core.sbj_shape_embedding)[:, None]
        body = torch.stack([
            core.projection_S_orient(human[..., :6]) + core.global_orient_embedding,
            core.projection_S_pose(human[..., 6:312]) + core.sbj_pose_embedding,
            core.projection_S_transl(human[..., 312:]) + core.global_transl_embedding,
        ], dim=2) + core.sbj_embedding + step[None, :, None]
        future = core.prepare_obj(object_pose.reshape(batch * 4, 9)).reshape(batch, 2, 2, 2, 312)
        future = future + identity[:, None, :, None] + step[None, :, None, None]
        contacts = (self.contact_projection(contact) + core.cnt_embedding
                    + identity[:, None] + step[None, :, None])
        timing = core.prepare_time(times[:, 0], times[:, 1], times[:, 2], shape.device)
        # 1 shape | 4 initial-object | 2 action | 3 time | 6 human | 8 object | 4 contact.
        tokens = torch.cat([shared, initial.reshape(batch, 4, 312), actions, *timing,
                            body.reshape(batch, 6, 312), future.reshape(batch, 8, 312),
                            contacts.reshape(batch, 4, 312)], dim=1)
        encoded = core.transformer_encoder(core.layernorm(tokens))
        body = encoded[:, 10:16].reshape(batch, 2, 3, 312)
        future = encoded[:, 16:24].reshape(batch, 2, 2, 2, 312)
        return dict(
            shape=core.decoder_S_shape(encoded[:, 0]),
            human=torch.cat([core.decoder_S_orient(body[:, :, 0]),
                             core.decoder_S_pose(body[:, :, 1]),
                             core.decoder_S_transl(body[:, :, 2])], dim=-1),
            object_pose=torch.cat([core.decoder_O_orient(future[:, :, :, 0]),
                                   core.decoder_O_transl(future[:, :, :, 1])], dim=-1),
            contact=self.contact_head(encoded[:, 24:28].reshape(batch, 2, 2, 312)),
        )

    def manifest(self):
        return dict(**self.initialization, num_actions=self.num_actions, descriptor_dim=self.descriptor_dim,
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            prediction_type="sample (clean x0)", token_count=28,
            noise_protocol="TriDi seven sparse human/object/contact timestep patterns",
            losses="Human and object L1; positive-balanced contact MSE adaptation",
            geometry_descriptor="Canonical mesh extents/mean/absmean/std (12 values), not PointNeXt",
            outputs=["shared shape", "two human poses", "two poses for each of two task objects", "per-object contacts"],
            conditions="S0 object poses and geometry; two ordered action IDs and object references",
            inference_rules="No IK, per-action pose edits, supplied future poses, or sample-specific correction")
