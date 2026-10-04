from typing import Dict, Union, List, Optional, Tuple

import numpy as np
import smplx
import torch
import trimesh

from tridi.data.batch_data import BatchData
from tridi.model.base import TriDiModelOutput
from tridi.utils.geometry import rotation_6d_to_matrix


class MeshModel:
    """
    A class that generates human and object mesh given the parameters
    predicted by the denoising model.
    """
    def __init__(
        self, model_path: str, batch_size: int,
        canonical_obj_meshes: Dict[int, trimesh.Trimesh],
        canonical_obj_keypoints: Dict[int, Dict[str, np.ndarray]],
        device='cpu'
    ) -> None:
        self.smpl_f = smplx.build_layer(
            model_path=model_path, model_type="smplh", gender="female",
            use_pca=False, num_betas=10, batch_size=batch_size
        ).to(device)

        self.smpl_m = smplx.build_layer(
            model_path=model_path, model_type="smplh", gender="male",
            use_pca=False, num_betas=10, batch_size=batch_size
        ).to(device)

        # object meshes and keypoints
        self.canonical_obj_meshes = canonical_obj_meshes
        self.canonical_obj_keypoints = canonical_obj_keypoints
        class_ids = sorted(self.canonical_obj_keypoints.keys())
        self.canonical_obj_keypoints_th = {}
        self.canonical_obj_normals_th = {}
        for class_id in class_ids:
            self.canonical_obj_keypoints_th[class_id] = torch.tensor(
                self.canonical_obj_keypoints[class_id]["cartesian"], dtype=torch.float,
                requires_grad=False, device=device
            )
            if "normals" in self.canonical_obj_keypoints[class_id]:
                normals = self.canonical_obj_keypoints[class_id]["normals"]
            elif "triangles_ids" in self.canonical_obj_keypoints[class_id]:
                normals = self.canonical_obj_meshes[class_id].face_normals[
                    self.canonical_obj_keypoints[class_id]["triangles_ids"]
                ]
            else:
                normals = None
            if normals is not None:
                self.canonical_obj_normals_th[class_id] = torch.tensor(
                    normals, dtype=torch.float, requires_grad=False, device=device
                )
        canonical_obj_pcs: Dict[int, torch.Tensor] = {}
        for class_id in class_ids:
            canonical_obj_pcs[class_id] = torch.tensor(
                self.canonical_obj_meshes[class_id].vertices,
                dtype=torch.float, requires_grad=False, device=device
            )
        self.canonical_obj_pcs = canonical_obj_pcs

        # batch size
        self.batch_size = batch_size
        self.device = device

    @staticmethod
    def _apply_scale(
            sbj_vertices: torch.Tensor, obj_vertices: torch.Tensor,
            sbj_joints: torch.Tensor,
            scale: Optional[Union[torch.Tensor, float]] = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if scale is not None:
            if not isinstance(scale, torch.Tensor):
                scale = torch.tensor(scale, dtype=torch.float, requires_grad=False)
            else:
                scale = scale.clone().detach().float()

            scale = scale.to(obj_vertices.device).reshape(-1, 1, 1)
            obj_vertices *= scale
            sbj_vertices *= scale
            sbj_joints *= scale

        return sbj_vertices, obj_vertices, sbj_joints

    def get_meshes_th(
        self, output: TriDiModelOutput, obj_class, scale=1.0,
        sbj_gender=None, return_joints=False
    ):
        B = min(self.batch_size, len(output))

        sbj_pose = output.sbj_pose.reshape(B, -1, 6)
        sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(B, -1, 9)
        sbj_global = output.sbj_global.reshape(B, 1, 6)
        sbj_global = rotation_6d_to_matrix(sbj_global).reshape(B, 1, 9)
        body_model_params = {
            "betas": output.sbj_shape,
            "transl": output.sbj_c,
            "global_orient": sbj_global,
            "body_pose": sbj_pose[:, :21],
            "left_hand_pose": sbj_pose[:, 21: 21 + 15],
            "right_hand_pose": sbj_pose[:, 21 + 15:]
        }

        # get sbj mesh
        sbj_vertices, sbj_joints = self.get_smpl_th(body_model_params, sbj_gender)

        # obj mesh
        obj_R = output.obj_R.reshape(B, 1, 6)
        obj_R = rotation_6d_to_matrix(obj_R).reshape(B, 3, 3)

        # we use list here to get rid of dependency on Pointclouds from pytorch3d
        obj_cls_list = obj_class.cpu().tolist()
        obj_verts_list = [self.canonical_obj_pcs[int(c)].clone() for c in obj_cls_list]  # each is (n_i, 3)
        max_n = max(v.shape[0] for v in obj_verts_list)
        obj_vertices = torch.zeros((B, max_n, 3), dtype=obj_verts_list[0].dtype)
        for i, v in enumerate(obj_verts_list):
            obj_vertices[i, : v.shape[0]] = v
        obj_vertices = obj_vertices.to(self.device)

        # Pose object
        obj_vertices = torch.bmm(obj_R, obj_vertices.transpose(2, 1)).transpose(2, 1)
        obj_vertices += output.obj_c.unsqueeze(1)

        # Optionally apply scale
        sbj_vertices, obj_vertices, sbj_joints = self._apply_scale(
            sbj_vertices, obj_vertices, sbj_joints, scale
        )

        if return_joints:
            return sbj_vertices, obj_vertices, sbj_joints
        else:
            return sbj_vertices, obj_vertices

    def get_meshes_wkpts_th(
        self, output: TriDiModelOutput, obj_class, scale=1.0,
        sbj_gender=None, return_joints=False
    ):
        B = min(self.batch_size, len(output))

        sbj_pose = output.sbj_pose.reshape(B, -1, 6)
        sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(B, -1, 9)
        sbj_global = output.sbj_global.reshape(B, 1, 6)
        sbj_global = rotation_6d_to_matrix(sbj_global).reshape(B, 1, 9)
        body_model_params = {
            "betas": output.sbj_shape,
            "transl": output.sbj_c,
            "global_orient": sbj_global,
            "body_pose": sbj_pose[:, :21],
            "left_hand_pose": sbj_pose[:, 21: 21 + 15],
            "right_hand_pose": sbj_pose[:, 21 + 15:]
        }

        # get sbj mesh
        sbj_vertices, sbj_joints = self.get_smpl_th(body_model_params, sbj_gender)

        # obj mesh
        obj_R = output.obj_R.reshape(B, 1, 6)
        obj_R = rotation_6d_to_matrix(obj_R).reshape(B, 3, 3)
        obj_keypoints = torch.stack([
            self.canonical_obj_keypoints_th[int(class_id)].clone()
            for class_id in obj_class.detach().cpu().tolist()
        ])

        # Pose object
        obj_keypoints = torch.bmm(obj_R, obj_keypoints.transpose(2, 1)).transpose(2, 1)
        obj_keypoints += output.obj_c.unsqueeze(1)

        # Optionally apply scale
        sbj_vertices, obj_vertices, sbj_joints = self._apply_scale(
            sbj_vertices, obj_keypoints, sbj_joints, scale
        )

        if return_joints:
            return sbj_vertices, obj_keypoints, sbj_joints
        else:
            return sbj_vertices, obj_keypoints

    def get_faces_np(self, obj_class=None):
        if obj_class == None:
            return self.smpl_m.faces
        else:
            return self.smpl_m.faces, [self.canonical_obj_meshes[c].faces for c in obj_class]

    def to(self, device):
        self.smpl_f = self.smpl_f.to(device)
        self.smpl_m = self.smpl_m.to(device)
        self.canonical_obj_keypoints_th = {
            class_id: keypoints.to(device)
            for class_id, keypoints in self.canonical_obj_keypoints_th.items()
        }
        self.canonical_obj_normals_th = {
            class_id: normals.to(device)
            for class_id, normals in self.canonical_obj_normals_th.items()
        }
        self.canonical_obj_pcs = {
            class_id: vertices.to(device)
            for class_id, vertices in self.canonical_obj_pcs.items()
        }

        self.device = device

        return self

    @torch.no_grad()
    def get_meshes(self, output: TriDiModelOutput, obj_class, scale=1.0, sbj_gender=None):
        sbj_vertices, obj_vertices = self.get_meshes_th(output, obj_class, scale, sbj_gender)

        # create subject mesh
        sbj_vertices = sbj_vertices.cpu().numpy()
        sbj_faces = self.smpl_m.faces
        sbj_meshes = [trimesh.Trimesh(sbj_vertices[j], sbj_faces) for j in range(sbj_vertices.shape[0])]

        # create object mesh
        obj_meshes = []
        obj_vertices = obj_vertices.cpu().numpy()
        for j in range(obj_vertices.shape[0]):
            canonical_obj_mesh = self.canonical_obj_meshes[obj_class[j].cpu().item()]
            obj_meshes.append(
                trimesh.Trimesh(
                    obj_vertices[j, : len(canonical_obj_mesh.vertices)],
                    canonical_obj_mesh.faces
                )
            )

        return sbj_meshes, obj_meshes

    def get_smpl_th(self, body_model_params: Union[Dict, BatchData], sbj_gender=None):
        if isinstance(body_model_params, BatchData):
            B = len(body_model_params['sbj_shape'])
            sbj_gender = body_model_params.sbj_gender

            sbj_pose = body_model_params['sbj_pose'].reshape(B, -1, 6)
            sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(B, -1, 9).float()
            sbj_global = body_model_params['sbj_global'].reshape(B, 1, 6)
            sbj_global = rotation_6d_to_matrix(sbj_global).reshape(B, 1, 9).float()

            # SMPL-H 人体骨架是这样定义的：1 个根关节、21 个身体关节，以及左右手各 15 个关节
            # 1 个根关节：控制整个人的朝向，单独存为 sbj_global。
            # 21 个身体关节：控制躯干、头、手臂、腿等，存为 body_pose。
            # 左手 15 个＋右手 15 个：每只手有 5 根手指，每根手指用 3 个关节旋转表示，分别存为 left_hand_pose、right_hand_pose。
            body_model_params = {
                "betas": body_model_params['sbj_shape'], # 体型
                "transl": body_model_params['sbj_c'], # 人物位置
                "global_orient": sbj_global, # 朝向
                "body_pose": sbj_pose[:, :21], # 21个身体关节
                "left_hand_pose": sbj_pose[:, 21:36], # 左手15个关节
                "right_hand_pose": sbj_pose[:, 36:] #右手15个关节
            }
        B = min(self.batch_size, len(body_model_params['betas']))
        body_model_params = {k: v.to(self.device) for k, v in body_model_params.items()}

        if sbj_gender is not None:
            if not sbj_gender.any():  # all female
                # get smpl(-h) vertices
                sbj_output = self.smpl_f(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
                sbj_vertices = sbj_output['vertices']
                sbj_joints = sbj_output['joints']
            elif sbj_gender.all():  # all male
                # get smpl(-h) vertices
                sbj_output = self.smpl_m(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
                sbj_vertices = sbj_output['vertices']
                sbj_joints = sbj_output['joints']
            else:  # mixed
                body_model_params_m = {k: v[sbj_gender] for k, v in body_model_params.items()}
                body_model_params_f = {k: v[~sbj_gender] for k, v in body_model_params.items()} # ～按位取反

                sbj_output_m = self.smpl_m(
                    pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params_m
                )
                sbj_output_f = self.smpl_f(
                    pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params_f
                )

                n_verts = sbj_output_m['vertices'].size(1)
                n_joints = sbj_output_m['joints'].size(1)

                sbj_vertices = torch.zeros((B, n_verts, 3), dtype=torch.float, device=self.device)
                sbj_joints = torch.zeros((B, n_joints, 3), dtype=torch.float, device=self.device)

                sbj_vertices[sbj_gender] = sbj_output_m['vertices']
                sbj_joints[sbj_gender] = sbj_output_m['joints']
                sbj_vertices[~sbj_gender] = sbj_output_f['vertices']
                sbj_joints[~sbj_gender] = sbj_output_f['joints']
        else:  # default to male
            # get smpl(-h) vertices
            sbj_output = self.smpl_m(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
            sbj_vertices = sbj_output['vertices']
            sbj_joints = sbj_output['joints']

        return sbj_vertices, sbj_joints

    @torch.no_grad()
    def get_smpl_np(self, body_model_params, sbj_gender=None):
        sbj_vertices, sbj_joints = self.get_smpl_th(body_model_params, sbj_gender)

        return sbj_vertices.cpu().numpy(), sbj_joints.cpu().numpy()
