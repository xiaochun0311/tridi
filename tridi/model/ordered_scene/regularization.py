"""Training-only local contact-point supervision; no inference pose correction."""
import torch
from torch.nn import functional as F


def contact_local_points(geometry, state):
    """Return [B, 2, 2, K, 3] body samples in each predicted object's frame.

    NativeSceneGeometry uses K=690. This function keeps input gradients, so a
    frozen SMPL-X layer still transmits gradients to shape and human pose.
    """
    vertices, _ = geometry.decode(state["shape"], state["human"])
    indices = torch.as_tensor(geometry.indices, dtype=torch.long, device=vertices.device)
    points = vertices.index_select(2, indices).unsqueeze(2)
    pose = state["object_pose"].to(device=vertices.device, dtype=vertices.dtype)
    rotation = geometry._rotations(pose[..., :6])
    return (points - pose[..., 6:].unsqueeze(-2)) @ rotation


def relative_contact_loss(geometry, prediction, targets, target_points=None):
    """Match GT positive contact points in object-local coordinates.

    SmoothL1 beta is 1cm and the mean point loss is normalized by 5cm. GT
    proximity points retain their recorded offsets, rather than being forced
    onto the surface. Only GT labels select points; the predicted contact head
    cannot switch this loss off. A cached target_points tensor must correspond
    to the same target batch and fixed contact indices.
    """
    labels = targets["contact"]
    expected = (len(prediction["shape"]), 2, 2, len(geometry.indices))
    if not isinstance(labels, torch.Tensor) or tuple(labels.shape) != expected or not bool(torch.isfinite(labels).all()):
        raise ValueError("GT contact labels must be a finite [B,2,2,K] tensor")
    mask = labels > 0
    if not bool(mask.any()):
        return sum(prediction[key].sum() for key in ("shape", "human", "object_pose")) * 0.
    predicted = contact_local_points(geometry, prediction)
    if tuple(mask.shape) != tuple(predicted.shape[:-1]):
        raise ValueError("GT contact labels do not match the native sampled point layout")
    if target_points is None:
        with torch.no_grad():
            target_points = contact_local_points(geometry, targets)
    if not isinstance(target_points, torch.Tensor) or tuple(target_points.shape) != tuple(predicted.shape):
        raise ValueError("Cached GT local points must match the predicted sampled point layout")
    target_points = target_points.detach().to(device=predicted.device, dtype=predicted.dtype)
    if not bool(torch.isfinite(target_points).all()):
        raise ValueError("GT local points must be finite")
    mask = mask.to(device=predicted.device)
    error = F.smooth_l1_loss(predicted, target_points, beta=.01, reduction="none").mean(-1)
    return error[mask].mean() / .05
