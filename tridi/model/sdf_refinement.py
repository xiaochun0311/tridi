"""Bounded arm/hand refinement of the predicted clean sample during diffusion."""
import torch
import torch.nn.functional as F


def refine_sdf_prediction(model, mode, reference, batch):
    reference = reference.detach()
    widths = (model.data_sbj_channels, model.data_obj_channels, model.data_contacts_channels)
    def pack(value):
        return torch.cat([part for active, part in zip(mode, value.split(widths, dim=1)) if active == '1'], dim=1)
    if model.cg_scale <= 0 or mode[0] != '1':
        return pack(torch.zeros_like(reference))
    gender = getattr(batch, 'sbj_gender', None)
    hands = model.mesh_model.get_hand_mask(gender, len(reference)).to(reference.device)
    indexes = batch.sbj_contact_indexes.to(reference.device)
    if indexes.ndim == 1:
        indexes = indexes.unsqueeze(0).expand(len(reference), -1)
    with torch.no_grad():
        target = getattr(batch, 'sbj_contact_target', None)
        if mode[2] == '0' and target is not None:
            target = target.to(device=reference.device, dtype=torch.bool)
        else:
            _, target = model.contact_model.decode_contacts_th(None, reference[:, sum(widths[:2]):], True)
        target = target.detach() & hands.gather(1, indexes)

    def losses(value):
        output = model.split_output(value)
        vertices, _ = model.mesh_model.get_meshes_wkpts_th(output, batch.obj_class, sbj_gender=gender)
        signed = model.mesh_model.query_object_sdf(vertices, batch.obj_class, output.obj_R, output.obj_c)
        distance = signed.gather(1, indexes)
        contact = (F.relu(distance.abs() - .002).square() * target).sum(1) / target.sum(1).clamp_min(1)
        depth = F.relu(-signed - .001)
        def penetration(mask):
            active = mask & (depth.detach() > 0)
            return (depth.square() * mask).sum(1) / active.sum(1).clamp_min(1)
        hand_pen, body_pen = penetration(hands), penetration(~hands)
        max_depth = depth.max(1).values
        stay = (value[:, 106:322] - reference[:, 106:322]).square().mean(1)
        score = (model.sdf_contact_weight * contact
                 + model.sdf_penetration_weight * (hand_pen + body_pen) + .01 * stay)
        hand_total = (depth.square() * hands).sum(1) / hands.sum(1).clamp_min(1)
        body_total = (depth.square() * ~hands).sum(1) / (~hands).sum(1).clamp_min(1)
        return score, contact, hand_pen, body_pen, max_depth, hand_total, body_total

    with torch.enable_grad():
        current = reference.clone()
        with torch.no_grad():
            initial = losses(current)
        best, best_score = current.clone(), initial[0].clone()
        for _ in range(model.sdf_refine_steps):
            current = current.detach().requires_grad_(True)
            score = losses(current)[0]
            gradient = torch.autograd.grad(score.sum(), current)[0]
            with torch.no_grad():
                # Full joints 16:52 are shoulders, elbows, wrists and fingers.
                delta = current[:, 106:322] - model.cg_scale * gradient[:, 106:322] - reference[:, 106:322]
                candidate = reference.clone()
                candidate[:, 106:322] += delta.clamp(-model.sdf_refine_max_delta, model.sdf_refine_max_delta)
                values = losses(candidate)
                safe = torch.isfinite(candidate).all(1) & torch.isfinite(values[0])
                safe &= values[1] <= initial[1] + 1e-10
                safe &= values[2] <= initial[2] + 1e-10
                safe &= values[3] <= initial[3] + 1e-10
                safe &= values[4] <= initial[4] + 1e-5
                safe &= values[5] <= initial[5] + 1e-10
                safe &= values[6] <= initial[6] + 1e-10
                improve = safe & (values[0] < best_score)
                best[improve] = candidate[improve]
                best_score[improve] = values[0][improve]
                # Unaccepted intermediate steps can move out of conflicting gradients,
                # but only the best feasible bounded candidate is ever returned.
                current = candidate
    return pack((best - reference) / model.cg_scale)
