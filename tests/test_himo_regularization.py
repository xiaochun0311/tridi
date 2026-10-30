import unittest

import torch

from tridi.model.ordered_scene.geometry import NativeSceneGeometry
from tridi.model.ordered_scene.regularization import contact_local_points, relative_contact_loss


class TinyGeometry:
    indices = [0, 2]
    _rotations = staticmethod(NativeSceneGeometry._rotations)

    def decode(self, shape, human):
        points = human + shape[:, None, None, :]
        return points, points


def scene():
    pose = torch.zeros(1, 2, 2, 9)
    pose[..., :6] = torch.tensor([1., 0., 0., 0., 1., 0.])
    pose[:, :, 1, 6:] = torch.tensor([.3, -.2, .1])
    human = torch.tensor([[[[.1, .2, .3], [.2, -.1, .4], [-.2, .1, .5]],
                           [[.15, .25, .35], [.25, -.05, .45], [-.15, .15, .55]]]])
    contact = torch.zeros(1, 2, 2, 2)
    contact[..., 0] = 1
    return dict(shape=torch.zeros(1, 3), human=human, object_pose=pose, contact=contact)


def transformed(state, rotation, translation):
    result = {key: value.clone() for key, value in state.items()}
    result["shape"] = state["shape"] @ rotation.T
    result["human"] = state["human"] @ rotation.T + translation
    old_rotation = TinyGeometry._rotations(state["object_pose"][..., :6])
    new_rotation = rotation @ old_rotation
    result["object_pose"] = torch.cat([new_rotation[..., :2, :].flatten(-2),
        state["object_pose"][..., 6:] @ rotation.T + translation], dim=-1)
    return result


class RelativeContactTraining(unittest.TestCase):
    def test_zero_at_gt_and_invariant_to_common_rigid_world_transform(self):
        geometry, target = TinyGeometry(), scene()
        self.assertEqual(float(relative_contact_loss(geometry, target, target)), 0.)
        predicted = {key: value.clone() for key, value in target.items()}
        predicted["human"] += torch.tensor([.04, -.02, .01])
        before = relative_contact_loss(geometry, predicted, target)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = torch.tensor([1., 2., -.3])
        after = relative_contact_loss(geometry, transformed(predicted, rotation, translation),
                                      transformed(target, rotation, translation))
        torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)

    def test_gradients_reach_shape_human_object_but_not_gt_or_contact_head(self):
        geometry, target = TinyGeometry(), scene()
        predicted = {key: value.clone() for key, value in target.items()}
        predicted["human"] += torch.tensor([.04, .02, -.01])
        predicted = {key: value.requires_grad_() for key, value in predicted.items()}
        target = {key: value.requires_grad_() for key, value in target.items()}
        cached = contact_local_points(geometry, target).detach().requires_grad_()
        loss = relative_contact_loss(geometry, predicted, target, target_points=cached)
        loss.backward()
        for key in ("shape", "human", "object_pose"):
            self.assertIsNotNone(predicted[key].grad)
            self.assertGreater(float(predicted[key].grad.abs().sum()), 0.)
        self.assertGreater(float(predicted["object_pose"].grad[..., 6:].abs().sum()), 0.)
        self.assertGreater(float(predicted["object_pose"].grad[..., :6].abs().sum()), 0.)
        self.assertIsNone(predicted["contact"].grad)
        self.assertIsNone(cached.grad)
        self.assertTrue(all(value.grad is None for value in target.values()))

    def test_gt_mask_only_empty_mask_connected_zero_and_cache_validation(self):
        geometry, target = TinyGeometry(), scene()
        predicted = {key: value.clone() for key, value in target.items()}
        predicted["human"] += .04
        predicted["contact"].zero_()
        baseline = relative_contact_loss(geometry, predicted, target)
        predicted["contact"].fill_(1.)
        torch.testing.assert_close(relative_contact_loss(geometry, predicted, target), baseline)
        # The second sampled body point has no GT contact, so changing only it is ignored.
        predicted["human"][:, :, 2] += 100.
        torch.testing.assert_close(relative_contact_loss(geometry, predicted, target), baseline)
        with self.assertRaises(ValueError):
            relative_contact_loss(geometry, predicted, target, target_points=torch.zeros(1))
        target["contact"].zero_()
        predicted = {key: value.detach().requires_grad_() for key, value in predicted.items()}
        zero = relative_contact_loss(geometry, predicted, target)
        self.assertEqual(float(zero.detach()), 0.)
        zero.backward()
        for key in ("shape", "human", "object_pose"):
            torch.testing.assert_close(predicted[key].grad, torch.zeros_like(predicted[key]))


if __name__ == "__main__":
    unittest.main()
