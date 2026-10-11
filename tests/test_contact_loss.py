import unittest
from types import SimpleNamespace as NS

import torch
from tridi.core.trainer import Trainer


class ContactLossTest(unittest.TestCase):
    def test_high_noise_object_is_excluded(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = NS(train=NS(loss_t_stamp_threshold=250, losses={'sbj_contacts': 1.0}))
        vertices = torch.tensor([[[0., 0., 0.]], [[100., 0., 0.]]], requires_grad=True)
        output = NS(sbj_vertices=vertices, obj_keypoints=torch.ones(2, 1, 3),
                    timesteps_sbj=torch.tensor([250, 250]), timesteps_obj=torch.tensor([250, 251]))
        batch = NS(sbj_contact_indexes=torch.tensor([[0], [0]]), sbj_contacts_full=torch.zeros(2, 1))
        loss, _ = trainer.compute_loss(batch, output, {})
        self.assertAlmostEqual(loss.item(), 3.0, places=5)
        loss.backward()
        self.assertTrue(torch.equal(vertices.grad[1], torch.zeros_like(vertices.grad[1])))

    def test_empty_geometric_losses_have_finite_zero_gradients(self):
        trainer = Trainer.__new__(Trainer)
        losses = {'smpl_v2v': 30.0, 'obj_v2v': 10.0, 'sbj_contacts': 20.0}
        trainer.cfg = NS(train=NS(loss_t_stamp_threshold=250, losses=losses))
        vertices = torch.zeros(2, 1, 3, requires_grad=True)
        points = torch.ones(2, 1, 3, requires_grad=True)
        output = NS(sbj_vertices=vertices, obj_keypoints=points,
                    timesteps_sbj=torch.tensor([251, 999]), timesteps_obj=torch.tensor([251, 999]))
        batch = NS(sbj_vertices=torch.ones_like(vertices), obj_keypoints=torch.zeros_like(points),
                   sbj_contact_indexes=torch.tensor([[0], [0]]), sbj_contacts_full=torch.zeros(2, 1))
        loss, log = trainer.compute_loss(batch, output, {})
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(all(value == 0.0 for value in log.values()))
        loss.backward()
        self.assertTrue(torch.equal(vertices.grad, torch.zeros_like(vertices)))
        self.assertTrue(torch.equal(points.grad, torch.zeros_like(points)))


if __name__ == '__main__':
    unittest.main()
