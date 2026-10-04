import torch

from tridi.model.tridi import _penetration_loss


def test_penetration_gradient_pushes_inside_point_outward():
    vertices = torch.tensor([[[0., 0., -0.01], [0., 0., 0.01]]], requires_grad=True)
    object_points = torch.zeros(1, 1, 3)
    normals = torch.tensor([[[0., 0., 1.]]])
    nearest = torch.zeros(1, 2, dtype=torch.long)
    distances = torch.tensor([[0.01, 0.01]])

    loss = _penetration_loss(vertices, object_points, normals, nearest, distances)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(0.005))
    assert vertices.grad[0, 0, 2] < 0  # Gradient descent moves the inside point outward.
    assert vertices.grad[0, 1, 2] == 0


if __name__ == "__main__":
    test_penetration_gradient_pushes_inside_point_outward()
