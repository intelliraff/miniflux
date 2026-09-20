import unittest

import torch

from models.cifar_miniflux import CifarMiniFlux
from models.cifar_reference import make_reference, reference_components
from models.flow_matching import sample_flow_path


class CifarModelTests(unittest.TestCase):
    def test_official_path_matches_project_equations(self):
        _, _, _, _, path_class = reference_components()
        x0 = torch.randn(3, 3, 8, 8)
        x1 = torch.randn_like(x0)
        t = torch.tensor([0.0, 0.4, 1.0])
        project_x, project_v = sample_flow_path(x0, x1, t)
        official = path_class().sample(x_0=x0, x_1=x1, t=t)
        self.assertTrue(torch.equal(project_x, official.x_t))
        self.assertTrue(torch.equal(project_v, official.dx_t))

    def test_miniflux_is_smaller_and_has_expected_shape(self):
        compact = CifarMiniFlux(hidden_dim=32, depth=2)
        reference = make_reference()
        self.assertLess(
            sum(p.numel() for p in compact.parameters()),
            sum(p.numel() for p in reference.parameters()),
        )
        x = torch.randn(2, 3, 32, 32)
        output = compact(x, torch.tensor([0.2, 0.8]), extra={})
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
