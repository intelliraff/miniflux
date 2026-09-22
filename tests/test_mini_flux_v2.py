import unittest
import torch
from models.mini_flux import MiniFlux
from models.mini_flux_v2 import MiniFluxV2, sinusoidal_positions
from models.strong_latent_miniflux import StrongLatentMiniFlux


class MiniFluxV2Tests(unittest.TestCase):
    def test_parameter_budget(self):
        v1 = sum(p.numel() for p in MiniFlux().parameters())
        v2 = sum(p.numel() for p in MiniFluxV2().parameters())
        self.assertLessEqual(v2, v1)

    def test_spatial_axes_and_gradient_coverage(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        model = MiniFluxV2(hidden_dim=32, depth=2)
        latent = torch.randn(2, 4, 8, 6)
        text = torch.randn(2, 7, 512)
        timestep = torch.tensor([.2, .8])
        target = torch.randn_like(latent)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Zero-initialized final layer/gates intentionally delay some gradients.
        for _ in range(4):
            optimizer.zero_grad()
            output = model(latent, text, timestep)
            self.assertEqual(output.shape, latent.shape)
            self.assertTrue(torch.isfinite(output).all())
            (output-target).square().mean().backward()
            optimizer.step()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        with torch.no_grad():
            original = model(latent, text, timestep)
            self.assertFalse(torch.allclose(original, model(latent, text.flip(0), timestep)))
            self.assertFalse(torch.allclose(original, model(latent, text, timestep.flip(0))))
        positions = sinusoidal_positions(torch.tensor([0, 1]), 16)
        self.assertFalse(torch.equal(positions[0], positions[1]))

    def test_rejects_odd_latent_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'even'):
            MiniFluxV2(hidden_dim=32, depth=1)(torch.randn(1,4,7,8),torch.randn(1,7,512),torch.zeros(1))

    def test_strong_latent_model_shape_text_effect_and_budget(self):
        model=StrongLatentMiniFlux(hidden_dim=32,depth=2,heads=4,condition_rank=8)
        model.final.weight.data.normal_(std=.01)
        latent=torch.randn(2,4,16,16); text=torch.randn(2,7,512); timestep=torch.tensor([.2,.8])
        output=model(latent,text,timestep)
        self.assertEqual(output.shape,latent.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.allclose(output,model(latent,text.flip(0),timestep)))
        self.assertLess(sum(p.numel() for p in StrongLatentMiniFlux().parameters()),10_000_000)


if __name__ == '__main__':
    unittest.main()
