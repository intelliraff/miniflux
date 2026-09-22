import unittest

import torch

from models.cifar_miniflux import CifarMiniFlux
from models.cifar_hierarchical_miniflux import HierarchicalCifarMiniFlux
from models.cifar_hybrid_miniflux import CifarHybridMiniFlux
from models.cifar_conditioned_hybrid_miniflux import ConditionedCifarHybridMiniFlux
from models.cifar_strong_conditioned_hybrid_miniflux import StrongConditionedCifarHybridMiniFlux
from experiments.cifar_benchmark import guided_velocity, heun_sample
from models.cifar_reference import make_reference, reference_components
from models.flow_matching import sample_flow_path


class CifarModelTests(unittest.TestCase):
    def test_conditioned_hybrid_shape_null_labels_and_class_effect(self):
        torch.manual_seed(7)
        model=ConditionedCifarHybridMiniFlux(local_dim=16,base_dim=32,bottleneck_dim=64,bottleneck_depth=1)
        x=torch.randn(2,3,32,32); t=torch.tensor([.2,.7])
        # Train through the zero-initialized output head so conditioning has a
        # path to alter the generated velocity, as it will after optimization.
        (model(x,t,labels=torch.tensor([0,1]))-torch.randn_like(x)).square().mean().backward()
        torch.optim.SGD(model.parameters(),lr=1e-3).step()
        model.output[-1].weight.data.fill_(0.001)
        for module in model.modules():
            if isinstance(module,torch.nn.Linear) and module is not model.class_embedding:
                if module.weight.abs().sum()==0:
                    torch.nn.init.normal_(module.weight,std=.01)
        a=model(x,t,labels=torch.tensor([0,1])); b=model(x,t,labels=torch.tensor([2,3]))
        null=model(x,t)
        self.assertEqual(a.shape,x.shape)
        self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(torch.allclose(a,b))
        self.assertEqual(null.shape,x.shape)
        self.assertLess(sum(p.numel() for p in ConditionedCifarHybridMiniFlux().parameters()),10_000_000)

    def test_guidance_formula_and_conditioned_heun(self):
        class Dummy(torch.nn.Module):
            NULL_CLASS=10
            def __init__(self):
                super().__init__(); self.seen=[]
            def forward(self,x,t,labels=None,extra=None):
                self.seen.extend(labels.tolist())
                return torch.ones_like(x)*(0 if torch.all(labels==10) else 2)
        model=Dummy(); x=torch.zeros(2,3,4,4); t=torch.zeros(2); labels=torch.tensor([1,1])
        self.assertTrue(torch.equal(guided_velocity(model,x,t,labels,1.5,True),torch.full_like(x,3)))
        sampled=heun_sample(model,x,torch.tensor([0.,.5,1.]),labels,2.,True)
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertEqual(len(model.seen),20)  # direct pair and both CFG branches at every Heun stage

    def test_classifier_free_dropout_replaces_selected_labels(self):
        labels=torch.tensor([0,1,2,3])
        mask=torch.rand((4,),generator=torch.Generator().manual_seed(5))<1.0
        labels=labels.clone(); labels[mask]=ConditionedCifarHybridMiniFlux.NULL_CLASS
        self.assertTrue(torch.equal(labels,torch.full((4,),10)))

    def test_strong_conditioning_is_direct_and_under_parameter_budget(self):
        model=StrongConditionedCifarHybridMiniFlux(local_dim=16,base_dim=32,bottleneck_dim=64,bottleneck_depth=1,class_rank=8)
        model.output[-1].weight.data.normal_(std=.01)
        x=torch.randn(2,3,32,32); t=torch.tensor([.25,.75])
        first=model(x,t,labels=torch.tensor([0,1]))
        second=model(x,t,labels=torch.tensor([2,3]))
        null=model(x,t)
        self.assertEqual(first.shape,x.shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.allclose(first,second))
        self.assertEqual(null.shape,x.shape)
        full=StrongConditionedCifarHybridMiniFlux()
        self.assertLess(sum(p.numel() for p in full.parameters()),10_000_000)

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

    def test_hierarchical_miniflux_shape_gradients_and_size(self):
        model = HierarchicalCifarMiniFlux(
            base_dim=32,
            bottleneck_dim=64,
            encoder_depth=1,
            bottleneck_depth=1,
            decoder_depth=1,
        )
        x = torch.randn(2, 3, 32, 32)
        output = model(x, torch.tensor([0.2, 0.8]), extra={})
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(model.output.weight.grad)

        full_model = HierarchicalCifarMiniFlux()
        parameters = sum(parameter.numel() for parameter in full_model.parameters())
        self.assertGreater(parameters, 6_000_000)
        self.assertLess(parameters, 10_000_000)

    def test_hybrid_miniflux_shape_gradients_and_size(self):
        model = CifarHybridMiniFlux(
            local_dim=16,
            base_dim=32,
            bottleneck_dim=64,
            bottleneck_depth=1,
        )
        x = torch.randn(2, 3, 32, 32)
        target = torch.randn_like(x)
        output = model(x, torch.tensor([0.2, 0.8]), extra={})
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())
        (output - target).square().mean().backward()
        self.assertIsNotNone(model.output[-1].weight.grad)

        full_model = CifarHybridMiniFlux()
        parameters = sum(parameter.numel() for parameter in full_model.parameters())
        self.assertGreater(parameters, 6_000_000)
        self.assertLess(parameters, 10_000_000)


if __name__ == "__main__":
    unittest.main()
