from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules.mot import C2fMoT, MoTBlock, anneal_mot_temperature, collect_mot_aux_loss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import _collect_mot_aux_loss


ROOT = Path(__file__).resolve().parents[1]


def _has_grad(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
        if p.requires_grad
    )


def test_mot_block_forward_backward_all_experts_trainable():
    torch.manual_seed(0)
    module = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, mlp_ratio=1.5).train()
    x = torch.randn(2, 32, 8, 8)

    out, aux = module(x)
    assert out.shape == x.shape
    assert aux.requires_grad
    assert torch.isfinite(aux)

    (out.mean() + aux).backward()
    assert _has_grad(module.router)
    for expert in module.experts:
        assert _has_grad(expert)


class _FailIfCalled(nn.Module):
    def forward(self, x):
        raise AssertionError("inactive expert should be skipped during eval")


def test_mot_eval_skips_inactive_experts():
    torch.manual_seed(0)
    module = MoTBlock(32, num_heads=4, top_k=1, window_size=4, n_points=2, mlp_ratio=1.5).eval()
    with torch.no_grad():
        final_router = module.router.router[-1]
        final_router.weight.zero_()
        final_router.bias.copy_(torch.tensor([8.0, -8.0, -9.0]))
    module.experts[1] = _FailIfCalled()
    module.experts[2] = _FailIfCalled()

    with torch.no_grad():
        out, aux = module(torch.randn(2, 32, 8, 8))
    assert out.shape == (2, 32, 8, 8)
    assert not aux.requires_grad


def test_mot_exploration_default_is_small():
    module = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2)
    assert module.router.exploration_eps == 0.001


def test_c2fmot_collects_aux_loss_and_keeps_shape():
    torch.manual_seed(0)
    module = C2fMoT(48, 64, n=2, num_heads=4, top_k=2, window_size=4, n_points=2).train()
    out = module(torch.randn(2, 48, 8, 8))

    assert out.shape == (2, 64, 8, 8)
    aux = collect_mot_aux_loss(module)
    assert aux.requires_grad
    assert torch.isfinite(aux)
    assert _collect_mot_aux_loss(module, torch.device("cpu")).requires_grad


def test_mot_router_z_loss_uses_expert_axis():
    torch.manual_seed(0)
    module = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2, balance_loss_coeff=0.01).train()
    z_loss = module.router.router_z_loss(torch.randn(2, 32, 5, 7))
    expected = torch.log(torch.tensor(3.0)).square()
    assert torch.allclose(z_loss, expected, atol=1e-5)


def test_mot_temperature_anneal():
    module = C2fMoT(64, 64, n=2, num_heads=4)
    before = [m.router.temperature for m in module.m]
    anneal_mot_temperature(module, factor=0.5, min_temp=0.3)
    after = [m.router.temperature for m in module.m]
    assert after == [max(t * 0.5, 0.3) for t in before]


def test_mot_model_configs_parse():
    configs = [
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml",
        ROOT / "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-mot-n.yaml",
    ]
    for cfg in configs:
        model = DetectionModel(str(cfg), ch=3, nc=80, verbose=False)
        assert sum(isinstance(m, C2fMoT) for m in model.modules()) == 3
        assert sum(isinstance(m, MoTBlock) for m in model.modules()) == 6
