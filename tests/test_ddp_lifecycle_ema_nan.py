from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from ultralytics.engine.trainer import BaseTrainer


class E(nn.Module):
    def __init__(self, p=False):
        super().__init__()
        self.register_buffer("diagnostic", torch.tensor(1.0), persistent=p)


def tr(m):
    t = object.__new__(BaseTrainer)
    t.ema = SimpleNamespace(ema=m)
    t.world_size = 2
    return t


def recovery_trainer(tmp_path, loss=1.0, fitness=0.0, best_fitness=0.4):
    t = object.__new__(BaseTrainer)
    t.loss = torch.tensor(loss)
    t.fitness = fitness
    t.best_fitness = best_fitness
    t.start_epoch = 0
    t.device = torch.device("cpu")
    t.healthy = tmp_path / "last_healthy.pt"
    t.last = tmp_path / "last.pt"
    t.wdir = tmp_path
    t._gradient_nonfinite = False
    t.nan_recovery_attempts = 0
    t.model = nn.Linear(1, 1)
    t.scheduler = SimpleNamespace(last_epoch=0)
    t._model_train = MagicMock()
    t._load_checkpoint_state = MagicMock()
    return t


def write_healthy(path):
    torch.save({"ema": nn.Linear(1, 1), "optimizer": None, "scaler": None, "best_fitness": 0.4, "updates": 0}, path)


def test_nccl_skips_nonpersistent_cpu():
    t = tr(E(False))
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_backend", return_value="nccl"
    ), patch("torch.distributed.broadcast") as broadcast:
        t._sync_ema_buffers_for_validation()
    broadcast.assert_not_called()


def test_nccl_moves_persistent_cpu_buffer_before_broadcast():
    t = tr(E(True))
    t.device = torch.device("cpu")
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.get_backend", return_value="nccl"
    ), patch("torch.distributed.broadcast") as broadcast:
        t._sync_ema_buffers_for_validation()
    broadcast.assert_called_once()


def test_train_destroys_group_on_error():
    t = object.__new__(BaseTrainer)
    t.ddp = False
    t._do_train = MagicMock(side_effect=RuntimeError("boom"))
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.destroy_process_group") as destroy, pytest.raises(RuntimeError):
        t.train()
    destroy.assert_called_once_with()


def test_nonfinite_without_checkpoint_fails(tmp_path):
    t = recovery_trainer(tmp_path, loss=float("nan"))
    with pytest.raises(RuntimeError, match="without a healthy recovery checkpoint"):
        t._handle_nan_recovery(0)


def test_zero_fitness_is_not_nonfinite_or_recovery_trigger(tmp_path):
    t = recovery_trainer(tmp_path, loss=1.0, fitness=0.0, best_fitness=0.4)
    assert t._handle_nan_recovery(3) is False


def test_nonfinite_loss_recovers_from_healthy_checkpoint(tmp_path):
    t = recovery_trainer(tmp_path, loss=float("nan"))
    write_healthy(t.healthy)
    assert t._handle_nan_recovery(3) is True
    assert t.nan_recovery_attempts == 1
    t._load_checkpoint_state.assert_called_once()


def test_healthy_checkpoint_rejects_nonfinite_state_and_preserves_prior(tmp_path):
    t = object.__new__(BaseTrainer)
    t.healthy = tmp_path / "last_healthy.pt"
    t.healthy.write_bytes(b"known-good")
    buffer = __import__("io").BytesIO()
    torch.save({"tensor": torch.tensor(float("nan"))}, buffer)
    assert t._save_healthy_checkpoint(buffer.getvalue()) is False
    assert t.healthy.read_bytes() == b"known-good"
