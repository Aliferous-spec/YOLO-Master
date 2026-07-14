from unittest.mock import patch
import torch
from ultralytics.nn.modules.moa.moa import _moa_router_aux_loss
from ultralytics.nn.modules.mot.mot import differentiable_balance_loss
def test_moa_local_gradient_global_value_formula():
 logits=torch.tensor([[[[.4]],[[.1]]]],requires_grad=True);w=logits.softmax(1)
 # world=2, synthetic global sum adds detached remote [0.2,0.8], count=2
 def reduce(t,op=None):
  if t.numel()==2:t.add_(torch.tensor([.2,.8]))
  else:t.add_(1.)
 with patch('torch.distributed.is_initialized',return_value=True),patch('torch.distributed.get_world_size',return_value=2),patch('torch.distributed.get_backend',return_value='gloo'),patch('torch.distributed.all_reduce',side_effect=reduce):
  loss=_moa_router_aux_loss(w,logits,1.);loss.backward()
 assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
def test_mot_only_reduces_detached_usage():
 p=torch.tensor([[.7,.3]],requires_grad=True);u=torch.tensor([1.,0.])
 with patch('ultralytics.nn.modules.mot.mot.all_reduce_mean',side_effect=lambda x:torch.tensor([.25,.75])) as r:
  differentiable_balance_loss(p,u,2,reduce_ddp=True).backward()
 r.assert_called_once();assert torch.isfinite(p.grad).all()
def test_eval_local_only():
 with torch.no_grad():
  p=torch.tensor([[.7,.3]]); differentiable_balance_loss(p,torch.tensor([1.,0.]),2,reduce_ddp=False)
