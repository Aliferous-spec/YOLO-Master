from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess,pytest
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.dist import generate_ddp_file
class D:
 def __init__(self,p): self.args=SimpleNamespace(model='dummy.pt',save_dir='',resume=False);self.hub_session=None;self.resume=True;self.world_size=2;self.save_dir=Path(p)
def test_worker_uses_record(tmp_path):
 f=generate_ddp_file(D(tmp_path))
 try:
  c=Path(f).read_text();assert 'multiprocessing.errors import record' in c;assert '@record\ndef main():' in c;assert 'if __name__ == "__main__":\n    main()' in c;compile(c,f,'exec')
 finally: Path(f).unlink(missing_ok=True)
def test_parent_reraises_without_capture():
 t=object.__new__(BaseTrainer);t.ddp=True;t.args=SimpleNamespace(rect=False,batch=8);t.world_size=2;e=subprocess.CalledProcessError(7,['torchrun'])
 with patch('ultralytics.engine.trainer.generate_ddp_command',return_value=(['torchrun'],Path('x.py'))),patch('ultralytics.engine.trainer.subprocess.run',side_effect=e) as r,patch('ultralytics.engine.trainer.ddp_cleanup'),pytest.raises(subprocess.CalledProcessError):t.train()
 r.assert_called_once_with(['torchrun'],check=True)
