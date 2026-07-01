"""Runtime compatibility hooks for running UDIS2 through this repository.

Python imports this module automatically when its directory is on PYTHONPATH.
The wrapper uses it to keep the external UDIS2 checkout unmodified.
"""

from __future__ import annotations

import os


if os.environ.get("UDIS2_FORCE_CPU_LOAD") == "1":
    import torch

    _original_torch_load = torch.load

    def _torch_load_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _torch_load_cpu
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
