"""Work-around for the broken GPU<->GPU peer path on the TITAN Xp nodes (s-003/s-006).

There, a cuda:i -> cuda:j copy costs ~50 ms even for 20 KB (~100 ms for 12 MB), vs
~0.1 ms / ~5 ms when staged through host memory (judge_rerun/titan_diag3.py). A
device_map="auto" model split over several GPUs does such hops constantly, so
decoding crawls. ``install()`` makes Tensor.to(<other cuda device>) go via the CPU.
"""
import torch

_orig_to = torch.Tensor.to
_installed = False


def _target_cuda_index(args, kwargs):
    dev = kwargs.get("device", args[0] if args else None)
    if isinstance(dev, int):
        return dev
    if isinstance(dev, str) and dev.startswith("cuda"):
        return int(dev.split(":")[1]) if ":" in dev else torch.cuda.current_device()
    if isinstance(dev, torch.device) and dev.type == "cuda":
        return dev.index if dev.index is not None else torch.cuda.current_device()
    return None


def _to(self, *args, **kwargs):
    if self.is_cuda:
        tgt = _target_cuda_index(args, kwargs)
        if tgt is not None and tgt != self.device.index:
            return _orig_to(_orig_to(self, "cpu"), *args, **kwargs)
    return _orig_to(self, *args, **kwargs)


def install() -> None:
    global _installed
    if not _installed:
        torch.Tensor.to = _to
        _installed = True
