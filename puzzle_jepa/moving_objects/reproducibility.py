from __future__ import annotations

import torch


def configure_reproducibility(deterministic: bool = True) -> None:
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(not deterministic)
        torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
        torch.backends.cuda.enable_math_sdp(True)
