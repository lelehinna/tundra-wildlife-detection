"""Frozen ImageNet ResNet embeddings for candidate crops."""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torchvision

log = logging.getLogger(__name__)

# timm publishes torchvision-compatible ResNet weights as GitHub release assets.
_URLS = {"resnet18": "https://github.com/huggingface/pytorch-image-models/releases/download/"
                     "v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth"}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class CropEmbedder:
    def __init__(self, name: str = "resnet18", weights: str | None = "timm"):
        m = getattr(torchvision.models, name)(weights=None)
        if weights == "timm":
            cache = Path(os.environ.get("WILDCAM_CACHE", Path.home() / ".cache" / "wildcam"))
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / Path(_URLS[name]).name
            if not path.exists():
                log.info("downloading %s", _URLS[name])
                urllib.request.urlretrieve(_URLS[name], path)
            weights = str(path)
        if weights not in (None, "none"):
            state = {k: v for k, v in torch.load(weights, map_location="cpu", weights_only=True).items()
                     if not k.startswith("fc.")}
            m.load_state_dict(state, strict=False)
        else:
            log.warning("random backbone weights: verifier features will be weak")
        # Up to layer3: enough resolution for a 64 px crop, cheaper than the full network.
        self.net = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3).eval()
        self.dim = 256

    @torch.no_grad()
    def __call__(self, crops: np.ndarray, batch: int = 128) -> np.ndarray:
        """crops: (N, h, w) or (N, h, w, 3) uint8 -> (N, dim) float32."""
        if len(crops) == 0:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(np.ascontiguousarray(crops[i:i + batch])).float().div(255)
            x = x[..., None].repeat(1, 1, 1, 3) if x.ndim == 3 else x
            x = (x.permute(0, 3, 1, 2) - _MEAN) / _STD
            out.append(self.net(x).mean(dim=(2, 3)).numpy())
        return np.concatenate(out).astype(np.float32)
