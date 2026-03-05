from typing import Any, Dict

from torchvision import transforms

from .base import BaseMapper
from .mappers_config import (
    KeyRenameMapperConfig,
    RescaleMapperConfig,
    TorchvisionMapperConfig,
    ResolutionBucketMapperConfig,
    ResolutionResizeMapperConfig,
)
import numpy as np
import torch.nn.functional as F


class KeyRenameMapper(BaseMapper):
    """
    Rename keys in a sample according to a key map

    Args:

        config (KeyRenameMapperConfig): Configuration for the mapper

    Examples
    ########

    1. Rename keys in a sample according to a key map

    .. code-block:: python

        from cr.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"}
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

    2. Rename keys in a sample according to a key map and a condition key

    .. code-block:: python

        from cr.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"},
            condition_key="condition",
            condition_fn=lambda x: x == 1
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1, "condition": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

        sample = {"old_key": 1, "condition": 0}
        new_sample = mapper(sample)
        print(new_sample)  # {"old_key": 1}

    ```
    """

    def __init__(self, config: KeyRenameMapperConfig):
        super().__init__(config)
        self.key_map = config.key_map
        self.condition_key = config.condition_key
        self.condition_fn = config.condition_fn
        self.else_key_map = config.else_key_map

    def __call__(self, batch: Dict[str, Any], *args, **kwrags):
        if self.condition_key is not None:
            condition_key = batch[self.condition_key]
            if self.condition_fn(condition_key):
                for old_key, new_key in self.key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

            elif self.else_key_map is not None:
                for old_key, new_key in self.else_key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

        else:
            for old_key, new_key in self.key_map.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)
        return batch


class TorchvisionMapper(BaseMapper):
    """
    Apply torchvision transforms to a sample

    Args:

        config (TorchvisionMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: TorchvisionMapperConfig):
        super().__init__(config)
        chained_transforms = []
        for transform, kwargs in zip(config.transforms, config.transforms_kwargs):
            transform = getattr(transforms, transform)
            chained_transforms.append(transform(**kwargs))
        self.transforms = transforms.Compose(chained_transforms)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if self.key in batch:
            batch[self.output_key] = self.transforms(batch[self.key])
        return batch


class RescaleMapper(BaseMapper):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        config (RescaleMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: RescaleMapperConfig):
        super().__init__(config)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if isinstance(batch[self.key], list):
            tmp = []
            for i, image in enumerate(batch[self.key]):
                tmp.append(2 * image - 1)
            batch[self.output_key] = tmp
        else:
            batch[self.output_key] = 2 * batch[self.key] - 1
        return batch


class ResolutionBucketMapper(BaseMapper):
    """
    Determines the target resolution bucket for a sample based on its aspect ratio and a sampled budget.
    """

    def __init__(self, config: ResolutionBucketMapperConfig):
        super().__init__(config)
        self.budgets = config.budgets
        self.probabilities = config.probabilities
        self.min_ar = config.min_ar
        self.max_ar = config.max_ar
        self.resolutions = self._generate_all_resolutions()

    def _generate_all_resolutions(self):
        # Base aspect ratios to consider (matches SDXL/LBM logic)
        ar_list = [0.25, 0.33, 0.5, 0.66, 1.0, 1.5, 2.0, 3.0, 4.0]
        all_res = {}
        for budget in self.budgets:
            res_for_budget = []
            for ar in ar_list:
                if ar < self.min_ar or ar > self.max_ar:
                    continue
                h = np.sqrt(budget / ar)
                w = h * ar
                h = int(round(h / 64) * 64)
                w = int(round(w / 64) * 64)
                res_for_budget.append((h, w))
            all_res[budget] = sorted(list(set(res_for_budget)))
        return all_res

    def __call__(self, batch: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        # Sample a budget
        budget = np.random.choice(self.budgets, p=self.probabilities)
        
        # Get current aspect ratio from image (assuming PIL or Tensor)
        image = batch[self.key]
        if hasattr(image, "size"): # PIL
            w_orig, h_orig = image.size
        else: # Tensor BHWC or CHW
            h_orig, w_orig = image.shape[-2:]
        
        ar_orig = w_orig / h_orig
        
        # Find closest resolution in the chosen budget
        best_res = min(self.resolutions[budget], key=lambda x: abs((x[1]/x[0]) - ar_orig))
        
        batch["target_h"] = best_res[0]
        batch["target_w"] = best_res[1]
        batch["resolution_bucket"] = f"{best_res[0]}x{best_res[1]}"
        
        return batch


class ResolutionResizeMapper(BaseMapper):
    """
    Resizes images to the resolution specified by 'target_h' and 'target_w' in the batch.
    """

    def __init__(self, config: ResolutionResizeMapperConfig):
        super().__init__(config)
        self.interpolation = getattr(InterpolationMode, config.interpolation) if isinstance(config.interpolation, str) else config.interpolation

    def __call__(self, batch: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        if "target_h" not in batch or "target_w" not in batch:
            return batch
            
        target_size = (batch["target_h"], batch["target_w"])
        
        image = batch[self.key]
        if hasattr(image, "resize"): # PIL
            batch[self.output_key] = image.resize((target_size[1], target_size[0]), resample=self._get_pil_resample())
        elif isinstance(image, torch.Tensor): # Tensor
            # F.interpolate expects (N, C, H, W)
            if image.ndim == 3:
                img_in = image.unsqueeze(0)
                out = F.interpolate(img_in, size=target_size, mode=self._get_torch_mode(), align_corners=False)
                batch[self.output_key] = out.squeeze(0)
            else:
                batch[self.output_key] = F.interpolate(image, size=target_size, mode=self._get_torch_mode(), align_corners=False)
        
        return batch

    def _get_pil_resample(self):
        from PIL import Image
        mapping = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        return mapping.get(str(self.interpolation).lower().split(".")[-1], Image.BILINEAR)

    def _get_torch_mode(self):
        mode = str(self.interpolation).lower().split(".")[-1]
        if "nearest" in mode: return "nearest"
        if "bilinear" in mode: return "bilinear"
        if "bicubic" in mode: return "bicubic"
        return "bilinear"
