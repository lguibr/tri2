# File: stats/tb_log_utils.py
import torch
import numpy as np
from typing import Union


def format_image_for_tb(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Formats an image (numpy or tensor) into CHW format for TensorBoard."""
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:  # HWC
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        elif image.ndim == 2:  # HW (grayscale)
            image_tensor = torch.from_numpy(image).unsqueeze(0)
        else:  # Assume CHW or other format, pass through
            image_tensor = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] not in [1, 3, 4]:  # Likely HWC
            if image.shape[-1] in [1, 3, 4]:
                image_tensor = image.permute(2, 0, 1)
            else:  # Unknown format, pass through
                image_tensor = image
        elif image.ndim == 2:  # HW
            image_tensor = image.unsqueeze(0)
        else:  # Assume CHW or other format
            image_tensor = image
    else:
        raise TypeError(f"Unsupported image type for TensorBoard: {type(image)}")

    # Ensure correct data type (e.g., uint8 or float) - TB handles this mostly
    return image_tensor
