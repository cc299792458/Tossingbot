import torch
import numpy as np
import torch.nn as nn

def np_image_to_tensor(np_image, device):
    """
    Converts a numpy image array of shape (H, W, 3) to a PyTorch tensor of shape (1, 3, H, W).
    """
    # Convert numpy array to float32 and rearrange dimensions from (H, W, 3) to (1, 3, H, W)
    return torch.from_numpy(np_image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

def tensor_to_np_image(tensor):
    """
    Converts a PyTorch tensor of shape (1, 3, H, W) back to a numpy image array of shape (H, W, 3).
    """
    # Remove batch dimension and rearrange back to (H, W, 3), converting to float64
    return tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.float64)

def initialize_weights(model):
    """ Apply Xavier initialization to all Conv2d and Linear layers. """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)