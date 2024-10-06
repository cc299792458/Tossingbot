import os
import torch
import numpy as np
import torch.nn as nn

def np_image_to_tensor(np_image, device):
    """
    Converts a numpy image array of shape (B, H, W, 3) to a PyTorch tensor of shape (B, 3, H, W).
    """
    # Convert numpy array to float32 and rearrange dimensions from (H, W, 3) to (1, 3, H, W)
    return torch.from_numpy(np_image.astype(np.float32)).permute(0, 3, 1, 2).to(device)

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

def save_model(agent, optimizer, episode, log_dir, model_name='agent_checkpoint'):
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(log_dir, model_name + '.pth')
    torch.save({
        'episode': episode,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"Model saved at {model_path}")

def load_model(agent, optimizer, log_dir, model_name='agent_checkpoint'):
    # Load the model if exists
    model_path = os.path.join(log_dir, model_name + '.pth')
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, weights_only=True)
        agent.load_state_dict(checkpoint['model_state_dict'])
        
        # Only load optimizer if it's not None
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        episode = checkpoint['episode']
        print(f"Model loaded from {model_path}, resuming from episode {episode}")
        return episode
    else:
        print("No checkpoint found, starting from scratch.")
        return 0