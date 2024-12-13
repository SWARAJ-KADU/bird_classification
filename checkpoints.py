import torch
import os
# Function to save checkpoints
def save_checkpoint(state, filename="checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# Function to load checkpoints
def load_checkpoint(filename, model, optimizer, device='cuda'):
    if os.path.exists(filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        return model, optimizer, start_epoch
    else:
        print("=> No checkpoint found at", filename)
        return model, optimizer, 0  # Start from epoch 0 if no checkpoint is found