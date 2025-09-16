
import os
import torch
from torchvision import transforms
from train_and_visualize import SimpleMLP

def get_device():
    if torch.backends.mps.is_available():
        print('Using device: mps (Apple Silicon GPU)')
        return torch.device('mps')
    elif torch.cuda.is_available():
        print('Using device: cuda (NVIDIA GPU)')
        return torch.device('cuda')
    else:
        print('Using device: cpu')
        return torch.device('cpu')

def get_model(hidden_size=64, device=None, checkpoint_path='outputs/model.pth'):
    model = SimpleMLP(hidden_size=hidden_size)
    if device is None:
        device = get_device()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state', ckpt))
        print(f'Loaded checkpoint from {checkpoint_path}')
    else:
        print(f'Checkpoint {checkpoint_path} not found. Using untrained model.')
    model.to(device)
    return model

def get_normalize_transform():
    return transforms.Normalize((0.1307,), (0.3081,))

def get_norm_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        get_normalize_transform()
    ])

def get_raw_transform():
    return transforms.ToTensor()
