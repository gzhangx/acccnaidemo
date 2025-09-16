import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from torchvision import transforms

class SimpleMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out, h



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
