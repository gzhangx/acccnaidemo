"""
Generate one random visualization per MNIST label (0..9) using visualzie_random_sample.visualize_sample
Saves files: outputs/random_sample_<label>.png
"""
from pathlib import Path
import random
import torch
from torchvision import datasets, transforms

from visualize_random_sample import visualize_sample
from train_and_visualize import SimpleMLP
import imageio

def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using device: mps (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: cuda (NVIDIA GPU)')
    else:
        device = torch.device('cpu')
        print('Using device: cpu')
    print('Using device:', device)

    raw_transform = transforms.ToTensor()
    norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_raw = datasets.MNIST('.', train=False, download=True, transform=raw_transform)
    test_norm = datasets.MNIST('.', train=False, download=True, transform=norm_transform)

    # load model
    model = SimpleMLP(hidden_size=64)
    ckpt = torch.load('outputs/model.pth', map_location=device)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.to(device)

    # for each label, choose a random index with that label
    labels_to_indices = {l: [] for l in range(10)}
    for i, (_, lbl) in enumerate(test_raw):
        labels_to_indices[lbl].append(i)

    Path('outputs').mkdir(exist_ok=True)

    out_dir = Path('outputs')
    imgs = []
    for lbl in range(10):
        idx = random.choice(labels_to_indices[lbl])
        img_raw, _ = test_raw[idx]
        img_norm, _ = test_norm[idx]
        out_path = f'outputs/random_sample_{lbl}.png'
        visualize_sample(model, device, img_raw.numpy(), img_norm, lbl, out_path, top_k_hidden=20)
        print('Wrote', out_path)
        imgs.append(imageio.v2.imread(str(out_path)))

    out_video = out_dir / 'random_samples.mp4'
    print('writing', out_video)
    imageio.mimwrite(str(out_video), imgs, fps=2, macro_block_size=None)
    print('Wrote', out_video)

if __name__ == '__main__':
    main()
