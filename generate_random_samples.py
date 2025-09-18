"""
Generate one random visualization per MNIST label (0..9) using visualzie_random_sample.visualize_sample
Saves files: outputs/random_sample_<label>.png
"""
from pathlib import Path
import random
import torch
from torchvision import datasets, transforms

from visualize_random_sample import visualize_sample
from common_utils import SimpleMLP
import imageio

def main():
    from common_utils import get_device, get_model, get_raw_transform
    device = get_device()
    raw_transform = get_raw_transform()    
    test_raw = datasets.MNIST('.', train=False, download=True, transform=raw_transform)    
    model = get_model()

    # for each label, choose a random index with that label
    labels_to_indices = {l: [] for l in range(10)}
    for i, (_, lbl) in enumerate(test_raw):
        labels_to_indices[lbl].append(i)


    video_out_dir = Path('outputs')
    out_dir = Path(video_out_dir / 'imgs')
    out_dir.mkdir(exist_ok=True)

    
    imgs = []
    for lbl in range(10):
        idx = random.choice(labels_to_indices[lbl])
        img_raw, _ = test_raw[idx]
        out_path = out_dir / f'random_sample_{idx}_{lbl}.png'
        visualize_sample(model, device, img_raw, lbl, out_path, top_k_hidden=20)
        print('Wrote', out_path)
        imgs.append(imageio.v2.imread(str(out_path)))


    for idx in range(100):        
        img_raw, lbl = test_raw[idx]
        out_path = out_dir / f'random_sample_{idx}_{lbl}.png'
        visualize_sample(model, device, img_raw, lbl, out_path, top_k_hidden=20)
        print('Wrote', out_path)
        imgs.append(imageio.v2.imread(str(out_path)))

    out_video = video_out_dir / 'random_samples.mp4'
    print('writing', out_video)
    imageio.mimwrite(str(out_video), imgs, fps=2, macro_block_size=None)
    print('Wrote', out_video)

if __name__ == '__main__':
    main()
