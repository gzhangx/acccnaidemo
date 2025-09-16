"""
Make an MP4 video from outputs/random_sample_0.png .. outputs/random_sample_9.png
"""
from pathlib import Path
import imageio

def main():
    out_dir = Path('outputs')
    imgs = []
    for i in range(10):
        p = out_dir / f'random_sample_{i}.png'
        if p.exists():
            imgs.append(imageio.v2.imread(str(p)))
        else:
            print('Missing', p)
    if not imgs:
        print('No images found')
        return
    out_video = out_dir / 'random_samples.mp4'
    imageio.mimwrite(str(out_video), imgs, fps=2, macro_block_size=None)
    print('Wrote', out_video)

if __name__ == '__main__':
    main()
