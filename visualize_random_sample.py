"""
Visualize a random MNIST test sample:
- Show the original image (left)
- Show top-K hidden neurons by activation (center) with their weight receptive fields and activation values
- Show output probabilities and highlight the predicted class (right)

Saves output to outputs/random_sample.png
"""
import os
import argparse
import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Import model architecture from train script
from train_and_visualize import SimpleMLP


def denormalize(tensor, mean=0.1307, std=0.3081):
    return tensor * std + mean


def visualize_sample(model, device, img_raw, img_norm, label, out_path, top_k_hidden=9):
    # img_raw: (1,28,28) numpy or torch tensor in [0,1]
    # img_norm: normalized tensor for model input (1,1,28,28)
    model.eval()
    with torch.no_grad():
        inp = img_norm.unsqueeze(0).to(device)
        out, h = model(inp)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        h = h.cpu().numpy()[0]  # hidden activations

    pred = int(np.argmax(probs))

    # get fc1 weights
    W1 = model.fc1.weight.detach().cpu().numpy()  # shape (hidden, input)

    # select top-K hidden neurons by activation for this sample
    topk_idx = np.argsort(h)[::-1][:top_k_hidden]
    topk_vals = h[topk_idx]

    # prepare figure: 3 columns: left image (small), center wide, right (small)
    cols = 1 + max(1, int(math.ceil(math.sqrt(top_k_hidden)))) + 1
    fig = plt.figure(figsize=(14, 6))
    # make the side panels small and center wide; right column will be split vertically
    # increase horizontal spacing to avoid axes overlapping
    gs = fig.add_gridspec(1, 3, width_ratios=[0.06, 6.5, 0.44], wspace=0.35)

    # center: connection graph showing top-K edges per stage (input->hidden, hidden->output)
    ax_center = fig.add_subplot(gs[0, 1])
    ax_center.axis('off')

    # prepare weights and shapes
    W2 = model.fc2.weight.detach().cpu().numpy()  # shape (out, hidden)
    b2 = model.fc2.bias.detach().cpu().numpy()

    input_flat = img_raw.reshape(-1)
    n_input = input_flat.size
    n_hidden = W1.shape[0]
    n_out = W2.shape[0]

    # node positions
    xs = [0.0, 1.0, 2.0]
    def layer_pos(n, x):
        ys = np.linspace(-1, 1, n)
        xs = np.full(n, x)
        return np.column_stack([xs, ys])

    pos_in = layer_pos(n_input, xs[0])
    pos_h = layer_pos(n_hidden, xs[1])
    pos_out_positions = layer_pos(n_out, xs[2])
    # layer_pos places nodes top->bottom as increasing y from -1 to 1; to have label 0 at top,
    # reverse the positions so index 0 maps to the top position
    pos_out_positions_flipped = pos_out_positions[::-1]
    pos_out_map = {label: tuple(pos) for label, pos in zip(range(n_out), pos_out_positions_flipped)}

    segments = []
    colors = []
    linewidths = []

    # select global top-K input->hidden edges using per-sample contribution = |w * input_pixel|
    # input_flat is in [0,1]
    contrib1 = np.abs(W1 * input_flat[np.newaxis, :])
    flat1 = contrib1.flatten()
    k1 = min(top_k_hidden, flat1.size)
    top_idx1 = np.argsort(flat1)[::-1][:k1]
    vals1 = flat1[top_idx1]
    vmin1 = vals1.min() if vals1.size > 0 else 0.0
    vmax1 = vals1.max() if vals1.size > 0 else 1.0
    h_idx1, i_idx1 = np.unravel_index(top_idx1, contrib1.shape)
    for h_i, i_i, val in zip(h_idx1, i_idx1, vals1):
        segments.append([pos_in[i_i], pos_h[h_i]])
        colors.append('k')
        norm = (val - vmin1) / (vmax1 - vmin1 + 1e-12)
        linewidths.append(0.8 + 4.0 * norm)

    # select global top-K hidden->output edges using per-sample contribution = |w * hidden_activation|
    # h contains the hidden activations for this sample
    contrib2 = np.abs(W2 * h[np.newaxis, :])
    flat2 = contrib2.flatten()
    k2 = min(top_k_hidden, flat2.size)
    top_idx2 = np.argsort(flat2)[::-1][:k2]
    vals2 = flat2[top_idx2]
    vmin2 = vals2.min() if vals2.size > 0 else 0.0
    vmax2 = vals2.max() if vals2.size > 0 else 1.0
    o_idx2, h_idx2 = np.unravel_index(top_idx2, contrib2.shape)
    for o_i, h_i, val in zip(o_idx2, h_idx2, vals2):
        # map output index (class label) to its ordered position
        segments.append([pos_h[h_i], pos_out_map[o_i]])
        colors.append('k')
        norm = (val - vmin2) / (vmax2 - vmin2 + 1e-12)
        linewidths.append(0.8 + 4.0 * norm)

    # add edges
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.8)
    # ensure edges are clipped to the center axes (avoid drawing over other axes)
    lc.set_clip_on(True)
    ax_center.add_collection(lc)

    # draw nodes: inputs (small, show as grayscale), hidden (by activation), outputs (by probs)
    pix_norm = (input_flat - input_flat.min()) / (input_flat.max() - input_flat.min() + 1e-12)
    ax_center.scatter(pos_in[:,0], pos_in[:,1], s=8, c=pix_norm, cmap='gray', edgecolors='none')

    h_act = np.maximum(0, h)
    h_norm = (h_act - h_act.min()) / (h_act.max() - h_act.min() + 1e-12)
    ax_center.scatter(pos_h[:,0], pos_h[:,1], s=40, c=h_norm, cmap='viridis', edgecolors='k')

    # outputs: color by prob, highlight prediction
    out_probs = probs
    out_norm = (out_probs - out_probs.min()) / (out_probs.max() - out_probs.min() + 1e-12)
    # construct ordered array of output positions matching label order 0..n_out-1
    pos_out_arr = np.vstack([pos_out_map[i] for i in range(n_out)])
    # scale sizes by probability (small base + scaled)
    base_size = 2
    max_scale = 220
    sizes = base_size + (out_probs * (max_scale - base_size))
    print('Output node sizes:', sizes, out_probs)
    # compute facecolors from the colormap for filled markers
    # use solid black facecolors for output nodes
    facecolors = np.array([[0.0, 0.0, 0.0, 1.0]] * n_out)
    edgecols = ['k'] * n_out
    # emphasize predicted class (bigger size and red edge)
    sizes[pred] = sizes[pred] * 1.6
    edgecols[pred] = 'red'
    ax_center.scatter(pos_out_arr[:,0], pos_out_arr[:,1], s=sizes, facecolors=facecolors, edgecolors=edgecols, linewidths=0.9)

    ax_center.set_xlim(-0.5, 2.5)
    ax_center.set_ylim(-1.1, 1.1)
    ax_center.set_title(f'Top connections (K={top_k_hidden} per stage)')

    # label each output node with class id and probability
    for oi in range(n_out):
        x, y = pos_out_arr[oi]
        prob = probs[oi]
        label_str = f'{oi}: {prob:.2f}'
        if oi == pred:
            ax_center.text(x + 0.12, y, label_str, fontsize=8, color='red', fontweight='bold', va='center')
        else:
            ax_center.text(x + 0.12, y, label_str, fontsize=7, color='black', va='center')

    # right: split into two rows: top small original image, bottom probabilities
    # make the top (image) smaller and the bottom (probs) compact
    gs_right = gs[0, 2].subgridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.04)
    ax_img = fig.add_subplot(gs_right[0, 0])
    ax_out = fig.add_subplot(gs_right[1, 0])
    # show original image above probs
    img_show = img_raw.squeeze()
    ax_img.imshow(img_show, cmap='gray')
    ax_img.set_title(f'Original\n(label: {label})', fontsize=9)
    ax_img.axis('off')
    classes = list(range(probs.shape[0]))
    bars = ax_out.bar(classes, probs, color='gray')
    bars[pred].set_color('red')
    ax_out.set_xticks(classes)
    ax_out.set_xlabel('Class', fontsize=7)
    ax_out.set_ylabel('Probability', fontsize=7)
    ax_out.tick_params(axis='x', labelsize=6)
    ax_out.tick_params(axis='y', labelsize=6)
    ax_out.set_title(f'Probs (pred={pred})', fontsize=8)

    plt.suptitle('ACCCN Sunday school prob demo')
    # use manual subplots_adjust rather than tight_layout to avoid compatibility warnings
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92)
    os.makedirs(Path(out_path).parent, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved visualization to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/model.pth')
    parser.add_argument('--top-hidden', type=int, default=9)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--index', type=int, default=None, help='optionally specify a test index instead of random')
    parser.add_argument('--output', type=str, default='outputs/random_sample.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load datasets
    raw_transform = transforms.ToTensor()
    norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_raw = datasets.MNIST('.', train=False, download=True, transform=raw_transform)
    test_norm = datasets.MNIST('.', train=False, download=True, transform=norm_transform)

    if args.seed is not None:
        random.seed(args.seed)

    if args.index is None:
        idx = random.randrange(len(test_raw))
    else:
        idx = args.index

    img_raw, label = test_raw[idx]
    img_norm, _ = test_norm[idx]

    # load model
    model = SimpleMLP(hidden_size=64)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.to(device)

    visualize_sample(model, device, img_raw.numpy(), img_norm, label, args.output, top_k_hidden=args.top_hidden)


if __name__ == '__main__':
    main()
