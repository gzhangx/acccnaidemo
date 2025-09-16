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
from torchvision import datasets

from common_utils import get_normalize_transform


def denormalize(tensor, mean=0.1307, std=0.3081):
    return tensor * std + mean



def display_original_image(fig, gs_right, img_raw, label):
    ax_img = fig.add_subplot(gs_right[0, 0])
    img_show = img_raw.squeeze()
    ax_img.imshow(img_show, cmap='gray')
    ax_img.set_title(f'Original\n(label: {label})', fontsize=9)
    ax_img.axis('off')
    return ax_img

def createSqure(num_points=28):
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx, yy], axis=-1)  # shape (28, 28, 2)
    return coords

def display_line_connections(fig, gs, model, img_raw, h, probs, top_k_hidden):
    ax_center = fig.add_subplot(gs[0, 1])
    ax_center.axis('off')
    W1 = model.fc1.weight.detach().cpu().numpy()
    W2 = model.fc2.weight.detach().cpu().numpy()
    input_flat = img_raw.reshape(-1)
    n_input = input_flat.size
    n_hidden = W1.shape[0]
    n_out = W2.shape[0]
    xs = [0.0, 1.0, 2.0]
    def layer_pos(n, x):
        ys = np.linspace(-1, 1, n)
        xs_arr = np.full(n, x)
        return np.column_stack([xs_arr, ys])
    pos_in = createSqure()  # shape (28, 28, 2)
    pos_h = layer_pos(n_hidden, xs[1])
    pos_out = layer_pos(n_out, xs[2])
    # Lines from input to all hidden neurons
    segments_in_h = []
    colors_in_h = []
    linewidths_in_h = []
    h_min, h_max = np.min(h), np.max(h)
    h_range = h_max - h_min + 1e-12
    h_80 = np.percentile(h, 80)
    input_80 = np.percentile(input_flat, 80)
    min_pos = np.min(input_flat[input_flat > 0]) if np.any(input_flat > 0) else 0
    input_thresh = max(input_80, min_pos)
    for h_i in range(n_hidden):
        h_val = h[h_i]
        # Only add if activation is above 80th percentile
        if h_val < h_80:
            continue
        norm_h = (h_val - h_min) / h_range
        gray = 1.0 - norm_h
        color = (gray, gray, gray, 1.0)
        lw = 1 #0.5 + 2.5 * norm_h
        for i_i in range(n_input):
            if input_flat[i_i] < input_thresh:
                continue
            row = i_i // 28
            col = i_i % 28
            segments_in_h.append([pos_in[row, col], pos_h[h_i]])
            colors_in_h.append(color)
            linewidths_in_h.append(lw)
    # Lines from all hidden to output neurons
    segments_h_out = []
    colors_h_out = []
    linewidths_h_out = []
    probs_min, probs_max = np.min(probs), np.max(probs)
    probs_range = probs_max - probs_min + 1e-12
    for h_i in range(n_hidden):
        h_val = h[h_i]
        # Only add if activation is above 80th percentile
        if h_val < h_80:
            continue
        for o_i in range(n_out):
            prob_val = probs[o_i]
            norm_prob = (prob_val - probs_min) / probs_range
            # Color: grayscale, darker for higher prob
            color = (1.0 - norm_prob, 1.0 - norm_prob, 1.0 - norm_prob, 1.0)
            lw = 0.5 + 2.5 * norm_prob
            segments_h_out.append([pos_h[h_i], pos_out[o_i]])
            colors_h_out.append(color)
            linewidths_h_out.append(lw)
    from matplotlib.collections import LineCollection
    lc_in_h = LineCollection(segments_in_h, colors=colors_in_h, linewidths=linewidths_in_h, alpha=0.25)
    lc_h_out = LineCollection(segments_h_out, colors=colors_h_out, linewidths=linewidths_h_out, alpha=0.7)
    ax_center.add_collection(lc_in_h)
    ax_center.add_collection(lc_h_out)
    # Draw nodes
    pix_norm = (input_flat - input_flat.min()) / (input_flat.max() - input_flat.min() + 1e-12)
    # Marker size proportional to pixel value (normalized)
    marker_sizes = 20 #8 + 32 * pix_norm
    pos_in_flat = pos_in.reshape(-1, 2)
    ax_center.scatter(pos_in_flat[:,0], pos_in_flat[:,1], s=marker_sizes, c=pix_norm, cmap='gray', edgecolors='none')
    h_act = np.maximum(0, h)
    h_norm = (h_act - h_act.min()) / (h_act.max() - h_act.min() + 1e-12)
    ax_center.scatter(pos_h[:,0], pos_h[:,1], s=40, c=h_norm, cmap='viridis', edgecolors='k')
    out_probs = probs
    out_norm = (out_probs - out_probs.min()) / (out_probs.max() - out_probs.min() + 1e-12)
    base_size = 2
    max_scale = 220
    sizes = base_size + (out_probs * (max_scale - base_size))
    facecolors = np.array([[0.0, 0.0, 0.0, 1.0]] * n_out)
    edgecols = ['k'] * n_out
    pred = int(np.argmax(probs))
    sizes[pred] = sizes[pred] * 1.6
    edgecols[pred] = 'red'
    ax_center.scatter(pos_out[:,0], pos_out[:,1], s=sizes, facecolors=facecolors, edgecolors=edgecols, linewidths=0.9)
    ax_center.set_xlim(-0.5, 2.5)
    ax_center.set_ylim(-1.1, 1.1)
    ax_center.set_title('Input→Hidden (width/color∝h), Hidden→Output (width/color∝probs)')
    for oi in range(n_out):
        x, y = pos_out[oi]
        prob = probs[oi]
        label_str = f'{oi}: {prob:.2f}'
        if oi == pred:
            ax_center.text(x + 0.12, y, label_str, fontsize=8, color='red', fontweight='bold', va='center')
        else:
            ax_center.text(x + 0.12, y, label_str, fontsize=7, color='black', va='center')
    return ax_center

def display_probability_bar_graph(fig, gs_right, probs, pred):
    ax_out = fig.add_subplot(gs_right[1, 0])
    classes = list(range(probs.shape[0]))
    bars = ax_out.bar(classes, probs, color='gray')
    bars[pred].set_color('red')
    ax_out.set_xticks(classes)
    ax_out.set_xlabel('Class', fontsize=7)
    ax_out.set_ylabel('Probability', fontsize=7)
    ax_out.tick_params(axis='x', labelsize=6)
    ax_out.tick_params(axis='y', labelsize=6)
    ax_out.set_title(f'Probs (pred={pred})', fontsize=8)
    return ax_out

def visualize_sample(model, device, img_raw_ten, label, out_path, top_k_hidden=9):
    norm_tran = get_normalize_transform()
    img_raw = img_raw_ten.numpy()
    img_norm = norm_tran(img_raw_ten)
    model.eval()
    with torch.no_grad():
        inp = img_norm.unsqueeze(0).to(device)
        out, h = model(inp)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        h = h.cpu().numpy()[0]
    pred = int(np.argmax(probs))
    cols = 1 + max(1, int(math.ceil(math.sqrt(top_k_hidden)))) + 1
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.06, 6.5, 0.44], wspace=0.35)
    gs_right = gs[0, 2].subgridspec(2, 1, height_ratios=[0.35, 0.65], hspace=0.04)
    display_original_image(fig, gs_right, img_raw, label)
    display_line_connections(fig, gs, model, img_raw, h, probs, top_k_hidden)
    display_probability_bar_graph(fig, gs_right, probs, pred)
    plt.suptitle('ACCCN Sunday school prob demo')
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92)
    os.makedirs(Path(out_path).parent, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved visualization to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/model.pth', help='path to model checkpoint')
    parser.add_argument('--top-hidden', type=int, default=9)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--index', type=int, default=None, help='optionally specify a test index instead of random')
    parser.add_argument('--output', type=str, default='outputs/random_sample.png')
    args = parser.parse_args()

    from common_utils import get_device, get_model, get_raw_transform
    device = get_device()
    raw_transform = get_raw_transform()
    #norm_transform = get_norm_transform()
    test_raw = datasets.MNIST('.', train=False, download=True, transform=raw_transform)
    #test_norm = datasets.MNIST('.', train=False, download=True, transform=norm_transform)

    if args.seed is not None:
        random.seed(args.seed)

    if args.index is None:
        idx = random.randrange(len(test_raw))
    else:
        idx = args.index

    img_raw, label = test_raw[idx]
    #img_norm, _ = test_norm[idx]

    print("Using index", idx)
    model = get_model()
    visualize_sample(model, device, img_raw, label, args.output, top_k_hidden=10)


if __name__ == '__main__':
    main()
