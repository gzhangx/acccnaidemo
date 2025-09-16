"""
Train a small MLP on MNIST and visualize neuron activations.

Outputs:
- activations.png: Heatmap of activations (neurons x inputs)
- activation_examples/: per-neuron top activating input grids
"""
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

showTopEdges = 20


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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.4f}")


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def collect_activations(model, device, data_loader, max_batches=1):
    model.eval()
    activations = []
    inputs = []
    labels = []
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            output, h = model(data)
            activations.append(h.cpu().numpy())
            inputs.append(data.cpu().numpy())
            labels.append(target.numpy())
            if i + 1 >= max_batches:
                break
    activations = np.concatenate(activations, axis=0)
    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return activations, inputs, labels


def plot_activation_heatmap(activations, out_path, max_neurons=64, max_inputs=64):
    # activations: (N_inputs, N_neurons)
    N_inputs, N_neurons = activations.shape
    n_neurons = min(N_neurons, max_neurons)
    n_inputs = min(N_inputs, max_inputs)
    arr = activations[:n_inputs, :n_neurons].T  # neurons x inputs

    plt.figure(figsize=(12, 8))
    sns.heatmap(arr, cmap='viridis')
    plt.xlabel('Inputs')
    plt.ylabel('Neurons')
    plt.title('Neuron activations (neurons x inputs)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_per_neuron_examples(activations, inputs, labels, out_dir, top_k=9):
    # activations: (N_inputs, N_neurons), inputs: (N_inputs, 1, 28, 28)
    N_inputs, N_neurons = activations.shape
    os.makedirs(out_dir, exist_ok=True)
    flat_inputs = inputs.reshape(N_inputs, 28, 28)

    for neuron in range(N_neurons):
        # get top-k indices that maximize activation for this neuron
        idx = np.argsort(activations[:, neuron])[::-1][:top_k]
        imgs = flat_inputs[idx]
        lbls = labels[idx]

        cols = int(np.sqrt(top_k))
        rows = (top_k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = axes.flatten()
        for i in range(top_k):
            ax = axes[i]
            ax.imshow(imgs[i], cmap='gray')
            ax.set_title(str(lbls[i]))
            ax.axis('off')
        for i in range(top_k, len(axes)):
            axes[i].axis('off')
        plt.suptitle(f'Neuron {neuron} top {top_k} activations')
        plt.tight_layout()
        fname = os.path.join(out_dir, f'neuron_{neuron:03d}.png')
        fig.savefig(fname)
        plt.close(fig)


def visualize_input_graph(model, input_img, input_activations, out_path, top_k_edges=10):
    """
    Draw a simple layered graph: input pixels (flattened) -> hidden neurons -> output neurons.
    - input_img: (1,28,28) numpy array
    - input_activations: activations for hidden layer for that input (N_neurons,)
    - top_k_edges: number of strongest incoming edges to draw per target neuron
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    # extract weights from model
    # assume model has attributes fc1 and fc2
    W1 = model.fc1.weight.detach().cpu().numpy()  # shape (hidden, input)
    b1 = model.fc1.bias.detach().cpu().numpy()
    W2 = model.fc2.weight.detach().cpu().numpy()  # shape (out, hidden)
    b2 = model.fc2.bias.detach().cpu().numpy()

    input_flat = input_img.reshape(-1)
    n_input = input_flat.shape[0]
    n_hidden = W1.shape[0]
    n_out = W2.shape[0]

    # layout positions: x coordinates for each layer
    layer_x = [0.0, 1.0, 2.0]
    # spread nodes vertically
    def layer_positions(n, x):
        ys = np.linspace(-1, 1, n)
        xs = np.full(n, x)
        return np.column_stack([xs, ys])

    pos_in = layer_positions(n_input, layer_x[0])
    pos_h = layer_positions(n_hidden, layer_x[1])
    pos_out = layer_positions(n_out, layer_x[2])

    fig, ax = plt.subplots(figsize=(12, 8))

    segments = []
    colors = []
    linewidths = []

    # draw strongest edges into hidden neurons from input (global top-k across all input->hidden weights)
    absW1 = np.abs(W1)  # shape (hidden, input)
    flat1 = absW1.flatten()
    k1 = int(top_k_edges)
    k1 = min(k1, flat1.size)
    top_flat_idx1 = np.argsort(flat1)[::-1][:k1]
    # convert flat indices to (h, i)
    h_idx1, i_idx1 = np.unravel_index(top_flat_idx1, absW1.shape)
    for h, i in zip(h_idx1, i_idx1):
        seg = [pos_in[i], pos_h[h]]
        segments.append(seg)
        weight = W1[h, i]
        colors.append('k')
        # normalize linewidth by the max of selected weights to keep scale stable
        linewidths.append(0.5 + 3.0 * (abs(weight) / (flat1[top_flat_idx1][0] + 1e-12)))

    # draw strongest edges from hidden to outputs (global top-k across hidden->output weights)
    absW2 = np.abs(W2)  # shape (out, hidden)
    flat2 = absW2.flatten()
    k2 = int(top_k_edges)
    k2 = min(k2, flat2.size)
    top_flat_idx2 = np.argsort(flat2)[::-1][:k2]
    o_idx2, h_idx2 = np.unravel_index(top_flat_idx2, absW2.shape)
    for o, h in zip(o_idx2, h_idx2):
        seg = [pos_h[h], pos_out[o]]
        segments.append(seg)
        weight = W2[o, h]
        colors.append('k')
        linewidths.append(0.5 + 3.0 * (abs(weight) / (flat2[top_flat_idx2][0] + 1e-12)))

    lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.7)
    ax.add_collection(lc)

    # draw nodes: size/color by activation
    # input nodes colored by pixel intensity (normalized)
    pix = input_flat
    pix_norm = (pix - pix.min()) / (pix.max() - pix.min() + 1e-12)
    ax.scatter(pos_in[:, 0], pos_in[:, 1], s=10, c=pix_norm, cmap='gray', edgecolors='none')

    # hidden nodes colored by activation (ReLU)
    h_act = np.maximum(0, input_activations)
    h_norm = (h_act - h_act.min()) / (h_act.max() - h_act.min() + 1e-12)
    ax.scatter(pos_h[:, 0], pos_h[:, 1], s=50, c=h_norm, cmap='viridis', edgecolors='k')

    # output nodes: show softmax scores magnitude (use weights*activations)
    # compute pseudo-output activations
    out_act = W2.dot(h_act) + b2
    out_norm = (out_act - out_act.min()) / (out_act.max() - out_act.min() + 1e-12)
    ax.scatter(pos_out[:, 0], pos_out[:, 1], s=80, c=out_norm, cmap='plasma', edgecolors='k')

    # labels
    for i in range(n_input):
        if n_input <= 100:
            ax.text(pos_in[i, 0] - 0.05, pos_in[i, 1], str(i), fontsize=6)
            # avoid labeling many input pixels
            pass
    for h in range(n_hidden):
        ax.text(pos_h[h, 0], pos_h[h, 1], str(h), fontsize=6, va='center')
    for o in range(n_out):
        ax.text(pos_out[o, 0] + 0.02, pos_out[o, 1], str(o), fontsize=8, va='center')

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    plt.title('Network activations and strongest weights (darker/thicker = larger magnitude)')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--quick', action='store_true', help='run quick mode (less data)')
    parser.add_argument('--input-graph', type=int, default=None, help='index of collected input to draw network graph for')
    parser.add_argument('--save-path', type=str, default=None, help='path to save model checkpoint (after training)')
    parser.add_argument('--load-path', type=str, default=None, help='path to load model checkpoint before evaluation/visualization')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    if args.quick:
        # smaller datasets for quick runs
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(2048)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(512)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = SimpleMLP(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Optionally load a checkpoint
    if args.save_path is not None:
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state', ckpt))
        if 'optimizer_state' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                # optimizer state may be incompatible across PyTorch versions
                print('Loaded model weights; optimizer state not restored')
        print('Loaded checkpoint from', args.save_path)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = evaluate(model, device, test_loader)
        print(f'Epoch {epoch} accuracy: {acc * 100:.2f}%')

    # Optionally save a checkpoint
    if args.save_path is not None:
        ckpt = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'args': vars(args)}
        torch.save(ckpt, args.save_path)
        print('Saved checkpoint to', args.save_path)

    # collect activations for a few batches
    activations, inputs, labels = collect_activations(model, device, test_loader, max_batches=4)
    # activations: (N_inputs, N_neurons)

    heatmap_path = out_dir / 'activations.png'
    plot_activation_heatmap(activations, heatmap_path, max_neurons=min(256, activations.shape[1]), max_inputs=activations.shape[0])

    examples_dir = out_dir / 'activation_examples'
    save_per_neuron_examples(activations, inputs, labels, examples_dir, top_k=9)

    # optional: visualize connections for a single input
    if args.input_graph is not None:
        # args.input_graph is an index into the collected inputs (0..N-1)
        idx = int(args.input_graph)
        if idx < 0 or idx >= inputs.shape[0]:
            print(f"input_graph index {idx} out of range (0..{inputs.shape[0]-1})")
        else:
            graph_path = out_dir / f'input_{idx}_graph.png'
            visualize_input_graph(model, inputs[idx], activations[idx], graph_path, top_k_edges=showTopEdges)
            print('Saved input graph to', graph_path)

    print('Saved visualizations to', out_dir)


if __name__ == '__main__':
    main()

