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
from torchvision import datasets


from common_utils import get_model, get_device, get_norm_transform, default_checkpoint_path


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--quick', action='store_true', help='run quick mode (less data)')
    parser.add_argument('--input-graph', type=int, default=None, help='index of collected input to draw network graph for')
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint_path, help='path to save/load model checkpoint')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    transform = get_norm_transform()

    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    if args.quick:
        # smaller datasets for quick runs
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(2048)))
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(512)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = evaluate(model, device, test_loader)
        print(f'Epoch {epoch} accuracy: {acc * 100:.2f}%')

    # Save checkpoint
    ckpt = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'args': vars(args)}
    torch.save(ckpt, args.checkpoint)
    print('Saved checkpoint to', args.checkpoint)


if __name__ == '__main__':
    main()

