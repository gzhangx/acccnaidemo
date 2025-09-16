# MNIST neuron activation visualizer

This project trains a small MLP on MNIST and visualizes how each neuron in the hidden layer is activated for a batch of inputs.

Files:
- `train_and_visualize.py` - main script. Trains a model and saves activation visualizations.
- `requirements.txt` - Python dependencies.

Quick start:

1. Create a virtualenv and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the script (it will download MNIST automatically):

```bash
python train_and_visualize.py --epochs 3 --batch-size 128 --hidden-size 128
```

Outputs:
- `activations.png` - heatmap of neuron activations for sample inputs.
- `activation_examples/` - per-neuron image grids showing top activating inputs.

Notes:
- The script is designed to be simple and self-contained for experimentation.
- For large hidden sizes or more epochs, training will take longer and may require a GPU.
