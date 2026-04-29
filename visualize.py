import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import MLP
from dataloader import CLASS_NAMES


def plot_training_curves(history, save_path='results/training_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history['train_accs']], 'b--', label='Train Accuracy', linewidth=1.5, alpha=0.7)
    ax2.plot(epochs, [a * 100 for a in history['val_accs']], 'g-', label='Val Accuracy', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def visualize_first_layer_weights(model_path, img_size=32, save_path='results/first_layer_weights.png'):
    model = MLP.from_file(model_path)
    W = model.layer1.weight.data
    input_dim = W.shape[0]
    channels = 3
    pixel_count = input_dim // channels
    h = w = int(np.sqrt(pixel_count))
    assert h * w == pixel_count, f"Cannot reshape {pixel_count} pixels to square"

    n_neurons = W.shape[1]
    n_cols = min(16, n_neurons)
    n_rows = (n_neurons + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_neurons):
        row, col = divmod(idx, n_cols)
        weight_vec = W[:, idx]
        img = weight_vec.reshape(h, w, channels)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'N{idx}', fontsize=7)

    for idx in range(n_neurons, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    plt.suptitle('First Layer Weight Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"First layer weights saved to {save_path}")


def visualize_misclassified(misclassified, save_path='results/misclassified.png'):
    n = len(misclassified)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if n == 0:
        print("No misclassified samples to visualize.")
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, sample in enumerate(misclassified):
        axes[i].imshow(np.clip(sample['image'], 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"True: {sample['true_label']}\nPred: {sample['pred_label']}", fontsize=9)
    plt.suptitle('Misclassified Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Misclassified samples saved to {save_path}")


def plot_hyperparam_results(results, save_path='results/hyperparam_results.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    accs = [r['best_val_acc'] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(max(10, len(results) * 0.5), 5))
    x = range(len(results))
    bars = ax.bar(x, accs, color='steelblue', alpha=0.8)
    ax.set_xlabel('Run Index')
    ax.set_ylabel('Best Validation Accuracy (%)')
    ax.set_title('Hyperparameter Search Results')
    ax.set_xticks(x)
    labels = []
    for r in results:
        cfg = r['config']
        label_parts = []
        for k in ['lr', 'hidden_dim1', 'weight_decay', 'activation']:
            if k in cfg:
                label_parts.append(f"{k}={cfg[k]}")
        labels.append('\n'.join(label_parts))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hyperparameter results saved to {save_path}")
