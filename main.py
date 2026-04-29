import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_dataset, split_dataset, compute_mean_std, normalize, augment_dataset, DataLoader, CLASS_NAMES
from train import train
from evaluate import test_model, find_misclassified
from hyperparam_search import grid_search, random_search
from visualize import (plot_training_curves, visualize_first_layer_weights,
                       visualize_misclassified, plot_hyperparam_results)


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EuroSAT_RGB')
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'hw1',
        'EuroSAT_RGB'
    )
if not os.path.exists(DEFAULT_DATA_DIR):
    DEFAULT_DATA_DIR = '/Users/jacicon/Downloads/hw1/EuroSAT_RGB'


def resolve_data_dir(args):
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"EuroSAT_RGB directory not found: {data_dir}. "
            "Pass --data-dir /path/to/EuroSAT_RGB."
        )
    return data_dir


def run_training(args):
    data_dir = resolve_data_dir(args)
    print("Loading dataset...")
    images, labels = load_dataset(data_dir, img_size=args.img_size)
    print(f"Dataset loaded: {images.shape[0]} images, {images.shape[1]} features, {len(CLASS_NAMES)} classes")

    print("Splitting dataset...")
    train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = split_dataset(
        images, labels,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=args.seed
    )
    print(f"Train: {len(train_lbl)}, Val: {len(val_lbl)}, Test: {len(test_lbl)}")

    if args.augment:
        print("Applying data augmentation (horizontal + vertical flip)...")
        train_img, train_lbl = augment_dataset(train_img, train_lbl, img_size=args.img_size)
        print(f"Augmented train set: {len(train_lbl)} samples")

    print("Computing normalization statistics...")
    mean, std = compute_mean_std(train_img)
    train_img = normalize(train_img, mean, std)
    val_img = normalize(val_img, mean, std)
    test_img = normalize(test_img, mean, std)

    os.makedirs(args.save_dir, exist_ok=True)
    norm_stats = {'mean': mean, 'std': std}
    np.savez(os.path.join(args.save_dir, 'norm_stats.npz'), **norm_stats)

    config = {
        'input_dim': train_img.shape[1],
        'hidden_dim1': args.hidden_dim1,
        'hidden_dim2': args.hidden_dim2,
        'num_classes': 10,
        'activation': args.activation,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'lr_decay_mode': args.lr_decay,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'seed': args.seed,
    }

    print(f"\nTraining config: {config}")
    model, history = train(config, train_img, train_lbl, val_img, val_lbl,
                           save_dir=args.save_dir, verbose=True)

    plot_training_curves(history, save_path=os.path.join(args.save_dir, 'training_curves.png'))

    print(f"\nBest validation accuracy: {history['best_val_acc']*100:.2f}% at epoch {history['best_epoch']+1}")

    print("\nEvaluating on test set...")
    best_model_path = os.path.join(args.save_dir, 'best_model.npz')
    acc, cm, preds = test_model(best_model_path, test_img, test_lbl)

    print("\nVisualizing first layer weights...")
    visualize_first_layer_weights(best_model_path, img_size=args.img_size,
                                  save_path=os.path.join(args.save_dir, 'first_layer_weights.png'))

    print("\nFinding misclassified samples...")
    test_img_raw, _ = load_dataset(data_dir, img_size=args.img_size)
    _, _, _, _, test_img_raw, test_lbl_raw = split_dataset(
        test_img_raw, labels,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )
    mis = find_misclassified(test_img, test_lbl, preds, test_img_raw, img_size=args.img_size, num_samples=8)
    for m in mis:
        print(f"  Index {m['index']}: True={m['true_label']}, Pred={m['pred_label']}")
    visualize_misclassified(mis, save_path=os.path.join(args.save_dir, 'misclassified.png'))

    return model, history


def run_hyperparam_search(args):
    data_dir = resolve_data_dir(args)
    print("Loading dataset...")
    images, labels = load_dataset(data_dir, img_size=args.img_size)
    print(f"Dataset loaded: {images.shape[0]} images")

    print("Splitting dataset...")
    train_img, train_lbl, val_img, val_lbl, test_img, test_lbl = split_dataset(
        images, labels,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=args.seed
    )

    if args.augment:
        print("Applying data augmentation...")
        train_img, train_lbl = augment_dataset(train_img, train_lbl, img_size=args.img_size)

    print("Computing normalization statistics...")
    mean, std = compute_mean_std(train_img)
    train_img = normalize(train_img, mean, std)
    val_img = normalize(val_img, mean, std)
    test_img = normalize(test_img, mean, std)

    if args.search_type == 'grid':
        param_grid = {
            'lr': [0.01, 0.005, 0.001],
            'hidden_dim1': [256, 512],
            'hidden_dim2': [128, 64],
            'activation': ['relu', 'tanh'],
            'weight_decay': [0.0, 1e-4],
            'momentum': [0.9],
            'batch_size': [64],
            'num_epochs': [args.epochs],
            'lr_decay_mode': ['step'],
            'lr_step_size': [20],
            'lr_gamma': [0.5],
        }
        results, best_config, best_history = grid_search(
            param_grid, train_img, train_lbl, val_img, val_lbl,
            save_dir=args.save_dir, verbose=True
        )
    else:
        param_distributions = {
            'lr': (0.0005, 0.02),
            'hidden_dim1': [128, 256, 512],
            'hidden_dim2': [64, 128, 256],
            'activation': ['relu', 'tanh', 'sigmoid'],
            'weight_decay': (0.0, 1e-3),
            'momentum': [0.9, 0.95],
            'batch_size': [32, 64, 128],
            'num_epochs': [args.epochs],
            'lr_decay_mode': ['step', 'cosine'],
            'lr_step_size': [10, 20, 30],
            'lr_gamma': (0.3, 0.7),
        }
        results, best_config, best_history = random_search(
            param_distributions, args.n_trials,
            train_img, train_lbl, val_img, val_lbl,
            save_dir=args.save_dir, seed=args.seed, verbose=True
        )

    plot_hyperparam_results(results, save_path=os.path.join(args.save_dir, 'hyperparam_results.png'))

    print(f"\nBest config: {best_config}")
    print(f"Best val accuracy: {best_history['best_val_acc']*100:.2f}%")

    print("\nEvaluating best model on test set...")
    best_run_dir = os.path.join(args.save_dir, 'best_final')
    config = best_config.copy()
    config['input_dim'] = train_img.shape[1]
    config['num_classes'] = 10
    model, history = train(config, train_img, train_lbl, val_img, val_lbl,
                           save_dir=best_run_dir, verbose=True)

    best_model_path = os.path.join(best_run_dir, 'best_model.npz')
    acc, cm, preds = test_model(best_model_path, test_img, test_lbl)

    plot_training_curves(history, save_path=os.path.join(args.save_dir, 'best_training_curves.png'))
    visualize_first_layer_weights(best_model_path, img_size=args.img_size,
                                  save_path=os.path.join(args.save_dir, 'best_first_layer_weights.png'))

    return results, best_config


def run_test(args):
    data_dir = resolve_data_dir(args)
    print("Loading dataset...")
    images, labels = load_dataset(data_dir, img_size=args.img_size)

    print("Splitting dataset...")
    _, _, _, _, test_img, test_lbl = split_dataset(
        images, labels,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=args.seed
    )

    print("Normalizing...")
    stats = np.load(os.path.join(os.path.dirname(args.model_path), 'norm_stats.npz'))
    test_img = normalize(test_img, stats['mean'], stats['std'])

    print("\nTesting...")
    acc, cm, preds = test_model(args.model_path, test_img, test_lbl)

    test_img_raw, _ = load_dataset(data_dir, img_size=args.img_size)
    _, _, _, _, test_img_raw, test_lbl_raw = split_dataset(
        test_img_raw, labels,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )
    mis = find_misclassified(test_img, test_lbl, preds, test_img_raw, img_size=args.img_size)
    for m in mis:
        print(f"  Index {m['index']}: True={m['true_label']}, Pred={m['pred_label']}")
    visualize_misclassified(mis, save_path='results/test_misclassified.png')


def main():
    parser = argparse.ArgumentParser(description='Three-Layer MLP for EuroSAT Classification')
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    train_parser.add_argument('--hidden-dim1', type=int, default=1024, help='First hidden layer size')
    train_parser.add_argument('--hidden-dim2', type=int, default=256, help='Second hidden layer size')
    train_parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'])
    train_parser.add_argument('--weight-decay', type=float, default=0.001, help='L2 regularization')
    train_parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    train_parser.add_argument('--batch-size', type=int, default=128)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--lr-decay', type=str, default='cosine', choices=['step', 'cosine', 'exponential'])
    train_parser.add_argument('--lr-step-size', type=int, default=30)
    train_parser.add_argument('--lr-gamma', type=float, default=0.5)
    train_parser.add_argument('--img-size', type=int, default=32, help='Resize images to this size')
    train_parser.add_argument('--data-dir', type=str, default=None, help='Path to EuroSAT_RGB directory')
    train_parser.add_argument('--augment', action='store_true', default=True, help='Enable data augmentation (flip)')
    train_parser.add_argument('--no-augment', dest='augment', action='store_false', help='Disable data augmentation')
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--save-dir', type=str, default='checkpoints')

    search_parser = subparsers.add_parser('search', help='Hyperparameter search')
    search_parser.add_argument('--search-type', type=str, default='grid', choices=['grid', 'random'])
    search_parser.add_argument('--n-trials', type=int, default=10, help='Number of random search trials')
    search_parser.add_argument('--epochs', type=int, default=30, help='Epochs per trial')
    search_parser.add_argument('--img-size', type=int, default=32)
    search_parser.add_argument('--data-dir', type=str, default=None, help='Path to EuroSAT_RGB directory')
    search_parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    search_parser.add_argument('--seed', type=int, default=42)
    search_parser.add_argument('--save-dir', type=str, default='checkpoints/hparam_search')

    test_parser = subparsers.add_parser('test', help='Test a trained model')
    test_parser.add_argument('--model-path', type=str, required=True, help='Path to saved model weights')
    test_parser.add_argument('--img-size', type=int, default=32)
    test_parser.add_argument('--data-dir', type=str, default=None, help='Path to EuroSAT_RGB directory')
    test_parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'search':
        run_hyperparam_search(args)
    elif args.mode == 'test':
        run_test(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
