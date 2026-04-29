import os
import itertools
import numpy as np
from train import train


def grid_search(param_grid, train_images, train_labels, val_images, val_labels,
                save_dir='checkpoints/hparam_search', verbose=True):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    best_acc = 0.0
    best_config = None
    best_history = None

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        config['input_dim'] = train_images.shape[1]
        config['num_classes'] = 10

        run_name = f"run_{i:03d}_" + "_".join(f"{k}={config[k]}" for k in keys)
        run_dir = os.path.join(save_dir, run_name)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Grid Search [{i+1}/{len(combinations)}]: {config}")
            print(f"{'='*60}")

        model, history = train(
            config, train_images, train_labels,
            val_images, val_labels,
            save_dir=run_dir, verbose=verbose
        )

        result = {
            'config': config,
            'best_val_acc': history['best_val_acc'],
            'best_epoch': history['best_epoch']
        }
        results.append(result)

        if history['best_val_acc'] > best_acc:
            best_acc = history['best_val_acc']
            best_config = config.copy()
            best_history = history

        if verbose:
            print(f"  -> Best Val Acc: {history['best_val_acc']*100:.2f}%")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Grid Search Complete!")
        print(f"Best Config: {best_config}")
        print(f"Best Val Acc: {best_acc*100:.2f}%")
        print(f"{'='*60}")

    return results, best_config, best_history


def random_search(param_distributions, n_trials, train_images, train_labels,
                  val_images, val_labels, save_dir='checkpoints/hparam_search',
                  seed=42, verbose=True):
    rng = np.random.RandomState(seed)
    results = []
    best_acc = 0.0
    best_config = None
    best_history = None

    for i in range(n_trials):
        config = {}
        for key, dist in param_distributions.items():
            if isinstance(dist, list):
                config[key] = dist[rng.randint(len(dist))]
            elif isinstance(dist, tuple) and len(dist) == 2:
                low, high = dist
                if isinstance(low, int) and isinstance(high, int):
                    config[key] = rng.randint(low, high + 1)
                else:
                    config[key] = rng.uniform(low, high)
            else:
                config[key] = dist

        config['input_dim'] = train_images.shape[1]
        config['num_classes'] = 10

        run_name = f"run_{i:03d}"
        run_dir = os.path.join(save_dir, run_name)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Random Search [{i+1}/{n_trials}]: {config}")
            print(f"{'='*60}")

        model, history = train(
            config, train_images, train_labels,
            val_images, val_labels,
            save_dir=run_dir, verbose=verbose
        )

        result = {
            'config': config,
            'best_val_acc': history['best_val_acc'],
            'best_epoch': history['best_epoch']
        }
        results.append(result)

        if history['best_val_acc'] > best_acc:
            best_acc = history['best_val_acc']
            best_config = config.copy()
            best_history = history

        if verbose:
            print(f"  -> Best Val Acc: {history['best_val_acc']*100:.2f}%")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Random Search Complete!")
        print(f"Best Config: {best_config}")
        print(f"Best Val Acc: {best_acc*100:.2f}%")
        print(f"{'='*60}")

    return results, best_config, best_history
