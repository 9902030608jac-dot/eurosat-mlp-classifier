import os
import numpy as np
from autograd import Tensor
from model import MLP
from optimizer import SGDOptimizer, LRScheduler, cross_entropy_loss
from dataloader import DataLoader
from evaluate import evaluate


def train(config, train_images, train_labels, val_images, val_labels,
          save_dir='checkpoints', verbose=True):
    os.makedirs(save_dir, exist_ok=True)

    model = MLP(
        input_dim=config.get('input_dim', 32 * 32 * 3),
        hidden_dim1=config.get('hidden_dim1', 256),
        hidden_dim2=config.get('hidden_dim2', 128),
        num_classes=config.get('num_classes', 10),
        activation=config.get('activation', 'relu')
    )

    optimizer = SGDOptimizer(
        model.parameters(),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 0.0),
        momentum=config.get('momentum', 0.9)
    )

    num_epochs = config.get('num_epochs', 50)
    lr_scheduler = LRScheduler(
        optimizer,
        mode=config.get('lr_decay_mode', 'step'),
        step_size=config.get('lr_step_size', 20),
        gamma=config.get('lr_gamma', 0.5),
        warmup_epochs=config.get('warmup_epochs', 0),
        total_epochs=num_epochs
    )

    train_loader = DataLoader(
        train_images, train_labels,
        batch_size=config.get('batch_size', 64),
        shuffle=True, seed=config.get('seed', 42)
    )

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    train_accs = []
    best_epoch = 0

    for epoch in range(num_epochs):
        lr_scheduler.update_optimizer_lr()
        epoch_loss = 0.0
        num_batches = 0

        for batch_images, batch_labels in train_loader:
            x = Tensor(batch_images)
            logits = model(x)
            loss = cross_entropy_loss(logits, batch_labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.data.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        val_acc, val_preds = evaluate(model, val_images, val_labels, batch_size=256)
        val_loss = _compute_val_loss(model, val_images, val_labels)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        train_acc, _ = evaluate(model, train_images, train_labels, batch_size=256)
        train_accs.append(train_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save_weights(os.path.join(save_dir, 'best_model.npz'))

        lr_scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == num_epochs - 1):
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | "
                  f"Val Acc: {val_acc*100:.2f}% | "
                  f"LR: {optimizer.lr:.6f} | "
                  f"Best Val Acc: {best_val_acc*100:.2f}%")

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'train_accs': train_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'config': config
    }
    return model, history


def _compute_val_loss(model, val_images, val_labels, batch_size=256):
    n = len(val_labels)
    total_loss = 0.0
    num_batches = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = Tensor(val_images[start:end])
        logits = model(x)
        loss = cross_entropy_loss(logits, val_labels[start:end])
        total_loss += loss.data.item()
        num_batches += 1
    return total_loss / num_batches
