import numpy as np
from autograd import Tensor
from model import MLP
from dataloader import DataLoader, CLASS_NAMES


def evaluate(model, images, labels, batch_size=256):
    n = len(labels)
    correct = 0
    all_preds = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = Tensor(images[start:end])
        logits = model(x_batch)
        preds = np.argmax(logits.data, axis=1)
        correct += np.sum(preds == labels[start:end])
        all_preds.extend(preds.tolist())
    accuracy = correct / n
    return accuracy, np.array(all_preds)


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    max_name_len = max(len(name) for name in class_names)
    header = ' ' * (max_name_len + 2) + ' '.join(f'{name:>{max_name_len}}' for name in class_names)
    print(header)
    print('-' * len(header))
    for i, name in enumerate(class_names):
        row = f'{name:>{max_name_len + 2}}' + ' '.join(f'{cm[i, j]:>{max_name_len}}' for j in range(len(class_names)))
        print(row)


def per_class_accuracy(cm):
    class_acc = cm.diagonal() / cm.sum(axis=1).astype(np.float64)
    return class_acc


def test_model(model_path, test_images, test_labels, batch_size=256):
    model = MLP.from_file(model_path)
    acc, preds = evaluate(model, test_images, test_labels, batch_size)
    cm = confusion_matrix(test_labels, preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print_confusion_matrix(cm, CLASS_NAMES)
    print("\nPer-class Accuracy:")
    class_acc = per_class_accuracy(cm)
    for name, ca in zip(CLASS_NAMES, class_acc):
        print(f"  {name:>25s}: {ca * 100:.2f}%")
    return acc, cm, preds


def find_misclassified(test_images, test_labels, preds, original_images, img_size=32, num_samples=5):
    mis_idx = np.where(preds != test_labels)[0]
    rng = np.random.RandomState(42)
    if len(mis_idx) > num_samples:
        selected = rng.choice(mis_idx, num_samples, replace=False)
    else:
        selected = mis_idx
    results = []
    for idx in selected:
        img = original_images[idx].reshape(img_size, img_size, 3)
        results.append({
            'index': int(idx),
            'true_label': CLASS_NAMES[test_labels[idx]],
            'pred_label': CLASS_NAMES[preds[idx]],
            'image': img
        })
    return results
