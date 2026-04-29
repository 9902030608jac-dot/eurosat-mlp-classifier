import os
import numpy as np
from PIL import Image


CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]


def load_dataset(data_dir, img_size=32):
    images = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        filenames = sorted(os.listdir(class_dir))
        for fname in filenames:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(class_dir, fname)
            img = Image.open(fpath).convert('RGB')
            img = img.resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float64) / 255.0
            images.append(arr.flatten())
            labels.append(class_idx)
    images = np.array(images, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)
    return images, labels


def augment_dataset(images, labels, img_size=32):
    augmented_images = [images]
    augmented_labels = [labels]
    channels = 3
    h = w = img_size

    hflip = images.reshape(-1, h, w, channels)[:, ::-1, :, :].reshape(-1, h * w * channels)
    augmented_images.append(hflip)
    augmented_labels.append(labels.copy())

    vflip = images.reshape(-1, h, w, channels)[:, :, ::-1, :].reshape(-1, h * w * channels)
    augmented_images.append(vflip)
    augmented_labels.append(labels.copy())

    return np.concatenate(augmented_images, axis=0), np.concatenate(augmented_labels, axis=0)


def compute_mean_std(images):
    mean = images.mean(axis=0)
    std = images.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def normalize(images, mean, std):
    return (images - mean) / std


def split_dataset(images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n = len(labels)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx],
            images[test_idx], labels[test_idx])


class DataLoader:
    def __init__(self, images, labels, batch_size=64, shuffle=True, seed=42):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.n = len(labels)
        self.indices = np.arange(self.n)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self._pos = 0
        return self

    def __next__(self):
        if self._pos >= self.n:
            raise StopIteration
        batch_idx = self.indices[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return self.images[batch_idx], self.labels[batch_idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size
