import numpy as np
from autograd import Tensor


def cross_entropy_loss(logits, labels, weight_decay=0.0, model_params=None):
    shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = exp_shifted.sum(axis=-1, keepdims=True)
    log_sum_exp = np.log(sum_exp)
    log_probs = shifted - log_sum_exp
    batch_size = logits.data.shape[0]
    loss_val = -log_probs[np.arange(batch_size), labels].mean()

    out = Tensor(np.array(loss_val), requires_grad=logits.requires_grad, _children=(logits,), _op='cross_entropy')

    def _backward():
        if logits.requires_grad:
            softmax = exp_shifted / sum_exp
            grad = softmax.copy()
            grad[np.arange(batch_size), labels] -= 1.0
            grad /= batch_size
            if out.grad.ndim == 0:
                logits._accumulate_grad(grad * out.grad.item())
            else:
                logits._accumulate_grad(grad * out.grad)
    out._backward = _backward

    return out


class SGDOptimizer:
    def __init__(self, params, lr=0.01, weight_decay=0.0, momentum=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.copy()
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                p.data = p.data - self.lr * self.velocities[i]
            else:
                p.data = p.data - self.lr * grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class LRScheduler:
    def __init__(self, optimizer, mode='step', step_size=10, gamma=0.5,
                 warmup_epochs=0, warmup_lr=1e-5, total_epochs=100):
        self.optimizer = optimizer
        self.mode = mode
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = optimizer.lr
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def step(self):
        self.current_epoch += 1

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            return self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)

        if self.mode == 'step':
            decay_epochs = self.current_epoch - self.warmup_epochs
            return self.base_lr * (self.gamma ** (decay_epochs // self.step_size))
        elif self.mode == 'cosine':
            total_decay_epochs = max(self.total_epochs - self.warmup_epochs, 1)
            progress = min((self.current_epoch - self.warmup_epochs) / total_decay_epochs, 1.0)
            return self.warmup_lr + 0.5 * (self.base_lr - self.warmup_lr) * (1 + np.cos(np.pi * progress))
        elif self.mode == 'exponential':
            return self.base_lr * (self.gamma ** self.current_epoch)
        else:
            return self.base_lr

    def update_optimizer_lr(self):
        self.optimizer.lr = self.get_lr()
