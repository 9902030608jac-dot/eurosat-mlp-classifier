import numpy as np
from autograd import Tensor


ACTIVATIONS = {
    'relu': lambda t: t.relu(),
    'sigmoid': lambda t: t.sigmoid(),
    'tanh': lambda t: t.tanh(),
}


class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros((1, out_features)),
            requires_grad=True
        )

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class MLP:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes=10, activation='relu'):
        self.layer1 = Linear(input_dim, hidden_dim1)
        self.layer2 = Linear(hidden_dim1, hidden_dim2)
        self.layer3 = Linear(hidden_dim2, num_classes)
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(ACTIVATIONS.keys())}")
        self.activation = ACTIVATIONS[activation]
        self.activation_name = activation

    def __call__(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x

    def parameters(self):
        params = []
        params.extend(self.layer1.parameters())
        params.extend(self.layer2.parameters())
        params.extend(self.layer3.parameters())
        return params

    def save_weights(self, filepath):
        weights = {}
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            weights[f'layer{i}_weight'] = self._to_numpy(layer.weight.data)
            weights[f'layer{i}_bias'] = self._to_numpy(layer.bias.data)
        weights['activation'] = self.activation_name
        weights['input_dim'] = self.layer1.weight.data.shape[0]
        weights['hidden_dim1'] = self.layer1.weight.data.shape[1]
        weights['hidden_dim2'] = self.layer2.weight.data.shape[1]
        weights['num_classes'] = self.layer3.weight.data.shape[1]
        np.savez(filepath, **weights)

    def load_weights(self, filepath):
        weights = np.load(filepath, allow_pickle=True)
        self.layer1.weight.data = weights['layer0_weight']
        self.layer1.bias.data = weights['layer0_bias']
        self.layer2.weight.data = weights['layer1_weight']
        self.layer2.bias.data = weights['layer1_bias']
        self.layer3.weight.data = weights['layer2_weight']
        self.layer3.bias.data = weights['layer2_bias']

    @staticmethod
    def _to_numpy(data):
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)

    @classmethod
    def from_file(cls, filepath):
        weights = np.load(filepath, allow_pickle=True)
        model = cls(
            input_dim=int(weights['input_dim']),
            hidden_dim1=int(weights['hidden_dim1']),
            hidden_dim2=int(weights['hidden_dim2']),
            num_classes=int(weights['num_classes']),
            activation=str(weights['activation'])
        )
        model.load_weights(filepath)
        return model
