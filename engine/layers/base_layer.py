# layers/base_layer.py

class Layer:

    def forward(self, X):
        raise NotImplementedError

    def backward(self,grag):
        raise NotImplementedError