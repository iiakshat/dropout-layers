import numpy as np
from dropout import Dropout

class SimpleNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size=128, dropout_rate=None):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        if dropout_rate:
            self.dropout1 = Dropout(p=dropout_rate)

    def relu(self, x):
        return np.maximum(0, x)
        
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)