import numpy as np
from dropout import Dropout

class SimpleNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size=128, dropout_rate=None):
        """
        Initialize a SimpleNeuralNetwork.
        
        Args:
            input_size (int): Number of input neurons
            output_size (int): Number of output neurons
            hidden_size (int): Number of hidden neurons (default=128)
            dropout_rate (float): Dropout probability (default=None)
        """
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        if dropout_rate:
            self.dropout = Dropout(p=dropout_rate)

    def relu(self, x):
        """
        Computes the ReLU activation function.
        
        Args:
            x (numpy array): Input array
            
        Returns:
            numpy array: Output array
        """
        return np.maximum(0, x)
        
    def softmax(self, x):
        """
        Computes the softmax activation function.
        
        Args:
            x (numpy array): Input array
            
        Returns:
            numpy array: Output array
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)