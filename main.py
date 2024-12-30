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
    
    def forward(self, X, training=True):
        """
        Computes the forward pass of the network.
        
        Args:
            X (numpy array): Input array
            training (bool): Whether the network is in training mode (default=True)
            
        Returns:
            numpy array: Output array
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        if training:
            self.dropout.train()
        else:
            self.dropout.eval()

        self.a1 = self.dropout.forward(self.a1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2
    
    def predict(self, X):
        """
        Predict the output class for the given input data.

        Args:
            X (numpy array): Input data array.

        Returns:
            numpy array: Indices of the predicted output classes.
        """
        return np.argmax(self.forward(X, training=False), axis=1)
