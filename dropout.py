import numpy as np

class Dropout:
    """
    Implementation of Dropout layer from scratch based on the paper:
    "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
    """
    def __init__(self, p=0.5, training=True):
        """
        Initialize the Dropout layer.
        
        Args:
            p (float): Probability of dropping a unit (setting it to 0)
            training (bool): Whether the layer is in training mode
        """

        # Check if p is between 0 and 1
        if p < 0 or p > 1:
            raise ValueError("Probability needs to be between 0 and 1")

        self.p = p
        self.training = training
        self.mask = None
        
    def forward(self, x):
        """
        Forward pass of dropout.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with dropped units
        """
        if not self.training:
            return x

        # Create mask
        self.mask = (np.random.rand(*x.shape) > self.p).astype(float)
        
        # Scale the mask during training
        self.mask /= (1 - self.p)
        
        # Apply mask
        return x * self.mask