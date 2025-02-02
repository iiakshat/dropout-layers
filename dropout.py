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
    
    def backward(self, d_out):
        """
        Backward pass of dropout.
        
        Args:
            d_out: Gradient of the output tensor
            
        Returns:
            Gradient of the input tensor
        """

        if not self.training:
            return d_out
            
        return d_out * self.mask
    
    def train(self):
        """
        Set the layer to training mode.
        """
        self.training = True
        
    def eval(self):
        """
        Set the layer to evaluation mode.
        """
        self.training = False

    @staticmethod
    def calc_prob(x, p, verbose=True):
        """
        Calculate the actual dropout rate.
        
        Args:
            x: Input tensor
            p: Dropout probability
            
        Returns:
            Actual dropout rate
        """
        dropout = Dropout(p=p)
        output = dropout.forward(x)
        actual_rate = 1 - (np.count_nonzero(output) / np.prod(x.shape))
        
        if verbose:
            print(f"Input values: {x} {x.shape}")
            print(f"Output values: {output} {output.shape}")
            print(f"\nDropout probability, p = {p:.1f}\nActual dropout rate = {actual_rate:.2f}")

        return actual_rate


if __name__ == "__main__":
    
    # Test dropout
    x = np.ones((1, 10))
    Dropout.calc_prob(x, 0.5)