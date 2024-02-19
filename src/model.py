import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    Neural Network model with configurable input, hidden, and output layers.
    
    Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of units in the hidden layer.
        num_classes (int): The number of classes in the output layer.
        
    Attributes:
        l1 (torch.nn.Linear): First linear transformation layer.
        l2 (torch.nn.Linear): Second linear transformation layer.
        l3 (torch.nn.Linear): Third linear transformation layer.
        relu (torch.nn.ReLU): ReLU activation function.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the NeuralNetwork model.
        
        Parameters:
            input_size (int): The number of input features.
            hidden_size (int): The number of units in the hidden layer.
            num_classes (int): The number of classes in the output layer.
        """
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Parameters:
            x (torch.Tensor): Input data tensor.
            
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
