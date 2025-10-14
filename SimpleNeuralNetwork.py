import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
import torch
from typing import List
import numpy as np

# Determine the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# DEVICE = get_device()
DEVICE = torch.device('cpu')
# print(f"Using device: {DEVICE}")

class SimpleNeuralNetwork(nn.Module):
    """Simple feedforward neural network using PyTorch"""

    def __init__(
        self, 
        input_size: int = 16, 
        hidden_layers: List[int] = [256], 
        output_size: int = 4, 
        empty: bool = False 
    ):
        super().__init__()
        
        if empty:
            return

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        # Build layers using PyTorch modules
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            # layers.append(nn.Tanh())
            layers.append(nn.Sigmoid())
            prev_size = hidden_size
            
        # Add output layer (no activation)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using He initialization
        self._initialize_weights()
        
        # Move to device
        self.to(DEVICE)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='tanh')
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Convert numpy array to tensor if needed and move to device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.to(DEVICE)
        
        return self.network(x)
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5):
        """Mutate the network's weights and biases"""
        with torch.no_grad():
            for param in self.parameters():
                if torch.rand(1).item() < mutation_rate:
                    mutation = torch.randn_like(param) * mutation_strength
                    param.add_(mutation)