import torch.nn as nn

class DenseModel(nn.Module):
    def __init__(self, input_size, hidden_layers=1, neurons_per_layer=1, activation_hidden='relu', activation_output='linear'):
        super(DenseModel, self).__init__()

        # Define activation functions dictionary
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity()
        }

        # Initialize a list to hold the layers
        layers = [nn.Linear(input_size, neurons_per_layer), activations[activation_hidden]]

        # Add hidden layers with the specified activation function
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activations[activation_hidden])

        # Output layer with the specified activation function
        layers.append(nn.Linear(neurons_per_layer, 1))
        layers.append(activations[activation_output])

        # Use nn.Sequential to create the module list
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class DenseModelBN(nn.Module):
    def __init__(self, input_size, hidden_layers=1, neurons_per_layer=10, activation_hidden='relu', activation_output='linear', batch_norm=False):
        super(DenseModelBN, self).__init__()

        # Activation functions dictionary
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'linear': nn.Identity()
        }

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, neurons_per_layer))
        if batch_norm:
            layers.append(nn.BatchNorm1d(neurons_per_layer))
        layers.append(activations[activation_hidden])

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            if batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(activations[activation_hidden])

        # Output layer
        layers.append(nn.Linear(neurons_per_layer, 1))
        layers.append(activations[activation_output])

        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
