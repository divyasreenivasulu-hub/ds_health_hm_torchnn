# Homework Report

## Personal details
** Name:Divya Nellepalli

## 1 Make a dataset to create random samples from different functions (10)

### inserting the code

```python

import torch
from torch.utils.data import Dataset
import numpy as np

class SimpleFunctionsDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function
        self.x = np.random.uniform(0, 2 * np.pi, n_samples)
        self.epsilon = np.random.uniform(-1, 1, n_samples)
        
        if function == 'linear':
            self.y = 1.5 * self.x + 0.3 + self.epsilon
        elif function == 'quadratic':
            self.y = 2.0 * self.x**2 + 0.5 * self.x + 0.3 + self.epsilon
        elif function == 'harmonic':
            self.y = 0.5**2 + np.sin(self.x) + 3 * np.cos(3*self.x) + 2 + self.epsilon
        else:
            raise ValueError("Unsupported function type")

        # Normalize the output
        self.y = (self.y - np.mean(self.y)) / np.std(self.y)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample_x = self.x[idx]
        sample_y = self.y[idx]
        return sample_x, sample_y

# Example usage:
# dataset = SimpleFunctionsDataset(n_samples=1000, function='linear')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
```
## 2 Make a dynamic NN module (10)

## code for question 2

```python

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
```
## 3 Make a dynamic Training module (10)

### code for question 3

```python

import torch
import torch.nn as nn
from MyModels import DenseModel

def training(train_dataloader, validation_dataloader, optimizer, model=None, loss=nn.MSELoss(), epochs=500, batch_size=10):
    # If no model is provided, instantiate a default DenseModel with an assumed input_size.
    # Note: Default DenseModel() might need an input_size, which isn't provided here.
    # This example assumes DenseModel has been adjusted to have default arguments or input_size isn't required.
    if model is None:
        model = DenseModel(input_size=1)  # Assuming input_size=1 for simplicity, adjust according to your dataset.

    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = loss(outputs, targets)
            loss_value.backward()
            optimizer.step()
            total_train_loss += loss_value.item()

        avg_train_loss = total_train_loss / len(train_dataloader.dataset)
        train_loss_list.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_dataloader:
                outputs = model(inputs)
                loss_value = loss(outputs, targets)
                total_val_loss += loss_value.item()

        avg_val_loss = total_val_loss / len(validation_dataloader.dataset)
        val_loss_list.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    return train_loss_list, val_loss_list, model
```
## 4 Make your test bed (10)

### code for question 4

```python
# TestBed.py

import ipywidgets as widgets
import matplotlib.pyplot as plt
from MyDatasets import SimpleFunctionsDataset
from MyModels import DenseModel
from MyTraining import training
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

# Define the widgets
hidden_layers = widgets.Dropdown(options=list(range(1, 6)), description='Hidden Layers:')
neurons_per_layer = widgets.Dropdown(options=list(range(1, 101)), description='Neurons/Layer:')
epochs = widgets.Dropdown(options=list(range(100, 1001, 100)), description='Epochs:')
activation_hidden = widgets.Dropdown(options=['relu', 'sigmoid', 'tanh', 'linear'], description='Act. Hidden:', value='relu')
activation_output = widgets.Dropdown(options=['relu', 'sigmoid', 'tanh', 'linear'], description='Act. Output:', value='linear')
function_to_approximate = widgets.Dropdown(options=['linear', 'quadratic', 'harmonic'], description='Function:', value='linear')
start_training = widgets.Button(description='Start Training')

# Display the widgets
widgets_display = widgets.VBox([hidden_layers, neurons_per_layer, epochs, activation_hidden, activation_output, function_to_approximate, start_training])
display(widgets_display)

# Event handler for the training button
def on_start_training_clicked(b):
    # Disable the button to prevent multiple clicks during training
    start_training.disabled = True

    # Prepare the dataset
    dataset = SimpleFunctionsDataset(n_samples=1000, function=function_to_approximate.value)
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    validation_dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Prepare the model
    model = DenseModel(input_size=1,
                       hidden_layers=hidden_layers.value,
                       neurons_per_layer=neurons_per_layer.value,
                       activation_hidden=activation_hidden.value,
                       activation_output=activation_output.value)

    # Prepare the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train the model
    train_loss, val_loss, trained_model = training(train_dataloader,
                                                   validation_dataloader,
                                                   optimizer,
                                                   model,
                                                   epochs=epochs.value)


    # Plotting the training and validation loss
    plt.figure(figsize=(12, 5))

    # Figure 1: Function Approximation
    plt.subplot(1, 2, 1)
    x_values = torch.linspace(0, 1, steps=1000).unsqueeze(1)
    with torch.no_grad():
        y_pred = trained_model(x_values).numpy()
    plt.plot(x_values.numpy(), y_pred, label='Model Prediction', color='red')
    plt.scatter(dataset.x, dataset.y, label='True Data', color='blue', alpha=0.5)
    plt.title('Function Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Figure 2: Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss', color='red')
    plt.plot(val_loss, label='Validation Loss', color='green')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Enable the button after training is complete
    start_training.disabled = False

# Link the button click event to the function
start_training.on_click(on_start_training_clicked)
```
## 5 Neural network analysis (10)





## 6 Extra points batch normalization (10)

### code for question 6

```python
import ipywidgets as widgets
from IPython.display import display
from MyModels import DenseModelBN
from MyDatasets import SimpleFunctionsDataset
from MyTraining import training

# Define widgets
hidden_layers = widgets.Dropdown(options=list(range(1, 6)), description='Hidden Layers:')
neurons_per_layer = widgets.Dropdown(options=list(range(1, 101)), description='Neurons/Layer:')
epochs = widgets.Dropdown(options=list(range(100, 1001, 100)), description='Epochs:')
activation_hidden = widgets.Dropdown(options=['relu', 'sigmoid', 'tanh', 'linear'], description='Act. Hidden:', value='relu')
activation_output = widgets.Dropdown(options=['relu', 'sigmoid', 'tanh', 'linear'], description='Act. Output:', value='linear')
batch_norm = widgets.Checkbox(value=False, description='Batch Normalization')
start_training_bn = widgets.Button(description='Start Training with BN')

# Display widgets
widgets_display_bn = widgets.VBox([hidden_layers, neurons_per_layer, epochs, activation_hidden, activation_output, batch_norm, start_training_bn])
display(widgets_display_bn)

# Define event handler for training with batch normalization
def on_start_training_bn_clicked(b):
    # You would include the training code here, similar to the one we have discussed previously
    start_training_bn.disabled = True

    # Prepare the dataset
    dataset = SimpleFunctionsDataset(n_samples=1000, function=function_to_approximate.value)
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    validation_dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Prepare the model
    model = DenseModel(input_size=1,
                       hidden_layers=hidden_layers.value,
                       neurons_per_layer=neurons_per_layer.value,
                       activation_hidden=activation_hidden.value,
                       activation_output=activation_output.value)

    # Prepare the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train the model
    train_loss, val_loss, trained_model = training(train_dataloader,
                                                   validation_dataloader,
                                                   optimizer,
                                                   model,
                                                   epochs=epochs.value)
    pass

# Link the button click event to the handler function
start_training_bn.on_click(on_start_training_bn_clicked)
```



