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
