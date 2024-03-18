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
