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
