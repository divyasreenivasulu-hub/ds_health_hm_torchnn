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
