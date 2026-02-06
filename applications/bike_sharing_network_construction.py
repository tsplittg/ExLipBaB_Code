from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from exlipbab.helper_classes.custom_activation_functions import GroupSort
from scipy.io import savemat
import copy

# simulation parameters
network_shape = [13,16, 12, 4, 1]
activations = [ReLU(), ReLU(), ReLU()]
#activations = [GroupSort(10), GroupSort(8), GroupSort(4)]
#activations = [GroupSort(5), GroupSort(4), GroupSort(2)]
#activations = [LeakyReLU(), LeakyReLU(), LeakyReLU()]

# optimization parameters
lr = 0.001
batch_size = 256
patience = 5
epochs = 2000
weight_decay = 1e-3

# we load the hour.csv dataset, downloaded from https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
bike_sharing_data = pd.read_csv("../bike+sharing+dataset/hour.csv")
X = bike_sharing_data.drop(columns=['instant', 'casual', 'registered', 'cnt'])

# we use three target variables: casual, registered, cnt
y = bike_sharing_data[['cnt']]

# we convert the date feature to numerical by formatting it as YYYYMMDD
X['dteday'] = pd.to_datetime(X['dteday']).dt.strftime('%Y%m%d').astype(int)


# we scale the data
X = (X - X.mean(axis=0)) / X.std(axis=0)
 


# generic neural network class
class NeuralNet(nn.Module):
    def __init__(self, shape, activations):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(shape)-2):
            layers.append(Linear(shape[i], shape[i+1]))
            layers.append(activations[i])
        layers.append(Linear(shape[-2], shape[-1]))
        self.model = Sequential(*layers)
    def forward(self, x):
        return self.model(x)
    def get_weights_and_biases(self):
        weights = []
        biases = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.detach().numpy().T)
                biases.append(layer.bias.detach().numpy())
        return weights, biases
    



net = NeuralNet(network_shape, activations)
optimizer = torch.optim.Adam(net.model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.MSELoss()
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype= torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype= torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype= torch.float32)
# create dataloaders:
train_datset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_datset, batch_size=batch_size)

train_loss_hist = []
val_loss_hist = []
net.train()
best_val_loss = float('inf')
epochs_without_improvement = 0

# training loop with early stopping
for epoch in tqdm(range(epochs)):
    batch_loss_hist = []
    net.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = net(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_loss_hist.append(loss.item())
    train_loss_hist.append(np.mean(batch_loss_hist))
    net.eval()
    with torch.no_grad():
        val_outputs = net(X_val_tensor)
        val_loss = loss_fn(val_outputs, y_val_tensor).item()
        val_loss_hist.append(val_loss)
    
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_state = copy.deepcopy(net.state_dict())
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        net.load_state_dict(best_model_state)
        break

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(train_loss_hist, label='Train Loss')
ax[0].set_title('Training Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('MSE Loss')
ax[0].legend()
ax[1].plot(val_loss_hist, label='Validation Loss', color='orange')
ax[1].set_title('Validation Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('MSE Loss')
ax[1].legend()
plt.show()


test_loss = loss_fn(net.forward(X_test_tensor), y_test_tensor) ** 0.5
print(f"Test Loss of net with activation {activations}: {test_loss.item()}")

activation_name = activations[0].__class__.__name__
if activation_name == "GroupSort":
    activation_name += f"_numGroups{activations[0].num_groups}"

# save network weights and biases for ExLipbab computation
weights, biases = net.get_weights_and_biases()
np.save(f'../exlipbab_saved_networks/bike_sharing/bike_net_({"x".join([str(s) for s in network_shape])})_{activation_name}_weights.npy', np.array(weights, dtype= object))
np.save(f'../exlipbab_saved_networks/bike_sharing/bike_net_({"x".join([str(s) for s in network_shape])})_{activation_name}_biases.npy', np.array(biases, dtype= object))

#also transpose the weights and save them in matlab format; note: LipSDP expects the double format
savemat(f'../exlipbab_saved_networks/bike_sharing/bike_net_({"x".join([str(s) for s in network_shape])})_{activation_name}_weights.mat', 
        {'weights': [w.T.astype(np.float64) for w in weights]})
savemat(f'../exlipbab_saved_networks/bike_sharing/bike_net_({"x".join([str(s) for s in network_shape])})_{activation_name}_biases.mat', 
        {'biases': [b.astype(np.float64) for b in biases]})

layerwise_norms = []
for W in weights:
    layerwise_norms.append(np.linalg.norm(W, ord=2))

print("layerwise norms", layerwise_norms)
print("layerwise bound", np.prod(layerwise_norms))

