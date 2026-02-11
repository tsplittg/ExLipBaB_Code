# We train a simple NN to learn the absolute value function.
# We try ReLU and GroupSort activation with Group size 2 (MaxMin) (Anil et al. 2019)
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from exlipbab.helper_classes.custom_activation_functions import GroupSort as GroupSortActivation
from exlipbab.exlipbab_main import exlipbab_main
from exlipbab.helper_classes.piece_wise_linear_function import PWL_Relu, GroupSort, PWL_Identity, LeakyReLu
from exlipbab.helper_classes.polyhedron import Polyhedron, FullPolyhedron
from scipy.io import savemat
import time
import copy
# set to a folder path to read from saved networks and skip training; training is performed if set to None
read_from_folder =  "../exlipbab_saved_networks/absolute_value"  

# set experiment parameters:
n_simul = 20
num_hidden_layers = 2
hidden_layer_size = 6
epochs = 2000
patience = 20
activations = [ReLU(), GroupSortActivation(hidden_layer_size//2)]

# set seeds for reproducibility
seed_list = np.random.SeedSequence(42).spawn(n_simul)


network_shape = [1] + [hidden_layer_size]*num_hidden_layers + [1]

class NeuralNet(nn.Module):
    """
    Genreric feedforward neural networkwork using the given shape and activation function.
    """
    def __init__(self, shape, activation):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(shape)-2):
            layers.append(Linear(shape[i], shape[i+1]))
            layers.append(activation)
        layers.append(Linear(shape[-2], shape[-1]))
        self.model = Sequential(*layers)
    def forward(self, x):
        return self.model(x).reshape(-1)
    def get_weights_and_biases(self):
        weights = []
        biases = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.detach().numpy())
                biases.append(layer.bias.detach().numpy())
        return weights, biases

def run_simulation(number_simul, activation, seed, network_shape, epochs, patience):
    """
    Run a single simulation, for a gicen random seed. Train a NN with the given activation function
    to learn the absolute value function on [-1, 1] with Gaussian noise.
    Returns the test RMSE and the trained network.
    """

    torch.manual_seed(seed.generate_state(1)[0])
    # initialize dataset
    X = np.linspace(-1, 1, 2000).reshape(-1, 1)
    y = np.abs(X) + np.random.default_rng(seed).normal(0, 0.1, X.shape)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed.generate_state(1)[0])
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=seed.generate_state(1)[0])
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype= torch.float32).reshape(-1)
    y_test_tensor = torch.tensor(y_test, dtype= torch.float32).reshape(-1)
    y_val_tensor = torch.tensor(y_val, dtype= torch.float32).reshape(-1)

    # initialize network, optimizer etc.
    net = NeuralNet(network_shape, activation)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    batch_size = 256
    loss_fn = nn.MSELoss()
    # create dataloaders:
    train_datset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_datset, batch_size=batch_size, shuffle=True)
    train_loss_hist = []
    val_loss_hist = []
    net.train()
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # training loop with early stopping
    for epoch in range(epochs):
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
    test_loss = torch.sqrt(loss_fn(net(X_test_tensor), y_test_tensor))
    return test_loss.item(), net

def evaluate_ExLipBab_on_trained_networks(wts, bs, activation):
    """
    Given the weights (wts), biases (bs) and activation function of a trained network, evaluate ExLipBab on the network
    """
    input_polytope = FullPolyhedron(1) # we compute the global Lipschitz constant
    #input_polytope = Polyhedron.from_intervals(np.array([(-1, 1)]))  # alternatively: we could compute the local Lipschitz constant on the input domain [-1, 1]


    # we construct a list representation of the network for ExLipBab with alternating wts/bs-tuples and activation functions
    # first, we check that the dimensionality of the input matches; for this, the dimension of each weight in layer l should be d_{l+1} x d_{l}
    dimensions_match = all(wts[i].shape[0] == len(bs[i]) for i in range(len(bs)))
    if not dimensions_match:
        raise ValueError(f"Dimensions of weights and biases do not match, got shapes {[ (wts[i].shape, bs[i].shape) for i in range(len(bs)) ]}.") 
    network = []
    for i in range(len(wts)):
        network.append((wts[i], bs[i]))
        if i < len(wts)-1:
            match activation.__class__.__name__:
                case "GroupSort":
                    network.append(GroupSort(wts[i].shape[0], group_size=wts[i].shape[0]//activation.num_groups))
                case "ReLU":
                    network.append(PWL_Relu(wts[i].shape[0]))
                case "LeakyReLU":
                    network.append(LeakyReLu(wts[i].shape[0], leak_factor=activation.negative_slope))
        else:
            network.append(PWL_Identity(wts[i].shape[0]))

    glb, gub = exlipbab_main(N = network, X=input_polytope, verbose = False)
    return gub


def run_experiments(activations, n_simul, seed_list, network_shape, epochs, patience):
    """
    Function to run all experiments for the different activation functions and seeds.
    Returns a dataframe with the results. Also saves the weights and biases of each trained network for ExLipBab evaluation.
    """

    activations = activations
    # we initialize an empty dataframe to store the results in
    results =  pd.DataFrame(columns=['Activation', 'Simulation', 'Test_RMSE', 'ExLipBaB', 'runtime_Exlipbab', 'Layerwise_Bound', 'runtime_layerwise']+ [f'Layer_{i+1}_Norm' for i in range(len(network_shape)-1)])
    for activation in activations:
        activation_name = activation.__class__.__name__
        if activation_name == "GroupSort":
            activation_name += f"_numGroups{activation.num_groups}"
        for sim in tqdm(range(n_simul)):
            simulation_results = {}
            seed = seed_list[sim]
            test_rmse, net = run_simulation(sim, activation, seed, network_shape, epochs, patience)
            simulation_results['Activation'] = activation_name
            simulation_results['Simulation'] = sim
            simulation_results['Test_RMSE'] = test_rmse
            print(f"Completed simulation {sim+1}/{n_simul} for activation {activation_name}")
            # we save the network weights and biases for ExLipbab computation
            weights, biases = net.get_weights_and_biases()
            np.save(f'../exlipbab_saved_networks/absolute_value/{network_shape}{activation_name}simul_{sim}_weights.npy', np.array(weights, dtype= object))
            np.save(f'../exlipbab_saved_networks/absolute_value/{network_shape}{activation_name}simul_{sim}_biases.npy', np.array(biases, dtype= object))
            # we also save the weights and biases in a matlab compatible format
            savemat(f'../exlipbab_saved_networks/absolute_value/{network_shape}{activation_name}simul_{sim}_weights.mat', {'W': weights})
            savemat(f'../exlipbab_saved_networks/absolute_value/{network_shape}{activation_name}simul_{sim}_biases.mat', {'b': biases})


            # now evaluate ExLipbab on the trained network
            start_time = time.time()
            exlipbab_bound = evaluate_ExLipBab_on_trained_networks(weights, biases, activation)
            end_time = time.time()
            simulation_results['runtime_Exlipbab'] = end_time - start_time
            simulation_results['ExLipBaB'] = exlipbab_bound

            # compute layerwise norms
            start_time = time.time()
            layerwise_norms = []
            layerwise_bound = 1
            for W in weights:
                norm = np.linalg.norm(W, ord=2)
                layerwise_norms.append(norm)
                layerwise_bound *= norm
            end_time = time.time()
            simulation_results['runtime_layerwise'] = end_time - start_time
            simulation_results['Layerwise_Bound'] = layerwise_bound
            for i, norm in  enumerate(layerwise_norms):
                simulation_results[f'Layer_{i+1}_Norm'] = norm
            results = pd.concat([results, pd.DataFrame([simulation_results])])


    return results.reset_index()

if read_from_folder is None:
    results_dataframe = run_experiments(activations, n_simul, seed_list, network_shape, epochs, patience)
    results_dataframe.to_csv(f'../exlipbab_saved_networks/absolute_value/absolute_value_experiments_results.csv')
else:
    results_dataframe = pd.read_csv(f'{read_from_folder}/absolute_value_experiments_results.csv', index_col=[0,1])

# we now read the mean and std of the results and append them to a summary file
summary_results = pd.DataFrame(columns=["Activation", "Test_RMSE_Mean",  "Test_RMSE_Std", "ExLipBaB_Mean", "ExLipBaB_Std",
                                        "Runtime_ExLipBab_mean", "Runtime_Exlipbab_std", "Layerwise_Bound_Mean", "Layerwise_Bound_Std",
                                        "Runtime_Layerwise_mean", "Runtime_Layerwise_std"])
summary_results.set_index("Activation")
for activation in activations:
    activation_name = activation.__class__.__name__
    if activation_name == "GroupSort":
        activation_name += f"_numGroups{activation.num_groups}"
    activation_subdf = results_dataframe[results_dataframe['Activation']== activation_name]
    test_rmse_mean = activation_subdf['Test_RMSE'].mean()
    test_rmse_std = activation_subdf['Test_RMSE'].std()
    exlipbab_mean = activation_subdf['ExLipBaB'].mean()
    exlipbab_std = activation_subdf['ExLipBaB'].std()
    exlipbab_runtime_mean = activation_subdf['runtime_Exlipbab'].mean()
    exlipbab_runtime_std = activation_subdf['runtime_Exlipbab'].std()
    layerwise_bound_mean = activation_subdf['Layerwise_Bound'].mean()
    layerwise_bound_std = activation_subdf['Layerwise_Bound'].std()
    layerwise_runtime_mean = activation_subdf['runtime_layerwise'].mean()
    layerwise_runtime_std = activation_subdf['runtime_layerwise'].std()
    
    summary_results = pd.concat([summary_results, pd.DataFrame([{"Activation": activation_name,
                                                                "Test_RMSE_Mean": test_rmse_mean,
                                                                "Test_RMSE_Std": test_rmse_std,
                                                                "ExLipBaB_Mean": exlipbab_mean,
                                                                "ExLipBaB_Std": exlipbab_std,
                                                                "Runtime_ExLipBab_mean": exlipbab_runtime_mean,
                                                                "Runtime_Exlipbab_std": exlipbab_runtime_std,
                                                                "Layerwise_Bound_Mean": layerwise_bound_mean,
                                                                "Layerwise_Bound_Std": layerwise_bound_std,
                                                                "Runtime_Layerwise_mean": layerwise_runtime_mean,
                                                                "Runtime_Layerwise_std": layerwise_runtime_std}])])
    
summary_results.to_csv(f'{read_from_folder if read_from_folder is not None else "../exlipbab_saved_networks/absolute_value"}/absolute_value_experiments_summary_results.csv')
    



