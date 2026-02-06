# In this file, the saved networks are loaded for and the ExLipbab computation is run
import numpy as np
import pandas as pd
from exlipbab.exlipbab_main import exlipbab_main
from exlipbab.helper_classes.piece_wise_linear_function import PWL_Relu, PWL_Identity, Componentwise_Relu, GroupSort
from exlipbab.helper_classes.polyhedron import Polyhedron, FullPolyhedron
from exlipbab.helper_classes.sub_problem import SubProblem
from exlipbab.helper_classes.custom_activation_functions import GroupSort as GroupSortActivation
from tqdm import tqdm
import torch
import time

# remember to choose the right activation function and group size when constructing the network below
group_size = 2

#model_string = f'bike_net_(13x20x16x8x3)_GroupSort_numGroups{10}'
model_string = f'bike_net_(13x20x16x8x3)_ReLU'

# load the saved bike sharing network weights and biases
wts = np.load(f'../exlipbab_saved_networks/bike_sharing/{model_string}_weights.npy', allow_pickle=True)
bs = np.load(f'../exlipbab_saved_networks/bike_sharing/{model_string}_biases.npy', allow_pickle=True)

# we also load the bike dataset to define the input polyhedron
# we load the hour.csv dataset, downloaded from https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
bike_sharing_data = pd.read_csv("../bike+sharing+dataset/hour.csv")
X = bike_sharing_data.drop(columns=['instant', 'casual', 'registered', 'cnt'])

# we use three target variables: casual, registered, cnt
y = bike_sharing_data[['casual', 'registered', 'cnt']]

# we convert the date feature to numerical by formatting it as YYYYMMDD
X['dteday'] = pd.to_datetime(X['dteday']).dt.strftime('%Y%m%d').astype(int)

# we scale the data
X = (X - X.mean()) / X.std()

# we use as input polyhedron the box formed by the first and third quantile of each feature in the dataset
input_bounds = [(X.iloc[:,i].quantile(0.25), X.iloc[:,i].quantile(0.75)) for i in range(X.shape[1])]
#input_bounds = np.array([(0, 0.1)]*13)
#input_polyhedron = Polyhedron.from_intervals(input_bounds)
input_polyhedron = FullPolyhedron(13)



# we build a network list of the expected form:
network = []
for i in range(len(wts)):
    network.append((wts[i].T, bs[i]))
    if i < len(wts)-1:
        #CHOOSE CORRECT ACTIVATION FUNCTION
        #network.append(GroupSort(wts[i].shape[1], group_size=group_size))
        network.append(PWL_Relu(wts[i].shape[1]))
    else:
        network.append(PWL_Identity(wts[i].shape[1]))

input_intervals = np.array([(np.min(X.iloc[:,i]), np.max(X.iloc[:,i])) for i in range(X.shape[1])])
lower_bound_samples = 1000000
lower_bound_vertices = np.random.uniform(low=input_intervals[:,0], high=input_intervals[:,1], size=(lower_bound_samples, 13))
lower_bound_net = []
for i in range(len(wts)):
    lower_bound_net.append(torch.nn.Linear(wts[i].shape[1], wts[i].shape[0]))
    lower_bound_net[-1].weight = torch.nn.Parameter(torch.tensor(wts[i].T, dtype=torch.float32))
    lower_bound_net[-1].bias = torch.nn.Parameter(torch.tensor(bs[i], dtype=torch.float32))
    if i < len(wts)-1:
        if network[2*i+1].__class__ == PWL_Relu:
            lower_bound_net.append(torch.nn.ReLU())
        elif network[2*i+1].__class__ == GroupSort:
            raise NotImplementedError("GroupSort activation not implemented in lower bound computation network construction")
lower_bound_network = torch.nn.Sequential(*lower_bound_net)
# now compute gradients at each of the sampled vertices
lower_bound_tensor = torch.tensor(lower_bound_vertices, dtype=torch.float32, requires_grad=True)
# get the gradient matrices:
gradient_matrices = np.zeros((lower_bound_samples, lower_bound_tensor.shape[1], lower_bound_network(lower_bound_tensor).shape[1]))
for i in range(lower_bound_network(lower_bound_tensor).shape[1]):
    lower_bound_network.zero_grad()
    outputs = lower_bound_network(lower_bound_tensor)[:,i]
    outputs.backward(torch.ones_like(outputs))
    grads = lower_bound_tensor.grad.detach().numpy()
    gradient_matrices[:, :, i] = grads
    lower_bound_tensor.grad.zero_()


#grad_norms = np.linalg.norm(gradient_matrices, ord=2, axis=(1,2))
initial_lower_bound = 0 #np.max(grad_norms)



start_time = time.time()
glb, gub, bound_hist = exlipbab_main(N = network, X=input_polyhedron, lower_bound = 0., record_ram = True, solver = "glpk", verbose= True, record_symprop=True)
print(f"LipBaB computation took {time.time()-start_time} seconds")
print("final lower bound", glb)
print("final upper bound", gub)

# we also compute the layerwise bounds for comparison
layerwise_norms = []
for W in wts:
    layerwise_norms.append(np.linalg.norm(W, ord=2))



print("finished computation for model", model_string)
print("layerwise norms", layerwise_norms)
print("layerwise bound", np.prod(layerwise_norms))

# we save the bound history
np.save(f'../exlipbab_saved_networks/bike_sharing/{model_string}_exlipbab_bound_history_with_lower_bound.npy', np.array(bound_hist, dtype=object))
# to be safe, we save the lower and upper bounds separately as well
lower_bound_history, upper_bound_history = bound_hist
np.save(f'../exlipbab_saved_networks/bike_sharing/{model_string}_exlipbab_lower_bound_history_with_lower_bound.npy', np.array(lower_bound_history, dtype=object))
np.save(f'../exlipbab_saved_networks/bike_sharing/{model_string}_exlipbab_upper_bound_history_with_lower_bound.npy', np.array(upper_bound_history, dtype=object))