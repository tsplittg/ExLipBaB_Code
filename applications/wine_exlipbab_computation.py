# In this file, the saved networks are loaded for and the ExLipbab computation is run
import numpy as np
import pandas as pd
from exlipbab.exlipbab_main import exlipbab_main
from exlipbab.helper_classes.piece_wise_linear_function import PWL_Relu, PWL_Identity, Componentwise_Relu, GroupSort
from exlipbab.helper_classes.polyhedron import Polyhedron, FullPolyhedron
from exlipbab.helper_classes.sub_problem import SubProblem
from exlipbab.helper_classes.custom_activation_functions import GroupSort as GroupSortActivation
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import time

# remember to choose the right activation function and group size when constructing the network below
group_size = 2

#model_string = f'wine_net_(11x24x24x1)_GroupSort_numGroups{12}'
model_string = f'wine_net_(11x24x24x1)_ReLU'

# load the saved bike sharing network weights and biases
wts = np.load(f'../exlipbab_saved_networks/wine/{model_string}_weights.npy', allow_pickle=True)

bs = np.load(f'../exlipbab_saved_networks/wine/{model_string}_biases.npy', allow_pickle=True)

# we also load the bike dataset to define the input polyhedron
# fetch dataset from UCI repository
wine = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets
# we only use quality as target
y = y[['quality']]

# we scale the data
X = (X - X.mean()) / X.std()

# we compute the global Lipschitz constant and use the intire R^{input_dim} as input polyhedron
input_polyhedron = FullPolyhedron(11)
#input_bounds = np.array([(-0.2, 0.2)]*11)
#input_polyhedron = Polyhedron.from_intervals(input_bounds)
# we do not use a lower bound for the ExLipbab computation
lower_bound = 0

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

start_time = time.time()
glb, gub, bound_hist = exlipbab_main(N = network, X=input_polyhedron, lower_bound = 0., record_ram = True, solver = "glpk", verbose= False, record_symprop=True, model_string=model_string, 
                                     timeout =3600*12)
print(f"LipBaB computation took {time.time()-start_time} seconds")
print("final lower bound", glb)
print("final upper bound", gub)

# we also compute the layerwise bounds for comparison
layerwise_time = time.time()
layerwise_norms = []
for W in wts:
    layerwise_norms.append(np.linalg.norm(W, ord=2))
print(f"Layerwise norm computation took {time.time()-layerwise_time} seconds")
print("layerwise norms", layerwise_norms)
print("layerwise bound", np.prod(layerwise_norms))

# we save the bound history
#np.save(f'../exlipbab_saved_networks/wine/{model_string}_exlipbab_bound_history.npy', np.array(bound_hist, dtype=object))
# to be safe, we save the lower and upper bounds separately as well
lower_bound_history, upper_bound_history = bound_hist
np.save(f'../exlipbab_saved_networks/wine/{model_string}_exlipbab_lower_bound_history.npy', np.array(lower_bound_history, dtype=object))
np.save(f'../exlipbab_saved_networks/wine/{model_string}_exlipbab_upper_bound_history.npy', np.array(upper_bound_history, dtype=object))
#plt.plot(lower_bound_history, label='Lower Bound')
#plt.show()