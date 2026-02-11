# In this file, we test that the Lipschitz constants of all networks tested in Bhowmick et al. (2021) are correctly computed.
# NOTE:  In order to run this file, the network weights and biases need to be downloaded from the repository of Bhowmick et al. (2021):
# https://github.com/pyrobits/LipBaB and saved in the corrresponding folder "../lipbab_saved_networks"

import numpy as np
from exlipbab.exlipbab_main import exlipbab_main, exlipbab_main_symbolic
from exlipbab.helper_classes.piece_wise_linear_function import PWL_Relu, PWL_Identity
from exlipbab.helper_classes.polyhedron import Polyhedron, FullPolyhedron
from exlipbab.helper_classes.sub_problem import SubProblem
from tqdm import tqdm
import time

# The original LipBaB paper reports the Lipschitz constant for 4 networks, whose weights and biases are loaded in the commented code below.
# Uncommend to test ExLipbab on these networks.
# Also note the input intervals used for the computation. Bhowmick et al. (2021) used [0, 0.1]^10 for the SD-networks and [0,1]^4 for the iris network.
input_bounds = (0, 0.1)
norm_to_use = 2 # In LipBaB paper reported: (1,2,np.inf)


wts=np.load('../lipbab_saved_networks/SDnet_(10,20,15,10,3)_weights.npy',allow_pickle=True)
bs=np.load('../lipbab_saved_networks/SDnet_(10,20,15,10,3)_biases.npy',allow_pickle=True)
#comparative values from original LipBab algorithm:
#Final Lipschitz estimation: 40.05651016370293 (2-norm)
# Final Lipschitz estimation: 48.04897158493324 (1-norm)
# Final Lipschitz estimation: 72.28569774866806(inf-norm)

#wts=np.load('../lipbab_saved_networks/SDnet_(10,30,30,30,3)_weights.npy',allow_pickle=True)
#bs=np.load('../lipbab_saved_networks/SDnet_(10,30,30,30,3)_biases.npy',allow_pickle=True)
#comparative values from original LipBab algorithm:
#Final Lipschitz estimation: 19.462799793264388 (2-norm)
# Final Lipschitz estimation: 19.369311332018757(1-norm)
# Final Lipschitz estimation: 39.11036425646691 (inf-norm)


#wts=np.load('../lipbab_saved_networks/SDnet_(10,15,10,3)_weights.npy',allow_pickle=True)
#bs=np.load('../lipbab_saved_networks/SDnet_(10,15,10,3)_biases.npy',allow_pickle=True)
#comparative values from original LipBab algorithm:
#Final Lipschitz estimation: 9.530716922306004 (2-norm)
#Final Lipschitz estimation: 10.41286850374477 (1-norm)
#Final Lipschitz estimation: 16.274810805318676 (inf-norm)

#wts=np.load('../lipbab_saved_networks/iris_net_(4,5,5,3)_weights.npy',allow_pickle=True)
#bs=np.load('../lipbab_saved_networks/iris_net_(4,5,5,3)_biases.npy',allow_pickle=True)
#comparative values from original LipBab algorithm:
#Final Lipschitz estimation: 6.771454513402937 (2-norm)
#Final Lipschitz estimation: 5.958048912394726 (1-norm) 
#Final Lipschitz estimation: 12.605085581416922 (inf-norm)

input_dim = wts[0].shape[0]
print([weights.shape for weights in wts])
input_intervals = np.array([input_bounds]*input_dim) 
input_polytope = Polyhedron.from_intervals(input_intervals)

network = []
for i in range(len(wts)):
    network.append((wts[i].T, bs[i]))
    if i < len(wts)-1:
        network.append(PWL_Relu(wts[i].shape[1]))
    else:
        network.append(PWL_Identity(wts[i].shape[1]))

start_time = time.time()
initial_lower_bound = 0
glb, gub = exlipbab_main_symbolic(N = network, X=input_polytope, lower_bound = 0., record_ram = True, solver = "glpk", pnorm = norm_to_use)
print(f"LipBaB computation took {time.time()-start_time} seconds")
print("final lower bound", glb)
print("final upper bound", gub)