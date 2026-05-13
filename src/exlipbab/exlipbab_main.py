import heapq
import numpy as np
from exlipbab.helper_classes.sub_problem import SubProblem
from exlipbab.helper_classes.polyhedron import Polyhedron
from exlipbab.helper_classes.maxheap import MaxHeap
from exlipbab.algorithms.minor_algorithms import lipschitz_bounds, branch, lipschitz_bounds, symprop
from exlipbab.algorithms.helper_functions import linprop
from tqdm import tqdm
import time
#from line_profiler_pycharm import profile


def exlipbab_main(N: list, X: Polyhedron, k: float =1., pnorm = 2, verbose = True, solver = "glpk", **kwargs) -> tuple:
    """
    Main function of the ExLipBaB algorithm in a symbolic (hopefully faster) version. Contains algorithm 4 from the paper.

    Parameters
    ----------
    N : list
        List containing the layers of the NN for which the Lipschiutz norm should be computed.
        The list entries are alternating between either tuples (W,b) of weight matrices and bias vectors or strings
        indicating piecewise linear functions which are instances of the PiecewiseLinearFunction class.
    X : polyhedron
        Input domain as a polyhedron.
    k : float, optional
        Approximation factor $\geq 1$ such that the algorithm runs until the greatest upper and lower bounds fulfill:
        $gub \leq k \cdot glb$, default is 1 which corresponds to an exact computation.
    solver : str, optional
        Solver to be used in the linear programming steps, default is "glpk". If "mosek" is installed and licensed, 
        it can also be used.

    Returns
    -------
    tuple
        Tuple containing the lower and upper bounds of the Lipschitz constant.
    """
    intial_polyhedron = X

    timeout = kwargs.get('timeout', np.inf)
    start_time = time.time()

    # we record the histories of bounds
    lower_bound_history = []
    upper_bound_history = []

    # for now: even indices are (W,b), odd indices are activation functions, later check that the types are actually alternating
    weights_list = [N[i] for i in range(0, len(N), 2)]
    activations_list = [N[i] for i in range(1, len(N), 2)]
    if "record_symprop" in kwargs and kwargs['record_symprop']:
        symprop_start_time = time.time()

    initial_activation_Lambda, initial_activation_lambda, initial_tilde_l, star_neurons = symprop(intial_polyhedron, weights_list, activations_list, verbose = verbose)

    #initialize first subproblem:
    initial_sub_problem = SubProblem(intial_polyhedron, initial_activation_Lambda, initial_activation_lambda)
    initial_sub_problem.tilde_l = initial_tilde_l
    initial_sub_problem.star_neurons = star_neurons
    if verbose:
        print(f"Initial star neurons: {star_neurons}")
    # initialize the propagated weights as the weights of the first layer:
    initial_sub_problem.propagated_weights = (weights_list[0][0], weights_list[0][1])

    initial_sub_problem.propagated_weights = linprop(old_tilde_W=initial_sub_problem.propagated_weights[0], old_tilde_b=initial_sub_problem.propagated_weights[1],
                                                     network_weigths=weights_list, network_Lambda=initial_activation_Lambda,
                                                     network_lambda=initial_activation_lambda, start_layer = 0 , end_layer = min(initial_tilde_l, len(weights_list)-1))
    gub = lipschitz_bounds(sub_problem=initial_sub_problem, weights=weights_list, pnorm = pnorm)
    if "record_symprop" in kwargs and kwargs['record_symprop']:
        symprop_time = time.time() - symprop_start_time
        symprop_approx = gub

    if verbose:
        print(f"initial gub: {gub}")
    if initial_tilde_l == len(weights_list):
        # all neurons fixed linear, we are done and can skip the main loop
        return gub, gub
    
    initial_sub_problem.upper_bound = gub
    # initialize max-heap of subproblems with the initial subproblem
    subproblem_heap = [initial_sub_problem]
    subproblem_heap = MaxHeap(subproblem_heap)
    #initialize global lower bound to zero and start core loop

    # if an initial lower bound is provided, use it
    if 'lower_bound' in kwargs:
        glb = kwargs['lower_bound']
    else:
        glb = 0.

    if 'record_ram' in kwargs and kwargs['record_ram']:
        # we simply record the size of the heap at each iteration
        ram_list = [1]

    
    iterations_counter = 0 
    with tqdm(total = gub, initial=gub, desc="ExLipBaB Progress") as pbar:
        while bool(gub > k * glb) and time.time() - start_time < timeout:
            # get subproblem with largest upper bound
            current_sub_problem = subproblem_heap.pop() 
            # branch subproblem into two new subproblems
            new_subproblems, glb = branch(sub_problem=current_sub_problem,
                                        old_glb=glb, weights=weights_list, alpha=activations_list, pnorm = pnorm, solver = solver)
            # add new subproblems to heap
            for sp in new_subproblems:
                if sp.upper_bound >= glb:
                    # only add subproblems that can improve the global upper bound
                    subproblem_heap.push(sp)
            # update global upper bound
            old_gub = gub
            gub= subproblem_heap[0].upper_bound
            pbar.update(gub - old_gub)
            star_neurons = current_sub_problem.star_neurons
            if verbose:
                print(f"Current star neurons: {star_neurons}")
            iterations_counter += 1
            if 'record_ram' in kwargs and kwargs['record_ram']:
                ram_list.append(subproblem_heap.len())
            lower_bound_history.append(glb)
            upper_bound_history.append(gub)
    if verbose:
        print(f"number iterations {iterations_counter}")
    if 'record_ram' in kwargs and kwargs['record_ram']:
        print("Max heap size:", np.max(ram_list))
        print("Average max heap size:", np.mean(ram_list))
        model_string = kwargs.get('model_string', 'exlipbab_model')
        np.save(f"{model_string}_ram_usage_exlipbab.npy", np.array(ram_list))
    if "record_symprop" in kwargs and kwargs['record_symprop']:
        print(f"Symbolic interval propagation time: {symprop_time} seconds, with initial upper bound approximation {symprop_approx}")
    return glb, gub, (lower_bound_history, upper_bound_history)

