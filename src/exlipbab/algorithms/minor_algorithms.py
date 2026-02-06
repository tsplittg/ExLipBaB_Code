import numpy as np
import copy
from tqdm import tqdm
from exlipbab.helper_classes.sub_problem import SubProblem
from exlipbab.helper_classes.piece_wise_linear_function import PiecewiseLinearFunction
from exlipbab.helper_classes.polyhedron import Polyhedron
from exlipbab.algorithms.helper_functions import linprop
from exlipbab.algorithms.interval_computations import interval_matrix_multiplication


def symprop(domain: Polyhedron, weights_list: list, activations_list: list[PiecewiseLinearFunction], verbose = False):
    """
    Implementation of the Symprop algorithm which uses symbolic propagation adding new variables if needed.
    These new variables are given intervals of possible values.
    """

    # initialize W, b for symbolic propagation
    tilde_W = weights_list[0][0]
    tilde_b = weights_list[0][1]
    dim_zero = domain.dimension
    new_variables_bounds = np.array([]).reshape(0,2)
    # compute the vertices of the input polyhedron which will be used in connection with the interval bounds
    input_vertices = domain.compute_vertices()
    star_neurons = []
    
    l_tilde = len(weights_list)  # initialize to max value
    activation_pattern_Lambda = []
    activation_pattern_lambda = []
    for ell in tqdm(range(len(weights_list)), desc = "SymProp"):
        new_variables = []
        polyhedron_list = activations_list[ell].list_of_polyhedra
        star_neurons_layer = []
        # for each neuron, check the polyhedra which decide the activation state of that neuron
        activation_states_for_W_tilde = np.zeros((weights_list[ell][0].shape[0], weights_list[ell][0].shape[0])) 
        biases_for_b_tilde = np.zeros(weights_list[ell][0].shape[0])
        Lambda_layer = np.stack((np.inf *np.ones((weights_list[ell][0].shape[0], weights_list[ell][0].shape[0])), 
                                 -np.inf *np.ones((weights_list[ell][0].shape[0], weights_list[ell][0].shape[0]))), 2)
        lambda_layer = np.stack((np.inf *np.ones(weights_list[ell][0].shape[0]), -np.inf *np.ones(weights_list[ell][0].shape[0])), 1)

        
        for neuron_index in range(weights_list[ell][0].shape[0]):
            # get all possible activation states for that neuron as well as a list of polyhedra for each activation state
            # that contain all points in pre-activation space that lead to that activation state
            all_possible_activation_states, polyhedra_list = activations_list[ell].get_all_possible_activation_states(neuron_index)
            # for each activation state, check whether the propagated polyhedron intersects with any of the polyhedra
            # if it does, we also compute the minimum and maximum value of the activated neuron over the propagated polyhedron (approximatively for now)
            is_star = False
            min_act_value = np.inf
            max_act_value = -np.inf
            number_of_different_states = 0
            for state_index in range(len(all_possible_activation_states)):
                # check intersection with all polyhedra for that activation state
                
                activation_state_weight, activation_state_bias = all_possible_activation_states[state_index]
                minimum_activation_over_input = None
                maximum_activation_over_input = None
                for poly_index in polyhedra_list[state_index]:
                    joint_constraints_A = np.vstack((polyhedron_list[poly_index].A @ tilde_W, domain.A))
                    joint_constraints_b = np.hstack((polyhedron_list[poly_index].b - polyhedron_list[poly_index].A @ tilde_b, domain.b))
                    
                    if len(joint_constraints_A) == 0 or not Polyhedron(joint_constraints_A, joint_constraints_b).is_empty():
                        if number_of_different_states >= 1:
                            # we have found more than one activation state for this neuron over the propagated polyhedron
                            is_star = True
                            activation_states_for_W_tilde[neuron_index,:] = 0
                            biases_for_b_tilde[neuron_index] = 0
                        else:
                            activation_states_for_W_tilde[neuron_index,:] = activation_state_weight.reshape(-1)
                            biases_for_b_tilde[neuron_index] = activation_state_bias.item()

                        number_of_different_states += 1

                        # update Lambda_layer and lambda_layer
                        Lambda_layer[neuron_index, :, 0] = np.minimum(Lambda_layer[neuron_index, :, 0], activation_state_weight)
                        Lambda_layer[neuron_index, :, 1] = np.maximum(Lambda_layer[neuron_index, :, 1], activation_state_weight)
                        lambda_layer[neuron_index, 0] = np.minimum(lambda_layer[neuron_index, 0], activation_state_bias.item())
                        lambda_layer[neuron_index, 1] = np.maximum(lambda_layer[neuron_index, 1], activation_state_bias.item())

                        # compute minimum and maximum value of the neuron over the propagated polyhedron
                        # for now, we simply compute the minimum and maximum activation over the vertices of the input polyhedron
                        # and intersect with the minimumn and maximum activation over the polyhedron
                        if minimum_activation_over_input is None:
                            # compute minimum and maximum activation over input vertices only once for each activation state
                            minimum_activation_over_input = +np.inf
                            maximum_activation_over_input = -np.inf
                            for vertex in input_vertices:
                                activation_value = (activation_state_weight @ (tilde_W[:, :dim_zero] @ vertex[:dim_zero] + tilde_b)).reshape(-1) + activation_state_bias.reshape(-1)
                                minimum_activation_over_input = min(minimum_activation_over_input, activation_value)
                                maximum_activation_over_input = max(maximum_activation_over_input, activation_value)
                            # now also add the minimum and maximum activation over the new variables introduced in previous layers
                            if domain.dimension > dim_zero: 
                                # we check if any of the new variables have infinite bounds; in that case, the min and max activation over input is unbounded
                                infinite_indices =  np.where(new_variables_bounds[:,1] == np.inf)[0]
                                negative_infinite_indices = np.where(new_variables_bounds[:,0] == -np.inf)[0]
                                positive_J_indices = activation_state_weight @ tilde_W >= 0
                                nonzero_infinite_row = np.any((activation_state_weight @ tilde_W * positive_J_indices)[:,infinite_indices + dim_zero] != 0,)
                                nonzero_negative_infinite_row = np.any((activation_state_weight @ tilde_W * positive_J_indices)[:,negative_infinite_indices + dim_zero] != 0,)
                                if np.any(nonzero_infinite_row) or np.any(nonzero_negative_infinite_row):
                                    minimum_activation_over_input = -np.inf
                                    maximum_activation_over_input = np.inf
                                else:
                                    # we remove the infinite columns for the following computation
                                    non_infinite_new_vars = new_variables_bounds[(new_variables_bounds[:,1] != np.inf) | (new_variables_bounds[:,0] != -np.inf)].reshape(-1,2)
                                    print("non infinite new vars:", non_infinite_new_vars)
                                    positive_J_indices = activation_state_weight @ tilde_W >= 0
                                    
                                    positive_matrix = np.delete((activation_state_weight @ tilde_W * positive_J_indices)[:, dim_zero:], infinite_indices, axis=1)
                                    negative_matrix = np.delete((activation_state_weight @ tilde_W * (1 - positive_J_indices))[:, dim_zero:], negative_infinite_indices, axis=1)
                                    minimum_posisitve_contribution = positive_matrix @ (non_infinite_new_vars[:,0]) if positive_matrix.shape[1] > 0 else 0
                                    maximum_positive_contribution = positive_matrix @ (non_infinite_new_vars[:,1]) if positive_matrix.shape[1] > 0 else 0
                                    minimum_negative_contribution = negative_matrix @ (non_infinite_new_vars[:,1]) if negative_matrix.shape[1] > 0 else 0
                                    maximum_negative_contribution = negative_matrix @ (non_infinite_new_vars[:,0]) if negative_matrix.shape[1] > 0 else 0
                                    min_new_vars = minimum_posisitve_contribution + minimum_negative_contribution + activation_state_bias
                                    max_new_vars = maximum_positive_contribution + maximum_negative_contribution + activation_state_bias
                                    #min_new_vars = positive_matrix @ (non_infinite_new_vars[:,0]) + \
                                    #            negative_matrix @ (non_infinite_new_vars[:,1]) + activation_state_bias
                                    #max_new_vars = positive_matrix @ (non_infinite_new_vars[:,1]) + \
                                    #            negative_matrix @ (non_infinite_new_vars[:,0]) + activation_state_bias
                                    minimum_activation_over_input += min_new_vars
                                    maximum_activation_over_input += max_new_vars
                        # compute min and max activation over the intersected polyhedron
                        min_activation_over_polyhedron, max_activation_over_polyhedron = activations_list[ell].compute_min_max_act_over_polyhedron_index(poly_index, neuron_index)
                        
                        # intersect with the previously computed min and max activation over input
                        if verbose:
                            print(f"Neuron {neuron_index} in layer {ell}, activation state {state_index}: min over polyhedron {min_activation_over_polyhedron}, max over polyhedron {max_activation_over_polyhedron}, min over input {minimum_activation_over_input}, max over input {maximum_activation_over_input}")
                        min_activation_over_polyhedron = max(min_activation_over_polyhedron, minimum_activation_over_input.item() if type(minimum_activation_over_input) != float else minimum_activation_over_input)
                        max_activation_over_polyhedron = min(max_activation_over_polyhedron, maximum_activation_over_input.item() if type(maximum_activation_over_input) != float else maximum_activation_over_input)
                        # update overall min and max activation
                        min_act_value = min(min_act_value, min_activation_over_polyhedron)
                        max_act_value = max(max_act_value, max_activation_over_polyhedron)

            if is_star:
                star_neurons_layer.append(neuron_index)
                new_variables.append((min_act_value, max_act_value))
                l_tilde = min(l_tilde, ell)
            else:
                # there is the off-chance that more than one activation state is possible but that this has no effect on the input polyhedron
                # in this case, we manually correct Lambda_layer and lambda_layer
                if (Lambda_layer[neuron_index, :, 0] != Lambda_layer[neuron_index, :, 1]).any() or (lambda_layer[neuron_index, 0] != lambda_layer[neuron_index, 1]).any():
                    Lambda_layer[neuron_index, :, 0] = activation_states_for_W_tilde[neuron_index,:]
                    Lambda_layer[neuron_index, :, 1] = activation_states_for_W_tilde[neuron_index,:]
                    lambda_layer[neuron_index, 0] = biases_for_b_tilde[neuron_index]
                    lambda_layer[neuron_index, 1] = biases_for_b_tilde[neuron_index]
        
        star_neurons.append(star_neurons_layer)

        activation_pattern_lambda.append(lambda_layer)
        activation_pattern_Lambda.append(Lambda_layer)

        # we now have to add new variables for the star neurons and update tilde_W, tilde_b
        new_tilde_W = np.zeros((tilde_W.shape[0], tilde_W.shape[1] + len(star_neurons_layer)))
        new_tilde_b = np.zeros(tilde_b.shape[0])

        # we first construct the new tilde_w:
        new_tilde_W[:,:tilde_W.shape[1]] = activation_states_for_W_tilde @ tilde_W
        for number, star_neuron_index in enumerate(star_neurons_layer):
            new_tilde_W[star_neuron_index, tilde_W.shape[1]+ number-1] = 1

        # now we construct the new tilde_b
        new_tilde_b = activation_states_for_W_tilde @ tilde_b + biases_for_b_tilde
        if ell < len(weights_list)-1:
            tilde_W = weights_list[ell+1][0] @ new_tilde_W
            tilde_b = weights_list[ell+1][0] @ new_tilde_b + weights_list[ell+1][1]

        # we add the new variables intervals to the domain
        if verbose:
            print("trying to add new variables with bounds:", new_variables)
        # if the variable bounds haven't been set at all, it might happen that we have an interval of (inf, -inf)
        # in that case, we set the bounds to (-inf, inf)
        for i in range(len(new_variables)):
            if new_variables[i][0] == np.inf and new_variables[i][1] == -np.inf:
                new_variables[i] = (-np.inf, np.inf)


        domain = domain.extend_by_new_vars(new_variables)
        if len(new_variables) > 0:
            new_variables_bounds = np.append(new_variables_bounds, np.array(new_variables), axis=0)
    if verbose:
        print(f"Variables created during symprop had bounds \n{new_variables_bounds}")

    return activation_pattern_Lambda, activation_pattern_lambda, l_tilde, star_neurons    


        

def lipschitz_bounds(sub_problem: SubProblem, weights: list, pnorm = 2):
    """
    Computes an upper bound on the Lipschitz constant for a given sub-problem.

    Parameters
    ----------
    sub_problem : SubProblem
        The sub-problem for which the Lipschitz upper bound should be computed.
    weights : list
        List of weight matrices of the neural network.
    pnorm : int
        The p norm for which the Lipschitz constant should be computed.

    Returns
    -------
    Lub : float
        Upper bound on the Lipschitz constant for the given sub-problem.
    """
    if sub_problem.tilde_l == len(weights):
        # all neurons fixed linear, the Lipschitz constant is exact, aplly the last activation layer and return
        assert np.all(sub_problem.activation_Lambda[-1][:,:,0] == sub_problem.activation_Lambda[-1][:,:,1])
        assert np.all(sub_problem.activation_lambda[-1][:,0] == sub_problem.activation_lambda[-1][:,1])
        U = sub_problem.activation_Lambda[-1][:,:,0] @ sub_problem.propagated_weights[0]
    else:
        # Jacobian not easily computable, approximate with interval matrix multiplication
        J = np.stack((sub_problem.propagated_weights[0], sub_problem.propagated_weights[0]), 2)
        J = interval_matrix_multiplication(sub_problem.activation_Lambda[sub_problem.tilde_l], J)
        for ell in range(sub_problem.tilde_l+1, len(weights)):
            W_ell = np.stack((weights[ell][0], weights[ell][0]), 2)
            Lambda_ell = sub_problem.activation_Lambda[ell]
            J = interval_matrix_multiplication(Lambda_ell, interval_matrix_multiplication(W_ell, J))
        U = np.abs(J).max(axis=2)
    # compute p-q norm of U
    Lub = np.linalg.norm(U, ord = pnorm)
    return Lub


def branch(sub_problem: SubProblem, old_glb: float, weights: list, 
           alpha: list[PiecewiseLinearFunction], pnorm, solver = 'glpk'):
    """
    Branching procedure for the ExLipBaB algorithm using symbolic propagation. In this versiion, the polytope is not explicitly
    propagated throught the network, but only the new half-space constraints are added to the sub-problem.

    Parameters
    ----------
    sub_problem : SubProblem
        The sub-problem to be branched.
    old_glb : float
        The current global lower bound of the ExLipBaB algorithm.
    weights : list
        List of tuples (W,b) of weight matrices and bias vectors of the neural network.
    alpha : list[PiecewiseLinearFunction]
        List of piecewise linear (activation) functions of the neural network.
    Returns
    -------
    sub_problems : list[SubProblem]
        List of new sub-problems created by branching the given sub-problem.
    glb : float
        Updated global lower bound after branching.
    """

    new_sub_problems = []
    # we selct the first star neuron for which we branch
    star_neuron_indices = sub_problem.star_neurons[sub_problem.tilde_l]
    if len(star_neuron_indices) == 0:
        raise ValueError(f"Tried to branch a subproblem with no star neurons. Problem has tilde_l {sub_problem.tilde_l} and star neurons {sub_problem.star_neurons}")
    # branch on the first star neuron:
    neuron_to_branch = star_neuron_indices[0]

    polyhedron_dict, activation_state_dict = alpha[sub_problem.tilde_l].get_polyhedron_decomposition(neuron_to_branch)

    for poly_index in polyhedron_dict.keys():
        # create a new polyhedron in input space by adding the new half-space constraints
        new_A_constraint = alpha[sub_problem.tilde_l].list_of_polyhedra[poly_index].A @ sub_problem.propagated_weights[0]
        new_b_constraint = alpha[sub_problem.tilde_l].list_of_polyhedra[poly_index].b - alpha[sub_problem.tilde_l].list_of_polyhedra[poly_index].A @ sub_problem.propagated_weights[1]

        polyhedron = Polyhedron(np.vstack((sub_problem.polyhedron.A, new_A_constraint)),
                                np.hstack((sub_problem.polyhedron.b, new_b_constraint)))
        # for activation functions with more than two possible activation states per neuron, it is possible to have
        # empty intersections here, in which case we skip this branch
        if polyhedron.is_empty():
            continue
        activation_Lambda = copy.deepcopy(sub_problem.activation_Lambda)
        activation_lambda = copy.deepcopy(sub_problem.activation_lambda)
        # update activation patterns for branched neuron as well as all other neurons set for the given polyhedron
        for other_neuron in polyhedron_dict[poly_index]:
            activation_Lambda[sub_problem.tilde_l][:, other_neuron, :] = activation_state_dict[poly_index][0][:, other_neuron]
            activation_lambda[sub_problem.tilde_l][other_neuron, :] = activation_state_dict[poly_index][1][other_neuron]
        new_subproblem = SubProblem(polyhedron, activation_Lambda, activation_lambda)

        new_subproblem.tilde_l = copy.deepcopy(sub_problem.tilde_l)
        new_subproblem.star_neurons = copy.deepcopy(sub_problem.star_neurons)
        new_subproblem.star_neurons[new_subproblem.tilde_l] = list(np.setdiff1d(new_subproblem.star_neurons[new_subproblem.tilde_l], polyhedron_dict[poly_index]))
        new_subproblem.propagated_weights = copy.deepcopy(sub_problem.propagated_weights)

        new_subproblem = new_subproblem.ffilter_prop(weights, alpha, solver = solver)
        
        new_subproblem.upper_bound = lipschitz_bounds(new_subproblem,  weights, pnorm=pnorm)

        new_sub_problems.append(new_subproblem)
        if new_subproblem.tilde_l == len(weights):
            old_glb = max(old_glb, new_subproblem.upper_bound)
    return new_sub_problems, old_glb




