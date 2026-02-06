from abc import abstractmethod
from itertools import permutations
import numpy as np
from exlipbab.helper_classes.polyhedron import Polyhedron, FullPolyhedron
from cvxopt import matrix, solvers
import mosek

solvers.options['show_progress'] = False

class PiecewiseLinearFunction:
    """
    Class representing a piecewise linear function used in neural networks.
    Contains the list of linear regions as polyhedra and the corresponding linear mappings as a separate list.
    Also contains methods needed in the LipBaB algorithm.
    """

    def __init__(self, dimensionality: int, function_name: str) -> None:
        """
        Parameters
        ----------
        function_name: str
            String with which the piecewise linear function can be conveniently initialized. 
            If a custom one is used, set to "custom".
        """
        self.dimensionality = dimensionality
        self.function_name = function_name
        self.neuron_extreme_values = (-np.inf, np.inf)  # tuple representing the extreme values that neurons of this type can take; helpful in get_min_max_activation
        self.polyhedronNeuronRelDict = self.generate_polyhedron_neuron_relationship_dict(dimensionality)

    def get_all_possible_activation_states(self, neuron_index):
        """
        Get all possible linear functions rows that represent the activation states of the specified neuron in 
        the linear functions that make up the piecewise linear function.
        
        Parameters
        ----------
        neuron_index : int
            Index of the neuron for which the possible activation states should be computed.
            
        Returns
        -------
        activation_states: list[tuple]
            List of tuples (W,b) representing the possible linear function rows corresponding to the specified neuron.
            W is a numpy array of dimension (1 x dimensionality)
            and b is a float.
            This represents the activation patterns for each region.
        polyhedron_list: list[list[int]]
            List of lists indices of polyhedra such that for each state the union of the polyhedra in the corresponding list corresponds exactly to the regions where the neuron
            with index neuron_index has the specified activation states.
        """
        activation_states = []
        polyhedron_list = []

        for i, polyhedron in enumerate(self.list_of_polyhedra):
            # check that the activation state of the neuron is fixed in this polyhedron
            if (self.list_of_linear_functions[i][0][neuron_index,:, 0] == self.list_of_linear_functions[i][0][neuron_index,:, 1]).all() and \
                                    (self.list_of_linear_functions[i][1][neuron_index, 0] == self.list_of_linear_functions[i][1][neuron_index, 1]).all():
                activation_state_weight = self.list_of_linear_functions[i][0][neuron_index,:, 0]
                activation_state_bias = self.list_of_linear_functions[i][1][neuron_index, 0]
                activation_state_in_dict = False
                index = -1
                for j, activation_state in enumerate(activation_states):
                    if np.all(activation_state[0] == activation_state_weight) and activation_state[1] == activation_state_bias:
                        activation_state_in_dict = True
                        index = j
                        break
                if activation_state_in_dict:
                    polyhedron_list[index].append(i)
                    
                else:
                    activation_states.append((activation_state_weight.reshape(1, -1), activation_state_bias.reshape(1,)))
                    polyhedron_list.append([i])
                    
        
        return activation_states, polyhedron_list
                


        pass 

    def get_neurons_for_polyhedron(self, polyhedron_index: int):
        return self.polyhedronNeuronRelDict.polyhedra_to_neurons.get(polyhedron_index, [])
    
    def get_polyhedra_for_neuron(self, neuron_index: int):
        return self.polyhedronNeuronRelDict.neuron_to_polyhedra.get(neuron_index, [])
                
    def compute_min_max_act_over_polyhedron_index(self, polyhedron_index: int, neuron_index: int):
        """
        Helper function to compute the min and max activation of a neuron over a specific polyhedron region.
        This is used in symbolic propagation as an alternative to intersecting with the propagated polyhedra.

        Parameters
        ----------
        polyhedron_index : int
            Index of the polyhedron in list_of_polyhedra.
        neuron_index : int
            Index of the neuron for which the min and max pre-activation values should be computed.
        """
        polyhedron_vertices = self.list_of_polyhedra[polyhedron_index].compute_vertices()
        min_activation_over_polyhedron = np.inf
        max_activation_over_polyhedron = -np.inf
        activation_state_weight = self.list_of_linear_functions[polyhedron_index][0][neuron_index, :, 0]
        activation_state_bias = self.list_of_linear_functions[polyhedron_index][1][neuron_index, 0]
        for vertex in polyhedron_vertices:
            activation_value = activation_state_weight @ vertex + activation_state_bias
            min_activation_over_polyhedron = min(min_activation_over_polyhedron, activation_value)
            max_activation_over_polyhedron = max(max_activation_over_polyhedron, activation_value)
        return min_activation_over_polyhedron, max_activation_over_polyhedron
        


    
    def get_activation_pattern(self, polyhedron: Polyhedron, old_star_neurons:list|None = None,
                               old_Lambda = None, old_lambda = None, propagated_weights: tuple = None, solver: str = "glpk") -> tuple:
        """
        
        Parameters
        ----------
        polyhedron : Polyhedron
            Polyhedron representing the input.

        Returns
        -------
        symbolic_post_activation_polyhedron : Polyhedron
            Polyhedron representing the symbolic post-activation output.
        activation_pattern_Lambda : np.array
            Interval matrix representing the weights activation pattern for the current layer.
        activation_patternlambda : np.array
            Interval vector representing the bias activation pattern for the current layer.
        star_neurons : np.array
            Array indicating which neurons are star neurons in the current layer.
        """

        # For now, we will simply compute the min max values for each neuron separately
        # and accumulate the activation pattern accordingly.
        propagated_dimension = propagated_weights[0].shape[0]
        activation_pattern_Lambda_ell = np.zeros((propagated_dimension, propagated_dimension, 2)) if old_Lambda is None else old_Lambda
        activation_pattern_lambda_ell = np.zeros((propagated_dimension, 2)) if old_lambda is None else old_lambda
        star_neurons = list(range(propagated_dimension)) if old_star_neurons is None else old_star_neurons
        neuron_ranges = []

        for neuron_id in range(propagated_dimension):
            
            neuron_activation_pattern, neuron_activation_bias_pattern, output_bounds, is_star = \
                self.get_min_max_activation(polyhedron, neuron_id, propagated_weights=propagated_weights, solver=solver)
            neuron_ranges.append(output_bounds)
            if neuron_id in star_neurons:
                activation_pattern_Lambda_ell[neuron_id, :, :] = neuron_activation_pattern
                activation_pattern_lambda_ell[neuron_id, :] = neuron_activation_bias_pattern
                if not is_star:
                    star_neurons.remove(neuron_id)
                
        return activation_pattern_Lambda_ell, activation_pattern_lambda_ell, star_neurons
    
    
    def get_min_max_activation(self, polyhedron: Polyhedron, neuron_index: int, propagated_weights: tuple = None, solver: str = "glpk"):
        """
        Parameters
        ----------
        polyhedron : Polyhedron
            Polyhedron representing the symbolic pre-activation input.
        neuron_index : int
            Index of the neuron for which the min and max pre-activation values should be computed.

        Returns
        -------
        activation_min_max: np.array
            Numpy of dimension (num_neurons x 2) representing the minimum and maximum pre-activation values for the specified neuron.
        activation_bias_min_max: np.array
            Numpy of dimension (2,) representing the minimum and maximum bias values for the specified neuron.
        output_bounds: tuple
            Minimum and maximum value of neuron
        is_star: bool
            Indicates whether the specified neuron is a star neuron for the given input polyhedron.
        solver : str, optional
            Solver to be used in the linear programming steps, default is "glpk".
        """
        polyhedron_dict, activation_state_dict = self.get_polyhedron_decomposition(neuron_index)
        activation_min_max = np.stack([np.ones((propagated_weights[0].shape[0],))*np.inf,
                                       np.ones((propagated_weights[0].shape[0],))*-np.inf], axis=1)
        activation_bias_min_max = np.array([np.inf, -np.inf])
        minimum_neuron_value = np.inf
        maximum_neuron_value = -np.inf
        is_star = False
        intersection_counter = 0
        for poly_id in polyhedron_dict.keys():
            # we now just check if there is a non-empty intersection by stacking the half-space constraints
            new_A_constraint = self.list_of_polyhedra[poly_id].A @ propagated_weights[0]
            new_b_constraint = self.list_of_polyhedra[poly_id].b - self.list_of_polyhedra[poly_id].A @ propagated_weights[1]
            intersection_polytope = Polyhedron(np.vstack([polyhedron.A, new_A_constraint]),
                                               np.hstack([polyhedron.b, new_b_constraint]))
            if not intersection_polytope.is_empty():
                intersection_counter += 1
                if intersection_counter > 1:
                    is_star = True
                W, b = activation_state_dict[poly_id]
                # The activation state of this neuron should now be fixed in this polyhedron:
                if not (np.all(W[neuron_index, :, 0] == W[neuron_index, :, 1]) and
                        b[neuron_index, 0] == b[neuron_index, 1]):
                    raise ValueError("The activation state of the specified neuron is not fixed in this polyhedron."
                    "Probably unwanted behavior occurred.")
                # we only use the row with index neuron_index of W and the entry with index neuron_index of b
                used_W = W[neuron_index, :, 0]
                used_b = b[neuron_index, 0]
                # especially for ReLu networks, W will often be zero, leading to constant activation patterns
                if np.all(used_W == 0):
                    minimum_neuron_value = min(used_b, minimum_neuron_value)
                    maximum_neuron_value = max(used_b, maximum_neuron_value)
                    activation_min_max[:,0] = np.minimum(activation_min_max[:,0], used_W)
                    activation_min_max[:,1] = np.maximum(activation_min_max[:,1], used_W)
                    activation_bias_min_max[0] = np.minimum(activation_bias_min_max[0], used_b)
                    activation_bias_min_max[1] = np.maximum(activation_bias_min_max[1], used_b)
                    continue
                
                # the result of the transformation and projection on only one neuron should be an interval for each neuron
                full_used_weights = used_W @ propagated_weights[0]
                full_used_bias = used_W @ propagated_weights[1] + used_b
                
                
                # we construct a simple linear program to minimize and maximize the neuron output over the intersection polytope
                c_cvx = matrix(full_used_weights)
                A_ub_cvx = matrix(intersection_polytope.A)
                b_ub_cvx = matrix(intersection_polytope.b)

                # solve minimization problem
                #
                # we turn off the output of both solvers here to avoid cluttering the output
                res_min = solvers.lp(c_cvx, A_ub_cvx, b_ub_cvx, solver= solver, options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}, 'mosek': {mosek.iparam.log: 0}})
                res_max = solvers.lp(-c_cvx, A_ub_cvx, b_ub_cvx , solver= solver, options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}, 'mosek': {mosek.iparam.log: 0}})

                #If anything but 'optimal' is returned, we set the max and min values to their extreme values
                if res_min['status'] != 'optimal':
                    min_value = self.neuron_extreme_values[0]
                else:
                    min_value = res_min['primal objective'] + full_used_bias

                if res_max['status'] != 'optimal':
                    max_value = self.neuron_extreme_values[1]
                else:
                    max_value = -res_max['primal objective'] + full_used_bias

                minimum_neuron_value = min(minimum_neuron_value, min_value)
                maximum_neuron_value = max(maximum_neuron_value, max_value)
                
                # subsume the activation pattern into the min_max pattern
                activation_min_max[:,0] = np.minimum(activation_min_max[:,0], used_W)
                activation_min_max[:,1] = np.maximum(activation_min_max[:,1], used_W)
                activation_bias_min_max[0] = np.minimum(activation_bias_min_max[0], used_b)
                activation_bias_min_max[1] = np.maximum(activation_bias_min_max[1], used_b)
        output_bounds = (minimum_neuron_value, maximum_neuron_value) 
        return activation_min_max, activation_bias_min_max, output_bounds, is_star


    @abstractmethod
    def get_polyhedron_decomposition(self, neuron_index: int) -> tuple:
        """
        Get a polyhedron deomposition of the pre-activation space such that the specified neuron is fixed linear for each 
        of these polyhedra. See paper, Sec. 1.
        
        Parameters
        ----------
        neuron_index : int
            Index of the neuron for which the polyhedron decomposition should be computed.
        Returns
        -------
        decomposition_polyhedra : list of Polyhedron
            List of polyhedra representing the decomposition of the pre-activation space.
        decomposition_linear_functions : list of tuple
            List of tuples (W,b) representing the linear functions corresponding to each region.
            W is a numpy array of dimension (1 x dimensionality)
            and b is a float.
            This represents the activation patterns for each region.
        """
        raise NotImplementedError("This method should be implemented in the child class.")      

    @abstractmethod
    def _generate_polyhedra(self, dimensionality: int) -> tuple[list[Polyhedron], list[tuple]]:
        """
        Generate the list of polyhedra representing the linear regions of the piecewise linear function as 
        well as the list of corresponding linear patterns (e.g. activation patterns). Usually called during initialization.
        
        Parameters
        ----------
        dimensionality : int
            Dimensionality of the input space of the piecewise linear function.
        
        Returns
        -------
        list_of_polyhedra : list[Polyhedron]
            List of polyhedra representing the linear regions of the piecewise linear function.
        list_of_linear_functions : list[tuple]
            List of tuples (W,b) representing the possible linear function(s), corresponding to each region.
            W and b are interval matrices/vectors representing the activation patterns for each region. More specifically,
            W is a numpy array of dimension (dimensionality x dimensionality x 2) and b is a numpy array of dimension (dimensionality x 2).       
        
        """
        raise NotImplementedError("This method should be implemented in the child class.")
    
    def generate_polyhedron_neuron_relationship_dict(self, dimensionality: int, **kwargs):
        raise NotImplementedError("This method should be implemented in the child class.")
    
    def __repr__(self):
        return f"PiecewiselinearFunction with polyhedra {self.list_of_polyhedra} and linear functions {self.list_of_linear_functions}"
    

class PWL_Identity(PiecewiseLinearFunction):
    """
    Class representing the Identity function as a PiecewiseLinearFunction.
    """

    def __init__(self, dimensionality: int) -> None:
        super().__init__(dimensionality=dimensionality, function_name="Identity")
        self.list_of_polyhedra, self.list_of_linear_functions = self._generate_polyhedra(dimensionality)

    def generate_polyhedron_neuron_relationship_dict(self, dimensionality: int, **kwargs):
        #the identity only has one polyhedron, which determines all neurons
        polyhedronNeuronRelDict = BiDirectionalDict()
        neuron_to_polyhedra_dict = {i: [0] for i in range(dimensionality)}
        polyhedra_to_neurons_dict = {0: list(range(dimensionality))}
        polyhedronNeuronRelDict.neuron_to_polyhedra = neuron_to_polyhedra_dict
        polyhedronNeuronRelDict.polyhedra_to_neurons = polyhedra_to_neurons_dict
        return polyhedronNeuronRelDict
        

    def _generate_polyhedra(self, dimensionality):
        list_of_polyhedra = [FullPolyhedron(dimensionality)]
        list_of_linear_functions = [(np.repeat(np.eye(dimensionality)[:,:,np.newaxis], 2, axis=2), np.zeros((dimensionality,2)))]
        return list_of_polyhedra, list_of_linear_functions

    def get_polyhedron_decomposition(self, neuron_index: int) -> dict:
        """
        Get a polyhedron deomposition of the pre-activation space such that the specified neuron is fixed linear for each 
        of these polyhedra. See paper, Sec. 1.
        For the Identity functon, we can return the entire space as one polyhedron.
        
        Parameters
        ----------
        neuron_index : int
            Index of the neuron for which the polyhedron decomposition should be computed.
        Returns
        -------
        polyhedron_dict: dict
            Dictionary of the polyhedron indices that make up the decomposition and a list of neuron indices whose neurons are determined
            for each polyhedron.
        activation_state_dict: dict[tuple]
            Dictionary containing for each polyhedron the activation state (Lambda, lambda)
        """
        polyhedron_dict = {0: list(range(self.dimensionality))}
        activation_state_dict = {0: self.list_of_linear_functions[0]}
        return polyhedron_dict, activation_state_dict

    
class PWL_Relu(PiecewiseLinearFunction):
    """
    Class representing ReLu as a piecewise linear function.
    """

    def __init__(self, dimensionality: int) -> None:
        super().__init__(dimensionality=dimensionality, function_name="ReLu")
        self.list_of_polyhedra, self.list_of_linear_functions = self._generate_polyhedra(dimensionality)
        self.polyhedronNeuronRelDict = self.generate_polyhedron_neuron_relationship_dict(dimensionality)


    
    def get_all_possible_activation_states(self, neuron_index):
        # for the relu function, there are only two possible activation states: active and inactive for each neuron
        # we return deterministic matrices instead of interval matrices here
        active_activation_state = (self.list_of_linear_functions[2*neuron_index][0][neuron_index, :, 1].reshape(1, -1),
                                      self.list_of_linear_functions[2*neuron_index][1][neuron_index, 1].reshape(1,))
        inactive_activation_state = (self.list_of_linear_functions[2*neuron_index + 1][0][neuron_index, :, 0].reshape(1, -1),
                                        self.list_of_linear_functions[2*neuron_index + 1][1][neuron_index, 0].reshape(1,))
        all_possible_activation_states = [active_activation_state, inactive_activation_state]

        # the corresponding polyhedra are the positive and negative half-spaces respectively
        polyhedron_list = [[2*neuron_index], [2*neuron_index + 1]]

        return all_possible_activation_states, polyhedron_list

    
    def compute_min_max_act_over_polyhedron_index(self, polyhedron_index: int, neuron_index:int):
        # for the Relu, the polyhedra bound simplify drastically:

        #the computation only makes sense if the activation state of the neuron is fixed in the polyhedron
        if not (polyhedron_index == 2*neuron_index or polyhedron_index == 2*neuron_index + 1):
            raise ValueError("The activation state of the specified neuron is not fixed in this polyhedron."
            "Probably unwanted behavior occurred.")

        # for the relu function, the min and max for the inactive states are always 0
        if polyhedron_index == 2*neuron_index + 1:
            return 0.0, 0.0
        else:
            #in the active state, the min is 0 and the max is unbounded
            return 0.0, np.inf
        

    def generate_polyhedron_neuron_relationship_dict(self, dimensionality: int, **kwargs):
        # generate the bidirectional dictionary mapping neurons to polyhedra and vice versa
        polyhedronNeuronRelDict = BiDirectionalDict()
        neuron_to_polyhedra_dict = {i: [2*i, 2*i + 1] for i in range(dimensionality)}
        polyhedra_to_neurons_dict = {2*i: [i] for i in range(dimensionality)}
        polyhedra_to_neurons_dict.update({2*i + 1: [i] for i in range(dimensionality)})
        polyhedronNeuronRelDict.neuron_to_polyhedra = neuron_to_polyhedra_dict
        polyhedronNeuronRelDict.polyhedra_to_neurons = polyhedra_to_neurons_dict
        return polyhedronNeuronRelDict

    def get_polyhedron_decomposition(self, neuron_index: int):
        """
        Get a polyhedron deomposition of the pre-activation space such that the specified neuron is fixed linear for each 
        of these polyhedra. See paper, Sec. 1.
        For the Relu functon, we can return a decomposition into two half spaces for each neuron, where the
        neuron is active on one and inactive on the other.
        
        Parameters
        ----------
        neuron_index : intimport pypoman
            Index of the neuron for which the polyhedron decomposition should be computed.
        Returns
        -------
        polyhedron_dict: dict
            Dictionary of the polyhedron indices that make up the decomposition and a list of neuron indices whose neurons are determined
            for each polyhedron.
        activation_state_dict: dict[tuple]
            Dictionary containing for each polyhedron the activation state (Lambda, lambda)
        """
        polyhedron_dict = {2*neuron_index: [neuron_index], 2*neuron_index + 1: [neuron_index]}
        # for ReLu, we know that the polyhedra do not set the activation state of other neurons, 
        # for other piecewise linear functions, this needs to be checked
        
        activation_state_dict = {2*neuron_index: self.list_of_linear_functions[2*neuron_index],
                                  2*neuron_index + 1: self.list_of_linear_functions[2*neuron_index + 1]}
        
        return polyhedron_dict, activation_state_dict

        

    def _generate_polyhedra(self, dimensionality: int):
        list_of_polyhedra = []
        list_of_linear_functions = []
        # for ReLu, there are 2^dimensionality regions in total

        for i in range(dimensionality):
            # each neuron can be either active or inactive, generate the corresponding polyhedra:
            active_polytope  = Polyhedron.from_intervals([(0, np.inf) if (i == j) else (-np.inf,np.inf) for j in range(dimensionality)])
            inactive_polytope = Polyhedron.from_intervals([(-np.inf, 0) if (i == j) else (-np.inf,np.inf) for j in range(dimensionality)])
            
            # create the interval matrices representing the activation patterns
            #For the ReLu function, the bias is always zero.
            b_active = np.zeros((dimensionality, 2))
            # The activation state of componentwise activations like ReLu is zero on the off-diagonal entries
            # On the active region, the activation state of the i-th neuron is one (active), all others are undefined (zero or one)
            W_active = np.stack([np.zeros((dimensionality, dimensionality)), np.eye(dimensionality)], axis=2)
            W_active[i,i,:] = 1
            list_of_polyhedra.append(active_polytope)
            list_of_linear_functions.append((W_active, b_active))

            #Same for the inactive region, here the i-th neuron is inactive per definition
            W_inactive = np.stack([np.zeros((dimensionality, dimensionality)), np.eye(dimensionality)], axis=2)
            W_inactive[i,i,:] = 0
            b_inactive = np.zeros((dimensionality, 2))
            list_of_polyhedra.append(inactive_polytope)
            list_of_linear_functions.append((W_inactive, b_inactive))
        return list_of_polyhedra, list_of_linear_functions
    

class BiDirectionalDict():
    """
    In our general PiecewiseLinearFunction, any neuron might be exactly specified in a number of polyhedra in the given list of polyhedra.
    At the same time, each polyhedron might exactly specify the activation state of a number of neurons.
    This class implements a bidirectional dictionary to conveniently store and access this information.
    """

    def __init__(self) -> None:
        self._neuron_to_polyhedra = {}
        self._polyhedra_to_neurons = {}

    @property
    def neuron_to_polyhedra(self) -> dict:
        return self._neuron_to_polyhedra
    
    @property
    def polyhedra_to_neurons(self) -> dict:
        return self._polyhedra_to_neurons
    
    @neuron_to_polyhedra.setter
    def neuron_to_polyhedra(self, dictionary: dict) -> None:
        self._neuron_to_polyhedra = dictionary
    
    @polyhedra_to_neurons.setter
    def polyhedra_to_neurons(self, dictionary: dict) -> None:
        self._polyhedra_to_neurons = dictionary

    def get_neurons_for_polyhedron(self, polyhedron_index: int) -> list:
        return self.polyhedra_to_neurons[polyhedron_index]
    
    def get_polyhedra_for_neuron(self, neuron_index: int) -> list:
        return self.neuron_to_polyhedra[neuron_index]
    

class Componentwise_PWL(PiecewiseLinearFunction):
    """
    Class representing a PWL function that acts componentwise, i.e. each neuron is tranformed
    independently of all others with a corresponding 1-dim PWL function represented by corresponding interval bounds
    and linear weights.
    We allow for different 1-dim PWL functions for each neuron.
    """

    def __init__(self, dimensionality: int, one_d_pwl: tuple|list[tuple], **kwargs) -> None:
        """
        Intialize the componentwise PWL function.

        Parameters
        ----------
            one_d_pwl : tuple|list[tuple]
                Either a single tuple (list_of_interval_bounds, list_of_weights) or a list of length dimensionality
                containing such tuples for each neuron.
                list_of_interval_bounds should contain all breakpoints of the 1-dim PWL function as a list of floats.
                If -inf and inf are not included, they will be added automatically.
                list_of_weights should contain the corresponding linear weights for each interval as a
                list of tuples (bias, slope), each a float.
        """
        super().__init__(dimensionality=dimensionality, 
                     function_name="Componentwise_PWL" if "function_name" not in kwargs else kwargs["function_name"])
        if isinstance(one_d_pwl, list):
            if len(one_d_pwl) != dimensionality:
                raise ValueError("If a list of 1-dim PWL functions is provided, its length must match the dimensionality.")
        elif isinstance(one_d_pwl, tuple):
                one_d_pwl = [one_d_pwl]*dimensionality
        else:
                raise ValueError("one_d_pwl must be either a tuple or a list of tuples.")
        
        for i, (list_of_interval_bounds, list_of_weights) in enumerate(one_d_pwl):
            if list_of_interval_bounds[0] != -np.inf and list_of_interval_bounds[0] != float('-inf'):
                one_d_pwl[i] = ([-np.inf] + one_d_pwl[i][0], list_of_weights)
            if list_of_interval_bounds[-1] != np.inf and list_of_interval_bounds[-1] != float('inf'):
                one_d_pwl[i] = (one_d_pwl[i][0] + [np.inf], list_of_weights)
            if len(one_d_pwl[i][0]) -1  != len(list_of_weights):
                raise ValueError("The number of interval bounds must be larger by one than the number of weights.")
        self.one_d_pwl = one_d_pwl

        self.list_of_polyhedra, self.list_of_linear_functions = self._generate_polyhedra(dimensionality, one_d_pwl)
        #self.polyhedronNeuronRelDict = self.generate_polyhedron_neuron_relationship_dict(dimensionality)

    def _generate_polyhedra(self, dimensionality: int, one_d_pwl: list[tuple]) -> tuple[list[Polyhedron], list[tuple]]:
        list_of_polyhedra = []
        list_of_linear_functions = []
        polyhedra_to_neurons_bidict = BiDirectionalDict()
        polyhedron_to_neurons_dict = {}
        neuron_to_polyhedra_dict = {}

        # we first create a list of minimum and maximum slopes and biases for each neuron
        # that we can use to create the interval matrices later in cases where a neuron is not fixed in a polyhedron
        min_slopes = np.inf * np.ones((dimensionality,))
        max_slopes = -np.inf * np.ones((dimensionality,))
        min_biases = np.inf * np.ones((dimensionality,))
        max_biases = -np.inf * np.ones((dimensionality,))
        for neuron_index in range(dimensionality):
            list_of_interval_bounds, list_of_weights = one_d_pwl[neuron_index]
            for (bias, slope) in list_of_weights:
                min_slopes[neuron_index] = min(min_slopes[neuron_index], slope)
                max_slopes[neuron_index] = max(max_slopes[neuron_index], slope)
                min_biases[neuron_index] = min(min_biases[neuron_index], bias)
                max_biases[neuron_index] = max(max_biases[neuron_index], bias)
        
        # now we create the polyhedra as described in the supplementary material of the paper, 
        # where each neuron is fixed in exactly len(list_of_weights) regions

        for neuron_index in range(dimensionality):
            neuron_to_polyhedra_dict[neuron_index] = []
            for j, (bias, slope) in enumerate(one_d_pwl[neuron_index][1]):
                polyhedron_bounds = [(-np.inf, np.inf)] * dimensionality
                polyhedron_bounds[neuron_index] = (one_d_pwl[neuron_index][0][j], one_d_pwl[neuron_index][0][j+1])
                polyhedron = Polyhedron.from_intervals(polyhedron_bounds)
                list_of_polyhedra.append(polyhedron)
                # create the interval matrices representing the activation patterns
                b = np.stack([min_biases, max_biases], axis=1)
                b[neuron_index, 0] = bias
                b[neuron_index, 1] = bias
                W = np.stack((np.diag(min_slopes), np.diag(max_slopes)), axis=2)
                W[neuron_index, neuron_index, 0] = slope
                W[neuron_index, neuron_index, 1] = slope
                list_of_linear_functions.append((W, b))
                polyhedron_to_neurons_dict[len(list_of_polyhedra)-1] = [neuron_index]
                neuron_to_polyhedra_dict[neuron_index].append(len(list_of_polyhedra)-1)
        polyhedra_to_neurons_bidict.polyhedra_to_neurons = polyhedron_to_neurons_dict
        polyhedra_to_neurons_bidict.neuron_to_polyhedra = neuron_to_polyhedra_dict
        self.polyhedronNeuronRelDict = polyhedra_to_neurons_bidict
        return list_of_polyhedra, list_of_linear_functions
    
    def generate_polyhedron_neuron_relationship_dict(self, dimensionality: int, **kwargs):
        # already generated during generation of polyhedra
        #return self.polyhedronNeuronRelDict
        pass
    
    def get_all_possible_activation_states(self, neuron_index):
        polyhedron_indices = self.polyhedronNeuronRelDict.get_polyhedra_for_neuron(neuron_index)
        all_possible_activation_states = [(self.list_of_linear_functions[poly_id][0][neuron_index, :,0].reshape(1, -1),
                                           self.list_of_linear_functions[poly_id][1][neuron_index, 0]) for poly_id in polyhedron_indices]

        return all_possible_activation_states, [[poly_id] for poly_id in polyhedron_indices]
    
    def get_polyhedron_decomposition(self, neuron_index):
        """
        Get a polyhedron deomposition of the pre-activation space such that the specified neuron is fixed linear for each 
        of these polyhedra.
        For a componentwise PWL function, we can return a decomposition into a number of hyper-rectangles for each neuron.
        
        Parameters
        ----------
        neuron_index : int
            Index of the neuron for which the polyhedron decomposition should be computed.
        Returns
        -------
        polyhedron_dict: dict
            Dictionary of the polyhedron indices that make up the decomposition and a list of neuron indices whose neurons are determined
            for each polyhedron.
        activation_state_dict: dict[tuple]
            Dictionary containing for each polyhedron the activation state (Lambda, lambda)
        """
        polyhedron_indices = self.polyhedronNeuronRelDict.get_polyhedra_for_neuron(neuron_index)
        polyhedron_dict = {poly_id: [neuron_index] for poly_id in polyhedron_indices}
        activation_state_dict = {poly_id: self.list_of_linear_functions[poly_id] for poly_id in polyhedron_indices}
        return polyhedron_dict, activation_state_dict
    
    def compute_min_max_act_over_polyhedron_index(self, polyhedron_index: int, neuron_index:int):
        """
        For a componentwise PWL function, we can simply apply the 1-dim linear function 
        on the borders of the 1-dim interval to get the min and max activation values for that neuron.
        """

        # computation only makes sense if the activation state of the neuron is fixed in the polyhedron
        if neuron_index not in self.polyhedronNeuronRelDict.get_neurons_for_polyhedron(polyhedron_index):
            raise ValueError("The activation state of the specified neuron is not fixed in this polyhedron."
            "Probably unwanted behavior occurred.")
        # we can read the min and max of the polyhedron directly from its half-space representation
        # as only one dimension is bounded; possibly only in one direction
        polyhedron = self.list_of_polyhedra[polyhedron_index]
        bound_1 = polyhedron.b[np.where(polyhedron.A[:, neuron_index] == 1)[0][0]] if (polyhedron.A == 1).any() else np.inf
        bound_2 = polyhedron.b[np.where(polyhedron.A[:, neuron_index] == -1)[0][0]] if (polyhedron.A == -1).any() else -np.inf
        min_input = min(bound_1, bound_2)
        max_input = max(bound_1, bound_2)

        slope = self.list_of_linear_functions[polyhedron_index][0][neuron_index, neuron_index, 0]
        bias = self.list_of_linear_functions[polyhedron_index][1][neuron_index, 0]

        output_bound_1 = slope * min_input + bias
        output_bound_2 = slope * max_input + bias
        return min(output_bound_1, output_bound_2), max(output_bound_1, output_bound_2)
    

class Componentwise_Relu(Componentwise_PWL):
    """
    Class representing ReLu as a subclass of Componentwise_PWL.
    """
    
    def __init__(self, dimensionality: int) -> None:
        # define the 1-dim ReLu function
        list_of_interval_bounds = [0.0]
        list_of_weights = [(0.0, 0.0), (0.0, 1.0)]  # (bias, slope) for each interval
        one_d_pwl = (list_of_interval_bounds, list_of_weights)
        super().__init__(dimensionality=dimensionality, one_d_pwl=one_d_pwl, function_name="Componentwise_ReLu")


class LeakyReLu(Componentwise_PWL):
    """
    Class representing LeakyReLu as a subclass of Componentwise_PWL.
    """
    
    def __init__(self, dimensionality: int, leak_factor: float = 0.01) -> None:
        # define the 1-dim LeakyReLu function
        list_of_interval_bounds = [0.0]
        list_of_weights = [(0.0, leak_factor), (0.0, 1.0)]  # (bias, slope) for each interval
        one_d_pwl = (list_of_interval_bounds, list_of_weights)
        super().__init__(dimensionality=dimensionality, one_d_pwl=one_d_pwl, function_name="LeakyReLu")


class GroupSort(PiecewiseLinearFunction):
    """
    Class representing the GroupSort activation function as a PiecewiseLinearFunction.
    """

    def __init__(self, dimensionality: int, group_size: int = 2) -> None:
        super().__init__(dimensionality=dimensionality, function_name="GroupSort")
        if dimensionality % group_size != 0:
            raise ValueError("Dimensionality must be divisible by group size.")
        self.group_size = group_size
        self.list_of_polyhedra, self.list_of_linear_functions = self._generate_polyhedra(dimensionality)

    def _generate_polyhedra(self, dimensionality):
        list_of_polyhedra = []
        list_of_linear_functions = []
        num_groups = dimensionality // self.group_size
        polyhedra_to_neurons_bidict = BiDirectionalDict()
        polyhedron_to_neurons_dict = {}
        neuron_to_polyhedra_dict = {i:[] for i in range(dimensionality)}

        for i in range(num_groups):
            for perm in permutations(range(self.group_size)):
                # first, create the low-dimensional polyhedron corresponding to this permutation
                A = np.zeros((self.group_size - 1, self.group_size))
                b = np.zeros((self.group_size - 1,))
                A[range(self.group_size - 1), perm[0:self.group_size - 1]] = 1
                A[range(self.group_size - 1), perm[1:self.group_size]] = -1
                
                low_dim_polyhedron = Polyhedron(A, b)
                # now, extend this to the full dimensionality by taking the cartesian product
                # Essentially, we just prepend and append zero columns to A until it has the correct dimensionality
                polyhedrdon_A = np.zeros((A.shape[0], dimensionality))
                polyhedrdon_A[:, i*self.group_size:(i+1)*self.group_size] = A
                polyhedron = Polyhedron(polyhedrdon_A, b)
                list_of_polyhedra.append(polyhedron)

                # now, create the corresponding linear function as an interval matrix
                # the activation state of all neurons not in the current group is undefined (min slope 0, max slope 1)
                W = np.zeros((dimensionality, dimensionality, 2))
                b = np.zeros((dimensionality, 2))
                # for the maximum activation, only the neurons in each block can affect each other, we therefore 
                # construct a block diagonal matrix with block of one-matrices on the diagonal
                for j in range(num_groups):
                    if j != i:
                        W[j*self.group_size:(j+1)*self.group_size, j*self.group_size:(j+1)*self.group_size, 1] = np.ones((self.group_size, self.group_size))
                
                # set the activation state of the neurons in the current group according to the permutation matrix
                permutation_matrix = (np.repeat(np.array(range(self.group_size)).reshape(-1,1), self.group_size, axis=1) ==
                                      np.repeat(np.array(perm).reshape(1,-1), self.group_size, axis=0)).astype(int)
                # we pad the columns of the permutation matrix into the correct position in W
                permutation_matrix = np.hstack((np.zeros((self.group_size, i*self.group_size)), permutation_matrix,
                                                np.zeros((self.group_size, dimensionality - (i+1)*self.group_size))))
                W[i*self.group_size:(i+1)*self.group_size, :, :] = np.repeat(permutation_matrix[:, :, np.newaxis], 2, axis=2)
                list_of_linear_functions.append((W, b))
                polyhedron_to_neurons_dict[len(list_of_polyhedra)-1] = list(range(i*self.group_size, (i+1)*self.group_size))
                for neuron_index in range(i*self.group_size, (i+1)*self.group_size):
                    neuron_to_polyhedra_dict[neuron_index].append(len(list_of_polyhedra)-1)
        polyhedra_to_neurons_bidict.polyhedra_to_neurons = polyhedron_to_neurons_dict
        polyhedra_to_neurons_bidict.neuron_to_polyhedra = neuron_to_polyhedra_dict
        self.polyhedronNeuronRelDict = polyhedra_to_neurons_bidict
        return list_of_polyhedra, list_of_linear_functions
    
    def generate_polyhedron_neuron_relationship_dict(self, dimensionality: int, **kwargs):
        # already generated during generation of polyhedra
        #return self.polyhedronNeuronRelDict
        pass

    def get_all_possible_activation_states(self, neuron_index):
        # we know from our construction of the polyhedron decomposition, that each permutation
        # that includes the specified neuron is a valid activation state in exactly one polyhedron
        # we can therefore simply return all these polyhedra and their corresponding activation states without 
        # having to check for overlaps
        # there should be group_size! such polyhedra
        polyhedron_indices = self.polyhedronNeuronRelDict.get_polyhedra_for_neuron(neuron_index)
        all_possible_activation_states = [(self.list_of_linear_functions[poly_id][0][neuron_index, :,0].reshape(1, -1),
                                           self.list_of_linear_functions[poly_id][1][neuron_index, 0]) for poly_id in polyhedron_indices]
        return all_possible_activation_states, [[poly_id] for poly_id in polyhedron_indices]
    
    def compute_min_max_act_over_polyhedron_index(self, polyhedron_index: int, neuron_index:int):
        # are infintiy for all neurons, as the GroupSort function is unbounded
        return -np.inf, np.inf


    def get_polyhedron_decomposition(self, neuron_index: int):
        """
        Get a polyhedron deomposition of the pre-activation space such that the specified neuron is fixed linear for each 
        of these polyhedra.

        Parameters
        ----------
        neuron_index : int
            Index of the neuron for which the polyhedron decomposition should be computed.
        Returns
        -------
        polyhedron_dict: dict
            Dictionary of the polyhedron indices that make up the decomposition and a list of neuron indices whose neurons are determined
            for each polyhedron.
        activation_state_dict: dict[tuple]
            Dictionary containing for each polyhedron the activation state (Lambda, lambda)
        """

        polyhedron_indices = self.polyhedronNeuronRelDict.get_polyhedra_for_neuron(neuron_index)
        polyhedron_dict = {poly_id: self.polyhedronNeuronRelDict.get_neurons_for_polyhedron(poly_id) for poly_id in polyhedron_indices}
        activation_state_dict = {poly_id: self.list_of_linear_functions[poly_id] for poly_id in polyhedron_indices}
        return polyhedron_dict, activation_state_dict
        



        
        



            

    

