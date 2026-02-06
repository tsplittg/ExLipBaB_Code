from __future__ import annotations
from exlipbab.helper_classes.piece_wise_linear_function import PiecewiseLinearFunction
from exlipbab.algorithms.helper_functions import linprop
import numpy as np

class SubProblem:
    """
    Class from which we will initialize the new sub-problems we will create during
    a run of the ExLipBab algorithm.
    """
    
    def __init__(self, polyhedron, activation_Lambda, activation_lambda):
        self.polyhedron = polyhedron    # domain constraints of the sub-problem
        self.activation_Lambda = activation_Lambda    # list of interval matrices representing the activation pattern for each layer    
        self.activation_lambda = activation_lambda    # list of interval vectors representing the bias activation pattern for each layer
        self.tilde_l = None  # index of first layer with not fixed linear neuron
        self.star_neurons = None  # list of star neurons in each layer
        self.upper_bound = float("inf")  # upper bound of the Lipschitz constant for this sub-problem
        self.propagated_weights = None  # tuple of propagated weights and biases up to tilde_l
        pass

    def __lt__(self, other_sub_problem: SubProblem) -> bool:
        """
        Less than operator for comparing two sub-problems based on their upper bounds.

        Parameters
        ----------
        other_sub_problem : SubProblem
            Another sub-problem to compare with.
        Returns
        -------
        bool
            True if the upper bound of this sub-problem is less than that of the other sub-problem, False otherwise.
        """
        return self.upper_bound < other_sub_problem.upper_bound
    
    def __eq__(self, other_sub_problem: SubProblem) -> bool:
        """
        Equality operator for comparing two sub-problems based on their upper bounds.

        Parameters
        ----------
        other_sub_problem : SubProblem
            Another sub-problem to compare with.
        Returns
        -------
        bool
            True if the upper bound of this sub-problem is equal to that of the other subproblem, False otherwise.
        """
        return self.upper_bound == other_sub_problem.upper_bound

    
    def ffilter_prop(self, weights: list, alpha: list[PiecewiseLinearFunction], solver):
        """
        Combination of the ffilter function and the LinProp function from the original LipBab paper.
        Propagates the linear relations through the network until a layer with a star neuron is reached and also 
        checks layerwise if any neuron can be fixed linear until a layer with a star neuron is reached.
        The network propagated weights have to be updated before the function is called!

        Parameters
        ----------
        weights : list
            List of tuples (W,b) of weight matrices and bias vectors of the neural network.
        alpha : list[PiecewiseLinearFunction]
            List of piecewise linear (activation) functions of the neural network.
        solver : str
            Solver to be used in linear programming.

        Returns
        -------
        sub_problem :
            Updated sub_problem after ffilter propagation with updated activation pattern $Lambda$ and $lambda$, propagated weights and tilde{l}.
        """
        # main loop: propagate linear relations until a layer with a star neuron is reached
        old_tilde_l = self.tilde_l
        propagated_weights = self.propagated_weights
        for ell in range(self.tilde_l, len(weights)):
            #next_W = weights[ell][0] *alpha[ell] * propagated_weights[0]
            #next_b = weights[ell][0] *alpha[ell] * propagated_weights[1] + weights[ell][1]
            Lambda_ell, lambda_ell, star_neurons_ell = alpha[ell].get_activation_pattern(self.polyhedron, old_star_neurons=self.star_neurons[ell],
                                                                                                old_Lambda=self.activation_Lambda[ell],
                                                                                                old_lambda=self.activation_lambda[ell], propagated_weights=propagated_weights,
                                                                                                solver = solver)

            #update sub_problem attributes:
            self.activation_Lambda[ell] = Lambda_ell
            self.activation_lambda[ell] = lambda_ell
            self.star_neurons[ell] = star_neurons_ell
            
            if(len(star_neurons_ell) > 0):
                self.tilde_l = ell
                break
            elif ell == len(weights)-1:
                self.tilde_l = len(weights)
            else:
                propagated_weights = linprop(old_tilde_W=propagated_weights[0],
                                         old_tilde_b=propagated_weights[1],
                                         network_weigths=weights,
                                         network_Lambda=self.activation_Lambda,
                                         network_lambda=self.activation_lambda,
                                         start_layer=ell,
                                         end_layer=ell + 1, in_branch=True)
        self.propagated_weights = propagated_weights
        return self
