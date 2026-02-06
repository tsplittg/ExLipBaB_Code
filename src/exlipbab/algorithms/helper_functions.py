import numpy as np

def linprop(old_tilde_W, old_tilde_b, network_weigths: list[tuple], network_Lambda: list, network_lambda: list, start_layer, end_layer, 
            in_branch=False):
    """
    Computes the total linear propagation from start_layer to end_layer as long as the activation patterns A are fixed.

    Parameters
    ----------
    old_tilde_W : np.array
        Propagated weight matrix up to the start_layer..
    old_tilde_b : np.array
        Propagated bias vector up to the start_layer.
    network_weigths : list[tuple]
        List of tuples (W,b) of weight matrices and bias vectors of the neural network.
    network_activations : list
        List of matrices representing the activation pattern for each layer.
    start_layer : int
        Index of the layer from which the propagation should start. Starting at 0 (indicating the first layer).
    end_layer : int
        Index of the layer at which the propagation should end, starting at 0.
    in_branch : bool
        If True, the activation pattern of the end_layer-1 is not applied. Used in the branching filter update.

    Returns
    -------
    tilde_W : np.array
        Propagated weight matrix up to the end_layer.
    tilde_b : np.array
        Propagated bias vector up to the end_layer.
    """
    
    if start_layer == end_layer:
        return old_tilde_W, old_tilde_b
    else:
        # check that the interval matrix is actually concrete for the current layer
        assert np.all(network_Lambda[start_layer][:,:,0] == network_Lambda[start_layer][:,:,1]), "linprop only works for fixed linear neurons"
        assert np.all(network_lambda[start_layer][:,0] == network_lambda[start_layer][:,1]), "linprop only works for fixed linear neurons"
        A = network_Lambda[start_layer][:,:,0]
        #ToDo implement linprop for activation functions with bias, if these ever occur
        b = network_lambda[start_layer][:,0]
        # ToDo: computation probably not correct if activation function has a bias (if it ever does)
        #check that the dimensions are compatible
        #assert network_weigths[start_layer][0].shape[1] == A.shape[0], f"Weight matrix and activation pattern dimensions do not match: {network_weigths[start_layer][0].shape} and {A.shape}"
        #assert A.shape[1] == old_tilde_W.shape[0], f"Activation pattern and old_tilde_W dimensions do not match: {A.shape} and {old_tilde_W.shape}"
        if in_branch:
            one_propagated_tilde_W = network_weigths[start_layer+1][0] @ A @ old_tilde_W
            one_propagated_tilde_b = network_weigths[start_layer+1][0] @ A @ old_tilde_b + network_weigths[start_layer+1][1]
        elif not in_branch:
            one_propagated_tilde_W = A @ network_weigths[start_layer][0] @ old_tilde_W
            one_propagated_tilde_b = A @network_weigths[start_layer][0] @ old_tilde_b + network_weigths[start_layer][1]
        else:
            one_propagated_tilde_W = old_tilde_W
            one_propagated_tilde_b = old_tilde_b
        return linprop(one_propagated_tilde_W, one_propagated_tilde_b, network_weigths, network_Lambda, network_lambda, start_layer + 1, end_layer, in_branch=in_branch)