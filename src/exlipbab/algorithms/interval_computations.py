# Helper functions for interval-wise computations in ExLipBaB algorithm
import numpy as np

def interval_addition(a: tuple, b: tuple) -> tuple:
    """
    Addition for two intervals a and b.

    Parameters
    ----------
    a : tuple
        Interval represented as (lower_bound, upper_bound).
    b : tuple
        Interval represented as (lower_bound, upper_bound).

    Returns
    -------
    result : tuple
        Resulting interval after addition.
    """
    lower_bound = a[0] + b[0]
    upper_bound = a[1] + b[1]
    return (lower_bound, upper_bound)

def interval_multiplication(a: tuple, b: tuple) -> tuple:
    """
    Multiplication for two intervals a and b.

    Parameters
    ----------
    a : tuple
        Interval represented as (lower_bound, upper_bound).
    b : tuple
        Interval represented as (lower_bound, upper_bound).
    Returns
    -------
    result : tuple
        Resulting interval after multiplication.
    """
    possible_products = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
    lower_bound = min(possible_products)
    upper_bound = max(possible_products)
    return (lower_bound, upper_bound)

def interval_matrix_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication for two interval matrices A and B.

    Parameters
    ----------
    A : np.ndarray
        Interval matrix represented as a numpy array of dimension (m x k x 2).
    B : np.ndarray
        Interval matrix represented as a numpy array of dimension (k x n x 2).

    Returns
    -------
    result : np.ndarray
        Resulting interval matrix after multiplication numpy array of dimension (m x n x 2).
    """

    m, k, _ = A.shape
    k2, n, _ = B.shape
    if k != k2:
        raise ValueError(f"Dimensions of matrices do not match, got dimensions {A.shape} and {B.shape}")
    
    result_matrix = np.zeros((m,n,2))
    for i in range(m):
        for j in range(n):
            initial_interval = np.array((0,0))
            for k_tilde in range(k):
                initial_interval = interval_addition(interval_multiplication(A[i, k_tilde, :], B[k_tilde, j, :]), initial_interval)
            result_matrix[i,j] = initial_interval

    return result_matrix

    