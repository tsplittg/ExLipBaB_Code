from __future__ import annotations

import numpy as np
from numpy._core.multiarray import array
import pypoman
import polytope as pc
from cvxopt import solvers, matrix



class Polyhedron:

    @classmethod
    def from_intervals(cls, intervals: list[tuple]) -> Polyhedron:
        """
        Creates a polyhedron from a list of (possibly infinite) intervals.
        
        Parameters
        ----------
        intervals : list of tuples
            List of tuples representing the intervals for each dimension. Each tuple is of the form (lower_bound, upper_bound),
            where lower_bound and upper_bound can be finite numbers or -np.inf/np.inf for unbounded intervals.
        
        Returns
        -------
        polyhedron : Polyhedron
            The polyhedron representing the intersection of the intervals.     
        """
        A_rows = []
        b_rows = []
        num_dimensions = len(intervals)
        for i, (lower, upper) in enumerate(intervals):
            if lower != -np.inf:
                # most dimensions are not constrained in this specific half space
                left_constraint_row = np.zeros(num_dimensions)
                #constrain the i-th dimension, x_i >= lower  <=> -x_i <= -lower
                left_constraint_row[i] = -1
                A_rows.append(left_constraint_row)
                b_rows.append(-lower)
            if upper != np.inf:
                # most dimensions are again not constrained in this specific half space
                right_constraint_row = np.zeros(num_dimensions)
                #constrain the i-th dimension, x_i <= upper:
                right_constraint_row[i] = 1
                A_rows.append(right_constraint_row)
                b_rows.append(upper)
        A = np.vstack(A_rows)
        b = np.array(b_rows)
        return cls(A, b)
    
    def extend_by_new_vars(self, new_vars: list[tuple]) -> Polyhedron:
        """
        Extends the polyhedron by adding new variables with given intervals. Used in the symprop algorithm.

        Parameters
        ----------
        new_vars : list of tuples
            List of tuples representing the intervals for each new variable. Each tuple is of the form (lower_bound, upper_bound),
            where lower_bound and upper_bound can be finite numbers or -np.inf/np.inf for unbounded intervals.
        Returns
        -------
        extended_polyhedron : Polyhedron
            The polyhedron extended by the new variables.
        """
        # check if any constraint is redundant, i.e. if for any new variable the interval is (-inf, inf) or (inf, -inf)
        additional_dimensions = len(new_vars)
        new_vars = [var for var in new_vars if not (var[1] == -np.inf and var[0] == np.inf)]
        new_vars = [var for var in new_vars if not (var[1] == np.inf and var[0] == -np.inf)]
        number_redundant_vars = additional_dimensions - len(new_vars)

        if len(new_vars) == 0:
            # we have to adjust the dimensionality of the polyhedron anyway, but add no new constraints
            if type(self) == FullPolyhedron or self.A.shape[0] == 0:
                return FullPolyhedron(self.dimension + additional_dimensions)
            else:
                new_A = np.hstack((self.A, np.zeros((self.A.shape[0], len(new_vars)))))
                return Polyhedron(new_A, self.b)
        

        new_A, new_b = Polyhedron.from_intervals(new_vars).A, Polyhedron.from_intervals(new_vars).b
        # we first append non-redundant new variables

        # we first extend the existing A matrix by adding zero columns for the new variables, because these do not appear in the existing constraints
        extended_A = np.hstack((self.A, np.zeros((self.A.shape[0], new_A.shape[1]))))
        
        # similarly, we extend the new A matrix by adding zero columns for the existing variables
        extended_new_A = np.hstack((np.zeros((new_A.shape[0], self.dimension)), new_A))

        # now we can combine the two A matrices and b vectors
        combined_A = np.vstack((extended_A, extended_new_A))
        combined_b = np.hstack((self.b, new_b))

        # now we add additional zero columns for the redundant variables
        if number_redundant_vars > 0:
            combined_A = np.hstack((combined_A, np.zeros((combined_A.shape[0], number_redundant_vars))))

        return Polyhedron(combined_A, combined_b)
    
    @classmethod
    def find_surrounding_polyhedron(cls, region: list[Polyhedron]) -> Polyhedron:
        """
        Finds a polyhedron that surrounds a list of polyhedra.

        Parameters
        ----------
        region : list of Polyhedron
            List of polyhedra to find a surrounding polyhedron for.

        Returns
        -------
        surrounding_polyhedron : Polyhedron
            A polyhedron that surrounds all the given polyhedra.
        """

        polytope_region = [pc.Polytope(p.A, p.b) for p in region]
        surrounding_polytope_pc = pc.envelope(polytope_region)
        return cls(surrounding_polytope_pc.A, surrounding_polytope_pc.b)




    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Initializes a polyhedron in half space representation defined by Ax <= b, where b is an
        n-dimensional vector and A an (n x m) matrix
        """
        # set attributes
        self.A = A
        self.b = b
        self.dimension = A.shape[1]

    def intersect(self, other_polyhedron: Polyhedron) -> Polyhedron:
        """
        Computes the alf space constraints of the intersection of two polyhedra.

        Parameters
        ----------
        other_polyhedron : Polyhedron
            The other polyhedron to compute the intersection with.
        Returns
        -------
        intersection_polyhedron : Polyhedron
            The polyhedron representing the intersection of the two polyhedra.
        """

        # check that the dimensions of the polyhedra are compatible:
        if self.dimension != other_polyhedron.dimension:
            raise ValueError("Polyhedra must have the same dimension to check intersection.")
        
        # combine the inequalities of both polyhedra
        combined_A = np.vstack((self.A, other_polyhedron.A))
        combined_b = np.hstack((self.b, other_polyhedron.b))
        
        return Polyhedron(combined_A, combined_b)
    
    def is_empty(self) -> bool:
        """
        Checks whether the polyhedron is empty.

        Returns
        -------
        is_empty : bool
            True if the polyhedron is empty, False otherwise.
        """
        # see if there is at least one point in \mathbbbb{R}^m that satisfies Ax <= b.
        c = np.zeros(self.dimension)
        solution = solvers.lp(matrix(c), matrix(self.A), matrix(self.b), solver='glpk')
        match solution['status']:
            case 'optimal':
                return False
            case 'primal infeasible':
                return True
            case _:
                return True
                raise ValueError("Unexpected result from LP solver when checking polyhedron emptiness. Got status: " + solution['status'])
            
    def apply_linearfunction(self, W: np.ndarray, c: np.ndarray) -> Polyhedron:
        """
        Applies a linear function f(x) = Wx + c to the polyhedron.

        Parameters
        ----------
        W : np.ndarray
            Weight matrix of the linear function.
        c : np.ndarray
            Bias vector of the linear function.

        Returns
        -------
        transformed_polyhedron : Polyhedron
            The polyhedron after applying the linear function.
        """
        #check that the dimensions are compatible
        if W.shape[1] != self.dimension:
            if W.shape[0] == self.dimension:
                #print("Warning: Weight matrix W seems to be transposed. Trying to transpose it automatically.")
                W = W.T
            else:
                raise ValueError(f"The weight matrix W must have shape (k x m) where m is the polyhedron dimension. Got shape {W.shape} and dimension {self.dimension} instead.")
        
        # if W is invertible, we can directly compute the new half space representation
        if np.linalg.matrix_rank(W) == W.shape[0] and W.shape[0] == W.shape[1]:
            W_inv = np.linalg.inv(W)
            new_A = self.A @ W_inv
            new_b = self.b - (self.A @ W_inv) @ c
            return Polyhedron(new_A, new_b)
        else:
            # if W is not invertible, the process becomes more complex. We choose to then first transform the
            # polyhedron to vertex representation, apply the linear function to the vertices and then convert back
            # to half space representation.
            transformed_polyhedron_V = self.apply_linearfunction_to_V(W, c)
            #print("Transformed vertices:", transformed_polyhedron_V)
            new_A, new_b = pypoman.duality.compute_polytope_halfspaces(transformed_polyhedron_V)
            return Polyhedron(new_A, new_b)
    
    def apply_linearfunction_to_V(self, W: np.ndarray, c: np.ndarray) -> list:
        """ Applies a linear function f(x) = Wx + c to the polyhedron returning a list of vertices.
        
        Parameters
        ----------
        W : np.ndarray
            Weight matrix of the linear function.
        c : np.ndarray
            Bias vector of the linear function.
        Returns
        -------
        transformed_vertices : list
            List of vertices after applying the affine function.
        """
        #check that the dimensions are compatible
        #if W.shape[1] != self.dimension:
        #    raise ValueError(f"The weight matrix W must have shape (k x m) where m is the polyhedron dimension. Got shape {W.shape} and dimension {self.dimension} instead.")
        transformed_polyhedron_V, rays = pypoman.projection.project_polyhedron(proj= (W, c), ineq = (self.A, self.b))
        #print("Rays after transformation:", rays)
        return transformed_polyhedron_V


    def apply_interval_linearfunction(self, W_interval: np.ndarray, c_interval: np.ndarray) -> Polyhedron:
        """
        Applies an interval linear function f(x) = W_interval * x + c_interval to the polyhedron.

        Parameters
        ----------
        W_interval : np.ndarray
            Interval weight matrix of the linear function. Shape (k x m x 2), where k is the output dimension,
            m is the input dimension, and the last dimension represents the lower and upper bounds of the intervals.
        c_interval : np.ndarray
            Interval bias vector of the linear function. Shape (k x 2), where k is the output dimension,
            and the last dimension represents the lower and upper bounds of the intervals.

        Returns
        -------
        transformed_polyhedra : list[Polyhedron]
            List of all possible polyhedra after applying the interval linear function.
        """
        list_of_possible_linear_fctns = []
        # get the above list from all combinations of lower and upper bounds of the intervals

        
    def find_surrounding_box(self) -> list[tuple]:
        """
        Finds a box (hyperrectangle) that surrounds the polyhedron.
        Returns
        -------
        box : list of tuples
            List of tuples representing the intervals for each dimension of the box.
        """
        pass
    
    def __repr__(self) -> str:
        return f"Polyhedron(A={self.A}, b={self.b})"
    
    def compute_vertices(self) -> list[np.array]:
        """
        Computes the vertices of the polyhedron.

        Returns
        -------
        vertices : list of np.ndarray
            List of vertices of the polyhedron.
        """

        vertices = pypoman.duality.compute_polytope_vertices(self.A, self.b)
        return vertices
    
class FullPolyhedron(Polyhedron):
    """
    A polyhedron that fills the entire space R^n.
    """
    def __init__(self, dimension: int):
        A = np.zeros((0, dimension))
        b = np.zeros((0,))
        super().__init__(A, b)
    
    def is_empty(self) -> bool:
        return False
    
    def intersect(self, other_polyhedron):
        return other_polyhedron
    
    def compute_vertices(self) -> list:
        return []

