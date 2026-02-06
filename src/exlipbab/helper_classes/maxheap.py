import heapq
import copy
from exlipbab.helper_classes.sub_problem import SubProblem

class MaxHeap:
    """
    In older Python versions, the `heapq` module only provides a min-heap implementation.
    We could simply invert the values to simulate a max-heap but to reduce potential cources of errors,
    we implement a simple adapter class.
    """

    def __init__(self, iterable):
        self.heap = copy.deepcopy(iterable)
        self.heap = [MaxHeapElement(list_element) for list_element in self.heap]
        heapq.heapify(self.heap)
    
    def push(self, subproblem: SubProblem):
        """
        Pushes a new sub-problem onto the max-heap.

        Parameters
        ----------
        sub_problem : SubProblem
            The sub-problem to be added to the heap.
        """
        
        heapq.heappush(self.heap, MaxHeapElement(subproblem))
    
    def pop(self) -> SubProblem:
        """
        Pops and returns the sub-problem with the highest upper bound from the max-heap.

        Returns
        -------
        SubProblem
            The sub-problem with the highest upper bound.
        """
        return heapq.heappop(self.heap).sub_problem
    
    def len(self) -> int:
        """
        Returns the number of sub-problems in the max-heap.

        Returns
        -------
        int
            The number of sub-problems in the heap.
        """
        return len(self.heap)
    
    def __getitem__(self, index):
        return self.heap[index].sub_problem


class MaxHeapElement:
    """
    Wrapper class to invert the comparison operators for max-heap behavior.
    """

    def __init__(self, sub_problem: SubProblem):
        self.sub_problem = sub_problem

    def __lt__(self, other):
        return self.sub_problem > other.sub_problem

    def __eq__(self, other):
        return self.sub_problem == other.sub_problem

         