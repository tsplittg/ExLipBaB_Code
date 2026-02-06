import torch.nn as nn
import numpy as np

class GroupSort(nn.Module):
    """
    Implementation of the GroupSort activation proposed in Anil et al. (2019); coded from scratch, roughly following the original code at https://github.com/cemanil/LNets/blob/master/lnets/models/activations/group_sort.py
    """

    def __init__(self, num_groups, axis = -1):
        super(GroupSort, self).__init__()
        self.num_groups = num_groups
        self.axis = axis

    def forward(self, x):
        grouped_shape, sort_dim = self._compute_grouped_shape(x)

        grouped_tensor = x.view(grouped_shape)
        grouped_tensor, _ = grouped_tensor.sort(dim=sort_dim)

        sorted_tensor = grouped_tensor.view(x.shape)
        return sorted_tensor
    
    def extra_repr(self):
        return 'num_groups={}'.format(self.num_groups)
    
    def _compute_grouped_shape(self, x):
        
        shape_list = list(x.shape)
        axis = len(shape_list)-1 if self.axis == -1 else self.axis

        if shape_list[axis] % self.num_groups != 0:
            raise ValueError("Number of elements on given axis not divisible by number of groups")
        
        sort_dim = axis+1
        shape_list.insert(sort_dim, shape_list[axis] // self.num_groups)
        shape_list[axis] = self.num_groups
        return shape_list, sort_dim


MaxMin = GroupSort(2)