# default parameters
from scipy.spatial import KDTree
import numpy as np
from point_cloud import Tree

# Radius
R = 1
# Attractoin range
Ar = 0.3*R
# Kill distance
Dk = 0.25*R

''' kill distance have to be lower than the attraction range, but should also be greater
than the branch length to avoid weird results'''

# Branch length
Bl = 0.05*R

# Number of points
N = 100

leaves = Tree(N, R)
tree = KDTree(leaves.leaves)