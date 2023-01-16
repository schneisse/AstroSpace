# create point cloud
import numpy as np
from random import random

def inner_points(R):
    while True:
        x = np.random.random()*2 - 1
        y = np.random.random()*2 - 1
        if x*x + y*y < R:
            return x, y
class Tree:
    
    def __init__(self, value, R):
        leaves = []
        for i in range(value):
            x = inner_points(R)
            leaves.append(x)
        
        self.value = value
        self.leaves = np.array(leaves)
        self.radius = R
        self.center = np.array([0, 0])
        