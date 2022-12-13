# branch
import numpy as np
from scipy.spatial import KDTree

from features import *
from parameters import R, Ar, Dk, Bl, N, leaves, tree

center = leaves.center

class Branch():
    def __init__(self, pos, di, l, thick=None):
        self.pos = pos
        self.direction = di
        self.thickness = thick
        self.length = l
        self.segment = Segm(pos, di, l)
        
        def plot_branch(self):
            pass

def kill_attr(leaves, br):
    """ Removes the points that have reached Dk
    """
    tree = KDTree(leaves)
    mask = np.array(leaves, dtype=bool)
    for i in sorted(tree.query_ball_point(br.segment, Dk), reverse=True):
        mask[i] = False

    new_leaves = np.array(leaves[mask]).reshape(int(len(leaves[mask])/2), 2)
    return new_leaves

def Sum2di(di1, di2):
    """ Returns sum of two direstions as unit vector 
    """
    new_di = np.array([di1, di2]).sum(axis=0)
    u_di = (new_di/(np.linalg.norm(new_di)))
    return u_di

def Grow(leaves, start_br, Ar, Dk, Bl):
    """ Returns next branch referred to attraction points
    Compute next direction as sum of a branch direction and 
    direction toward nearest attraction point in Ar
    """
    tree = KDTree(leaves)
    dist, num = tree.query(start_br.segment, k=1)
    new_segm = Segm(start_br.segment, 
                start_br.direction, 
                start_br.length)
    
    if dist < Ar:
    
        # new_direction = np.array([new_segm, lvs.leaves[num]]).sum(axis=0)
        # new_direction =(new_direction/(np.linalg.norm(new_direction)))
        new_direction = Sum2di(new_segm, lvs.leaves[num])
        new_br = Branch(start_br.segment, new_direction, Bl)
        
        
    else:
        new_br = Branch(start_br.segment, 
                    start_br.direction, 
                    start_br.length)
    
    return new_br

def Grow2(leaves, start_br, Ar, Dk, Bl):
    """ Returns next branch referred to attraction points
    Compute next direction as sum of a branch direction and 
    average of the normalized vectors toward all the 
    attraction points in Ar
    """
    tree = KDTree(leaves)
    new_segm = Segm(start_br.segment, 
            start_br.direction, 
            start_br.length)
    
    di_neighbors = []
    for nbr in tree.query_ball_point(start_br.segment, Ar):
        di = leaves[nbr]
        di_neighbors.append(di)
    
    if di_neighbors:
        new_direction = sum(map(lambda x: Sum2di(new_segm, x), di_neighbors))/len(di_neighbors)
        n = new_direction/np.linalg.norm(new_direction)
        new_br = Branch(start_br.segment, n, Bl)
    else:
        new_br = Branch(start_br.segment, start_br.direction, Bl)
        
    return new_br

def match(collection, br):
    for i in collection:
        if i.segment[0] == br.segment[0] and i.segment[1] == br.segment[1]:
            return True
        else:
            return False

def borders_reached(br):
    if square_dist(center, br.segment)< R:
        return False
    else:
        return True
    
def prob(val, collection, v=1):
    if  v == 1:
        num = len(collection)
        return np.random.random()-np.exp(-val/(num/3)) > 0.5
    elif v == 2:
        segm = collection[val].segment
        length = collection[val].length
        num = len(tree.query_ball_point(segm, Ar/length))
        return num>2
    else:
        raise ValueError('Version {} do not exist. Try 1 or 2'.format(v))
 
        
    