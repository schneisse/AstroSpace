# features
import numpy as np

def SumDirection(vectors):
    direction = vectors[:, 1].sum(axis=0)
    #np.linalg.norm() is a magnitude of the vector
    unit =(direction/(np.linalg.norm(direction)))
    return unit

def RotVect(v, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    RM = np.array([[c, -s], [s, c]])
    rot = np.dot(RM, v)
    return rot

def Segm(parent, vect, l, thick=None):
    
    if np.linalg.norm(vect) != 1:
        vect = vect/(np.linalg.norm(vect))
        
    x, y = parent
    dx, dy = l*vect
    return np.array([x+dx, y+dy])

def square_dist(p1,p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2