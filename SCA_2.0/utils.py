# utils
import numpy as np
from random import gauss

def SumDirection(vectors):
    '''
    Calculate sum direction from set of vectors
    '''
    direction = vectors[:, 1].sum(axis=0)
    #np.linalg.norm() is a magnitude of the vector
    unit =(direction/(np.linalg.norm(direction)))
    return unit

def RotVect(v, angle):
    '''
    Rotate vector v. Angle input in grades
    '''
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    RM = np.array([[c, -s], [s, c]])
    rot = np.dot(RM, v)
    return rot

def Segm(parent, vect, l, thick=None):
    ''' 
    Creates line segment from position coordinates,
    unit vector direction and chosen lenght
    '''

    if np.linalg.norm(vect) != 1:
        vect = vect/(np.linalg.norm(vect))
        
    x, y = parent
    dx, dy = l*vect
    return np.array([x+dx, y+dy])

def square_dist(p1,p2):
    '''
    rejecting np.linalg, just because.
    '''
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def make_rand_di(dims):
    '''
    Creates random direction for unit vector 
    in space with dimentions = dims
    '''
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .25
    return np.array([x/mag for x in vec])
