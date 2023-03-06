import numpy as np
from numpy import array, arange, zeros, ones, sin, cos, pi
from numpy import linalg
from importlib import reload
import sys
import time
from pathlib import Path
import itertools as itt
import scipy as sp
from scipy import ndimage as ndi
from scipy import stats
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure, gcf, gca, plot, close, xlim, ylim, xlabel, ylabel, title,\
                              subplots
from tqdm.auto import tqdm
import pandas as pd

class TreeNode:
    max_branches=5 # safety switch to prevent infinite branching
    def __init__(self, v, rad=0.005, parent=None, tree=None):
        self.parent = parent
        self.radius = rad
        self.tree = set() if tree is None else tree
        self.tree.add(self)
        if parent is not None and not self in parent.children:
            parent.children.append(self)
            
        self.children = []
        self.v = np.array(v) # spatial coordinate of the node
    
    def spawn(self, S : "attractor set", Dg : "growth distance" = 0.025, eps=0.00001, jitter=0.01, verbose=False):
        
        if not len(S):
            return
        S = np.array(S)
        d = (S - self.v)
        
        n = np.sum(d/(1e-6 + linalg.norm(d, axis=1)[:,np.newaxis]), axis=0)
                
        nnorm = np.linalg.norm(n)            
        
        n = n / (1e-6 + nnorm)
            
        vdash = self.v + Dg*n
                
        if len(self.children) < self.max_branches:
            tip = TreeNode(vdash, parent=self, tree=self.tree)

def space_colonization(tree, sources, iterations, Dg, Di, Dk):

    for j in tqdm(range(iterations)):
        
        tree_prev = [n for n in tree if len(n.children) <= n.max_branches]
        
        kdt = sp.spatial.KDTree([n.v for n in tree_prev])
        
        d,inds = kdt.query(sources, distance_upper_bound=Di)
            
        if len(d) and np.min(d) > Di:
            d,inds = kdt.query(sources, distance_upper_bound=np.min(d))
            
        for i, n in enumerate(tree_prev):
            S = sources[inds==i]
            n.spawn(S, Dg)
                    
        kdt2 = sp.spatial.KDTree([n.v for n in tree])
        too_close = kdt2.query_ball_point(sources, Dk, return_length=True)        
        sources = sources[too_close == 0] 
        
        if not len(sources):
            break
        
        # add small jitter to break up ties
        sources  = sources + np.random.randn(*sources.shape)*Dg*0.05
    
    return tree, sources

# def set_thickness(tree):
#     pass

def plot_tree(tree, root=None, sources=None, ax=None, view_init=None,
               grid = True, show_sources =True, show_leaflets=True, params = None):
    
    if ax is None:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(projection='3d')

    if root is not None:
        ax.plot(root.v[0], root.v[1], root.v[2], 'ro')
    for n in tree:
        v = n.v
        for ch in n.children:
            vx = np.vstack([v, ch.v])
            plot(vx[:,0], vx[:,1], vx[:, 2], 'k-', lw=1, color = 'navy', alpha=0.7)
    if sources is not None and show_sources == True:
        ax.plot(sources[:,0], sources[:,1], sources[:, 2], '.', color='salmon', ms=0.5)
    ax.axis('equal')

    if view_init is not None:
        elev, azim = view_init
        ax.view_init(elev=elev, azim=azim)
    if grid == False:
        ax.grid(False)
        ax.axis('off')
    if show_leaflets == True:
        terminals = np.array([n.v for n in tree if len(n.children) == 0])
        ax.plot(terminals[:, 0], terminals[:, 1], terminals[:, 2], 'o', color = 'maroon', ms=1.5)
    if params is not None:
        ax.legend

def tree_to_graph(tree, optimize_graph=False):
    def get_node(node):
        '''
        represent node as tuple
        '''
        v = node.v
        return tuple(v)
    root = [i for i in tree if i.parent == None][0]
    graph = nx.DiGraph()
    segms = [(get_node(node.parent), get_node(node)) for node in tree if node != root]
    children = root.children
    root_connect = [(get_node(root), get_node(child)) for child in children]
    graph.add_node(get_node(root), root = get_node(root))
    graph.add_nodes_from([get_node(node) for node in tree if node != root], root = get_node(root))
    graph.add_edges_from(segms, root = get_node(root))
    graph.add_edges_from(root_connect, root = get_node(root))

    return graph

class Synapse:
    def __init__(self, coords):
        self.age = 0
        self.coords = coords
        self.nodes = None
        self.alive = True
        self.mature = False
    
    def age_flag(self):
        ms = self.age + 0.5
        alpha = 1/(self.age+1)
        color = 'r'
        if self.age > 5:
            ms = 6
            alpha = 0.25
            color = 'maroon'

        kwards = {'ms': ms,
                  'color': color,
                  'alpha': alpha
        }
        return kwards

def generate_random_synapse(radius):
    coords = radius*(np.random.rand(1,3) - 0.5)[0]
    synapse = Synapse(coords)
    return synapse

def generate_synapse(tree, radius, Dk, Dg):
    coords = radius*(np.random.rand(1,3) - 0.5)[0]
    synapse = Synapse(coords)
    kdt = sp.spatial.KDTree([n.v for n in tree])
    d,inds = kdt.query(synapse.coords, distance_upper_bound=Dg*10)
    if d < Dg*10 and d > Dk:
        return synapse
    else:
        return generate_synapse(tree, radius, Dk, Dg)

def update_synapses(tree, synapses, Dk, Dg, condition=None, radius=2, life_cycle=45):
    new_synapses = list(synapses)
    # p = eprob
    # condition is a function
    if condition == None: 
        def default_condition():
            return np.random.random() < 0.8
        condition = default_condition()

    for s in new_synapses:
        if s.age > life_cycle:
            s.alive = False
        s.age += 1

    # while p <1:
    while condition == True:
        synapse = generate_synapse(tree, radius, Dk, Dg)
        if len(synapse.coords) == 0:
            continue
        new_synapses.append(synapse)
        # p += np.random.uniform(0,0.5)
        #reset condition
        condition=default_condition()
    
    new_synapses = list(filter(lambda x: x.alive == True, synapses))
    new_synapses = list(filter(lambda x: x.coords.shape != np.array([]).shape, synapses))
    return np.array(new_synapses)

def space_colonization_synupd(tree, sources, iterations, Dg, Di, Dk, upd_cycle = 50):
   
    for j in tqdm(range(iterations)):

        if j > upd_cycle:
            sources = update_synapses(tree, sources, Dk, Dg, eprob=-1)
        
        tree_prev = [n for n in tree if len(n.children) <= n.max_branches]
        
        kdt = sp.spatial.KDTree([n.v for n in tree_prev])
        
        sources_coords = [s.coords for s in sources]

        d,inds = kdt.query(sources_coords, distance_upper_bound=Di)
            
        if len(d) and np.min(d) > Di:
            d,inds = kdt.query(sources_coords, distance_upper_bound=np.min(d))
            
        for i, n in enumerate(tree_prev):
            S = sources[inds==i]
            S = [s.coords for s in S]
            n.spawn(S, Dg)
                    
        kdt2 = sp.spatial.KDTree([n.v for n in tree])
        too_close = kdt2.query_ball_point(sources_coords, Dk, return_length=True)        
        sources = sources[too_close == 0] 
        
        if not len(sources):
            break
        
        # add small jitter to break up ties
        for s in sources:
            s.coords = s.coords + np.random.randn(*s.coords.shape)*Dg*0.05

    return tree, sources

class Space_colonization:
    def __init__(self, root, sources):
        self.tree = set()
        self.root = TreeNode(root, tree=self.tree)
        self.sources = sources
        self.iterations = 1000
        self.Di = 1
        self.Dg = 0.02
        self.Dk = 0.04
        if len(self.tree) == 0:
            self.remaining_sources = sources

    
    def set_parameters(self, iters, Di, Dg, Dk):
        """
        Set growth parameters:

        iters: # iterations
        Di: Influence Distance
        Dg: Segment Distance
        Dk: Kill Distance
        """
        self.iterations = iters
        self.Di = Di
        self.Dg = Dg
        self.Dk = Dk

    def go_tree_classic(self):
        start_time = time.perf_counter()
        new_tree, remaining_sources = space_colonization(self.tree, self.sources, 
                                                 iterations=self.iterations,
                                                 Di=self.Di,
                                                 Dg=self.Dg,
                                                 Dk=self.Dk,
                                                )
        stop_time = time.perf_counter()
        # time in seconds
        self.time = round(stop_time - start_time)
        self.tree = new_tree
        self.remaining_sources = remaining_sources
    
    def go_tree_synupd(self, upd_cycle=50):
        self.upd_cycle = upd_cycle
        sources_syn = np.array([Synapse(s) for s in self.sources])
        for s in sources_syn:
            s.age = -(self.upd_cycle)
        self.sources = sources_syn

        start_time = time.perf_counter()

        new_tree, remaining_sources = space_colonization_synupd(self.tree, self.sources, 
                                                 iterations=self.iterations,
                                                 Di=self.Di,
                                                 Dg=self.Dg,
                                                 Dk=self.Dk,
                                                 upd_cycle=self.upd_cycle
                                                )
        stop_time = time.perf_counter()
        # time in seconds
        self.time = round(stop_time - start_time)

        self.tree = new_tree
        self.remaining_sources = remaining_sources

    def make_graph(self):
        if self.tree:
            return tree_to_graph(tree=self.tree, root=self.root)
        
    def set_thickness(self, sigma):
        tips = [n for n in self.tree if n.children == 0]
        for n in self.tree:

def content(SCA):
    return SCA.__dict__
