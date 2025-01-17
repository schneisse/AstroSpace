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
from scipy import spatial

import sys
sys.path.insert(1, '/Users/anyak/Documents/Lab/SWC/astroTanya')
sys.path.insert(1, '/Users/anyak/Documents/Lab/neuro.im-proc')

from astro_graph import AstroGraph as AG
from astropy.table import Table

class TreeNode:
    max_branches=5 # safety switch to prevent infinite branching
    def __init__(self, v, rad=0.005, parent=None, tree=None):
        self.label = None
        self.weight = 1
        self.root = None
        self.parent = parent
        self.radius = rad
        self.tree = set() if tree is None else tree
        self.tree.add(self)
        if parent is not None and not self in parent.children:
            parent.children.append(self)

        self.children = []
        self.v = np.array(v) # spatial coordinate of the node
        
        available_labels = {
            'soma': 0,
            'root': 1,
            'bifurcation': 2,
            'tip': 3,
            'trunk': 4,
            'leaflet': 5
        }
        self.node_labels_collection = available_labels
    
    def spawn(self, S : "attractor set", Dg : "growth distance" = 0.025, eps=0.00001, jitter=0.01, verbose=False, label=None):
        
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
            tip.root = self.root
            self.root.weight += 1

            '''
            available labels:

            0. soma
            1. root
            2. bifurcation
            3. tip
            4. trunk
            5. leaflet

            '''
    
            if label is not None:
                tip.label = label

def space_colonization(tree, sources, iterations, Dg, Di, Dk, w_lim=400):
    roots = tree
    for r in roots:
        r.root = r
    
    Dk = Dk * Dg
    Di = Di * Dg
    w_lim = w_lim / Dg 
    
    for j in tqdm(range(iterations)):
        
        tree_prev = [n for n in tree if len(n.children) <= n.max_branches]

        kdt = sp.spatial.KDTree([n.v for n in tree_prev])

        d,inds = kdt.query(sources, distance_upper_bound=Di)
            
        if len(d) and np.min(d) > Di:
            d,inds = kdt.query(sources, distance_upper_bound=np.min(d))
            
        for i, n in enumerate(tree_prev):
            # weight limitation
            if n.root.weight > w_lim:
                continue
            S = sources[inds==i]
            n.spawn(S, Dg)
                    
        kdt2 = sp.spatial.KDTree([n.v for n in tree])
        too_close = kdt2.query_ball_point(sources, Dk, return_length=True)        
        sources = sources[too_close == 0]

        # break if no sources available
        if not len(sources):
            break
        # break if roots reach weight limit
        if all(r.weight > w_lim for r in roots):
            break
        
        # add small jitter to break up ties
        sources  = sources + np.random.randn(*sources.shape)*Dg*0.05
    
    return tree, sources

def sep_cells(tree):
    roots = [i for i in tree if i.root == i]
    
    cells = []
    for root in roots:
        cell = np.array([i for i in tree if i.root == root])
        cells.append(cell)
    return np.array(cells)

def plot_tree(tree, root=None, sources=None, ax=None,
               view_init=None, 
               color_tree = 'navy',
               color_leaflets = 'lightcoral',
               color_sources = 'grey',
               color_soma = 'r',
               cmap='Pastel1',
               size_soma = 5,
               size_tree = 1,
               alpha = 0.7,
               grid = True, 
               show_sources =True, 
               show_leaflets=True, 
               colored_cells=False):
    
    if ax is None:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(projection='3d')

    if root is not None:
        if len([root]) > 1:
            for r in root:
                ax.plot(root.v[0], root.v[1], root.v[2], 'ro', ms=size_soma)
        else:
            ax.plot(root.v[0], root.v[1], root.v[2], 'ro', ms=size_soma)

    
    if colored_cells == True:
        cells = sep_cells(tree)
        colormap = cm.get_cmap(cmap)
        colors = np.linspace(0, 1, len(cells))
        for i, cell in enumerate(cells):
            for n in cell:
                v = n.v
                for ch in n.children:
                    vx = np.vstack([v, ch.v])
                    plot(vx[:,0], vx[:,1], vx[:, 2], 'k-', lw=1, color = colormap(colors[i]), alpha=0.7)

    else:
        for n in tree:
            v = n.v
            for ch in n.children:
                vx = np.vstack([v, ch.v])
                plot(vx[:,0], vx[:,1], vx[:, 2], 'k-', lw=1, color = color_tree, alpha=0.7)

    if sources is not None and show_sources == True:
        ax.plot(sources[:,0], sources[:,1], sources[:, 2], '.', color=color_sources, ms=0.5)
    ax.axis('equal')

    if view_init is not None:
        elev, azim = view_init
        ax.view_init(elev=elev, azim=azim)

    if grid == False:
        ax.grid(False)
        ax.axis('off')

    if show_leaflets == True:
        terminals = np.array([n.v for n in tree if len(n.children) == 0])
        thin_branches = np.array([n for n in tree if n.label == 5])
        for t in thin_branches:
            for ch in t.children:
                vx = np.vstack([t.v, ch.v])
                plot(vx[:,0], vx[:,1], vx[:, 2], 'k-', lw=1, color = color_leaflets, alpha=1)
        ax.plot(terminals[:, 0], terminals[:, 1], terminals[:, 2], 'o', color = color_leaflets, ms=1)


# multiple trees
def tree_to_graph(tree, multiple_cells=False):
    def get_node(node):
        '''
        represent node as tuple
        '''
        v = node.v
        return tuple(v)
    
    if multiple_cells == True:
        roots = [i for i in tree if i.root == i]
        graphs = []
        for root in roots:
            graph = nx.DiGraph()
            cell = [i for i in tree if i.root == root]
            segms = [(get_node(node.parent), get_node(node)) for node in cell if node != root]
            children = root.children
            root_connect = [(get_node(root), get_node(child)) for child in children]
            graph.add_node(get_node(root), root = get_node(root))
            graph.add_nodes_from([get_node(node) for node in cell if node != root], root = get_node(root))
            graph.add_edges_from(segms, root = get_node(root))
            graph.add_edges_from(root_connect, root = get_node(root))
            graphs.append(graph)

        return graphs

    else:
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
    root = [n for n in tree if n.root == n][0]
    h, k, l = root.v
    coords = (np.random.uniform(h-radius, h+radius), np.random.uniform(l-radius, l+radius), np.random.uniform(k-radius, k+radius))
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
            return np.random.random() < 0.3
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

def space_colonization_synupd(tree, sources, iterations, Dg, Di, Dk, w_lim=400, upd_cycle = 50):
    roots = tree
    for r in roots:
        r.root = r
    Di = Di * Dg
    Dk = Dk * Dg
    w_lim = w_lim / Dg
    start_sources = len(sources)
 
    for j in tqdm(range(iterations)):
        if j > upd_cycle & len(sources)<start_sources:
            sources = update_synapses(tree, sources, Dk, Dg, life_cycle=upd_cycle)
        
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
        if all(r.weight > w_lim for r in roots):
            break
        # add small jitter to break up ties
        for s in sources:
            s.coords = s.coords + np.random.randn(*s.coords.shape)*Dg*0.05
    return tree, sources

class Space_colonization:
    def __init__(self, root, sources, tree=set()):
        self.tree = tree
        if len([root]) > 1:
            self.root = [TreeNode(r, tree=self.tree) for r in root]
        else:
            self.root = TreeNode(root, tree=self.tree)
        self.sources = sources
        self.iterations = 1000
        self.Di = 1
        self.Dg = 0.02
        self.Dk = 0.04
        self.w_lim = 400
        if len(self.tree) == 0:
            self.remaining_sources = sources
    
    def set_parameters(self, iters, Di, Dg, Dk, w_lim=400, upd_cycle = 100):
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
        self.w_lim = w_lim
        self.upd_cycle = 100

    def report(self):
        # print initialization info
        pass

    def go_tree_classic(self):
        start_time = time.perf_counter()
        new_tree, remaining_sources = space_colonization(self.tree, self.sources, 
                                                 iterations=self.iterations,
                                                 Di=self.Di,
                                                 Dg=self.Dg,
                                                 Dk=self.Dk,
                                                 w_lim=self.w_lim
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

    def make_graph(self, multiple_cells=False):
        if self.tree:
            return tree_to_graph(tree=self.tree, multiple_cells=multiple_cells)
        

def set_thickness(graph, Dg, sigma = 2, r_min = 0.05):
    attrs = {}
    len_tree = len(graph) - 1
    for n in graph.nodes:
        weight = len(nx.dfs_successors(graph, n))/len_tree * Dg
        rad = sigma * weight**(1/sigma) + r_min
        attrs[n] = {'sigma_mask': rad}
    nx.set_node_attributes(graph, attrs)
    return graph


def swc_save(graph, filename, center, Dg =0.01, cell_type=5, ratio=None, sigmas_rad = False, compartment_type=False):

    root = list(graph.nodes.data('root'))[0][0]
    tips = [n for n in list(graph.nodes) if len(list(graph.successors(n))) == 0]
    max_dist = np.max([spatial.distance.euclidean(root, t) for t in tips])
    radius = 0.05

    astro = AG(graph).swc(center=tuple(center))
    data = Table()
    # ratio = ratio if ratio else self.ratio

    X = []
    Y = []
    Z = []
    POS = []
    PAR = []

    for r in astro:
        for n in r.items(): 
            x, y, z = n[0]
            if (x != None) and (y != None) and (z != None):
                X.append(x)
                Y.append(y)
                Z.append(z)
                if len(n[1]) == 2:
                    pos, par = n[1]
                    POS.append(pos)
                    PAR.append(par)
                elif len(n[1]) == 3:
                    pos, par, s = n[1]
                    POS.append(pos)
                    PAR.append(par)
                    radius = s


    if compartment_type == True:
        ntype = np.array(graph.nodes.data('label'))
    
    else:
        ntype = np.full(len(POS), cell_type)
        ntype[0] = 1
        
    if sigmas_rad == True:
        # sigmas_vals = set_thickness(graph, Dg=Dg).nodes.data('sigma_mask')
        sigmas_vals = graph.nodes.data('sigma_mask')
        soma_rad = max(np.array(list(sigmas_vals))[:, 1]) * 0.5
        radius = [soma_rad]
        
        for z in list(zip(X, Y, Z))[1:]:
            #  radii = 1/2 sigma
            radius.append((sigmas_vals[z])/2)
        radius = np.array(radius)

    data['#index'] = np.array(POS)
    data['type'] = ntype
    if ratio != None:
        data['X'] = np.array(X) * ratio[0]
        data['Y'] = np.array(Y) * ratio[1]
        data['Z'] = np.array(Z) * ratio[2]
    else:
        data['X'] = np.array(X)
        data['Y'] = np.array(Y)
        data['Z'] = np.array(Z)
    data['radius'] = radius
    data['parent'] = np.array(PAR)
    data.write(filename, format='ascii', overwrite=True)


