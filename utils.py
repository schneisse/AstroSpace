import os
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3
import SCA
import copy
from glob import glob

from treem import Morph, SWC

import tmd 
from tmd.view import view, plot
import morphio 
import neurom as nm
from neurom.view import matplotlib_impl, matplotlib_utils

from neuron_morphology.swc_io import morphology_from_swc
from neuron_morphology.feature_extractor.data import Data
import neuron_morphology.swc_io as swcio
from neuron_morphology.morphology import Morphology
from neuron_morphology.feature_extractor.feature_extractor import FeatureExtractor
from neuron_morphology.constants import (SOMA, AXON, BASAL_DENDRITE, APICAL_DENDRITE)
from neuron_morphology.features.soma import calculate_number_of_stems
from neuron_morphology.feature_extractor.marked_feature import specialize
from neuron_morphology.feature_extractor.feature_specialization import NEURITE_SPECIALIZATIONS
from neuron_morphology.feature_extractor.utilities import unnest
from neuron_morphology.features.dimension import dimension
from neuron_morphology.features.intrinsic import num_branches, num_tips, num_nodes, max_branch_order
from neuron_morphology.features.branching.bifurcations import num_outer_bifurcations
from neuron_morphology.features.default_features import total_length, total_volume, total_surface_area
import neuron_morphology.feature_extractor.feature_writer as fw

def get_files(path, format = '.swc'):
    files = []
    for f in glob(path + '/*' + format):
        files.append(f)
    return files

def define_domain(tree, _plot_=None, plot_save = False, filename = None, dpi=300):
    '''
    Input: tree class
    Output: scipy Convex Hull
    '''
    tips = [i for i in tree if len(i.children) == 0]
    pts = np.array([j.v for j in tips]) 
    hull = ConvexHull(pts)

    if _plot_ == 'line':
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Corners
        ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko", ms=0.5)

        # Surfaces
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], c="maroon", lw=0.5)
        plt.title(f'{filename}')

    if _plot_ == 'surface':
        fig = plt.figure(dpi=300, facecolor='black')
        ax = fig.add_subplot(111, projection="3d")
        ax.set_alpha(0.1)
        ax.set_facecolor('black')
        ax.grid(False)
        ax.axis('off')
        # ax.view_init(elev=50)

        for s in hull.simplices:
            tri = Poly3DCollection([pts[s]])
            tri.set_edgecolor('slateblue')
            tri.set_facecolor('blue')
            tri.set_alpha(0.09)
            tri.set_linewidth(0.3)
            ax.add_collection3d(tri)

        SCA.plot_tree(tree=tree, show_leaflets=False, ax=ax, tree_color='orangered')
        # ax.set_xlim3d(40, 80)
        # ax.set_ylim3d(40, 80)
        # ax.set_zlim3d(40, 80)
        plt.title(f'{filename}')

    if plot_save == True:
        plt.savefig(f'{filename}.png')

    return hull

def need_fix(tree):

    check = [node.v for node in tree if node.v[0]*node.v[1]*node.v[2]>0]
    if False in check:
        return True
    else: 
        return False

def fix_swc_coordinates(tree):
    center = tree.center
    R = tree.R
    new_tree = []

    # if need_fix(tree) == True:

class Units:
    def __init__(self, number, unit):
        Params = {}
        Params['milli'] = 1
        Params['micro'] = Params['milli'] * 10**(3)
        Params['centi'] = Params['milli'] * 10**(-1)
        Params['nano'] = Params['micro'] * 10**(3)
        Params['pico'] = Params['nano'] * 10**(3)
        self.params = Params
        self.number = number
        if unit in Params.keys():
            self.unit = unit
        else:
            raise KeyError(print('Unit do not recognized. Please add it according to the existing units'))
        
    def add_param(self, new_param, default_param, relation_to_default_param):
        if new_param in self.params.keys():
            raise KeyError(print('Parametes already exist'))
        if default_param in self.params.keys():
            self.params[new_param] = default_param * relation_to_default_param
            
    def convert(self, to_b):
        if to_b in self.params.keys():
            ratio = self.params[to_b]/self.params[self.unit]
            return self.number * ratio

def prepare_neuron_tree(swc_data):
    nodes = swc_data.to_dict('record')
    replace_type = 2 # default node type
    for node in nodes:
        node['parent'] = int(node['parent'])
        node['id'] = int(node['id'])
        node['type'] = int(node['type'])

        if node['parent'] == -1 and node['type'] != 1:
            replace_type = node['type']

        if node['type'] == 1 and node['parent'] != -1:
            node['type'] = replace_type

    soma_list = []
    for node in nodes:
        if node['type'] == 1:
            soma_list.append(node)

    # create a new soma point
    if len(soma_list) > 1:
        x = 0
        y = 0
        z = 0
        n = len(soma_list)
        for node in soma_list:
            x += node['x']
            y += node['y']
            z += node['z']

        soma = copy.deepcopy(soma_list[0])
        soma['id'] = nodes[-1]['id']
        soma['x'] = x/n
        soma['y'] = y/n
        soma['z'] = z/n
        nodes.append(soma)

        for node in soma_list:
            node['parent'] = soma['id']
            node['type'] = replace_type
    
    return nodes

def Sholl(morph, step=1):
    
    def dist(a, b):
        return np.linalg.norm(a-b)
    
    def crossings(morph, ident, step):
        h=step
        root = morph.root
        node = morph.node(ident)
        parent = node.parent.coord()
        r1 = dist(root.coord(), parent)
        r2 = dist(root.coord(), node.coord())
        k1 = int(r1/h)
        k2 = int(r2/h)

        return k2-k1, k1
        
        
    root = morph.root
    idents = [node.ident() for node in root.walk()]
    coords = [n.coord() for n in morph.nodes]
    
    rmax = np.max([dist(root.coord(), c) for c in coords])
    radx = np.array([k*step for k in range(int(rmax/step))])
    crox = np.zeros(int(rmax/step), dtype=int)
    
    for ident in idents[1:]:
        ncross, icross = crossings(morph, ident, step)
        if ncross > 0:
            for k in range(ncross):
                crox[icross+k] += 1
        elif ncross < 0:
            for k in range(1,-ncross):
                crox[icross-k] += 1
    
    return radx, crox

class SWC_analyse:
    def __init__(self, input_path, save_path):
        self.path = input_path
        self.filename = os.path.basename(input_path)
        self.name = os.path.splitext(self.filename)[0]
        self.format = os.path.splitext(self.filename)[1]
        self.savepath = save_path
        self.swc_data = swcio.read_swc(input_path)

    def tmd(self, save=True):
        cell = tmd.io.load_neuron(self.path)
        ph_astro = tmd.methods.get_ph_neuron(cell)
        plt.figure(dpi=200)
        plot.diagram(ph_astro, subplot=True)
        if save == True:
            plt.savefig(f'{self.name}_diagram.png', dpi=300)

        plot.barcode(ph_astro, subplot=True)
        if save == True:
            plt.savefig(f'{self.name}_barcode.png', dpi=300)

        plot.persistence_image(ph_astro, subplot=True)
        if save == True:
            plt.savefig(f'{self.name}_persistence_image.png', dpi=300)

    def morphofeatures(self, features=None, save=True, show_results=False, special_id = None):
        heavy_path = self.savepath + '/morphofeatures_' + self.name + '.h5'
        table_path = self.savepath + '/morphofeatures_' + self.name + '.csv'
        swc = swcio.read_swc(self.path)
        nodes = prepare_neuron_tree(swc)
        data = Data(Morphology(nodes, node_id_cb=lambda node: node['id'], parent_id_cb=lambda node: node['parent']))

        if features == None:
            features = [total_volume,
            total_surface_area, 
            total_length, 
            num_outer_bifurcations, 
            max_branch_order, 
            num_tips,
            num_branches,
            num_nodes]

        if save == True:
            feature_extraction_run = FeatureExtractor().register_features(features).extract(data)
            features_writer = fw.FeatureWriter(heavy_path, table_path)
            if special_id is not None:
                features_writer.add_run(special_id, feature_extraction_run.serialize())
            else:
                features_writer.add_run(self.name, feature_extraction_run.serialize())
            features_writer.write_table()

        else:
            results = feature_extraction_run.results
        
        if show_results == True and save == False:
            unnest(results)
    
    def sholl(self, step = 1, save = False):
        sholl = {}
        morphology = Morph(self.path)
        radx, crox = Sholl(morphology, step=step)
        sholl['radius'] = radx
        sholl['cross'] = crox

        if save == True:
            file = open(self.savepath+f'/{self.name}_sholl.txt', 'w')
            file.write('radius cross\n')
            for i in zip(radx, crox):
                file.write('{} {}\n'.format(i[0], i[1]))
            file.close()
        return sholl
