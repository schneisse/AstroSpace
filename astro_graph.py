from collections import defaultdict

import numpy as np
import networkx as nx

from tqdm.auto import tqdm


def draw_nodes(pos, nodelist):
    return np.asarray([pos[n] for n in nodelist])


def choose_main(chosen_keys, values, mass_func=len):
    '''values - dict with keys contain chosen_keys and which values we should compare'''
    max_mass = 0
    if not len(chosen_keys):
        raise Exception('ERROR! chosen_keys are empty. Please check your data and try again')
    for key in chosen_keys:
        value = values[key]
        value_mass = mass_func(value)
        if value_mass > max_mass:
            max_mass = value_mass
            main_key = key
            main_value = values[main_key]
    return main_key, main_value


class AstroGraph(nx.Graph):
    version = 1.0
    def __init__(self, graph):
        self.graph = graph

    @classmethod
    def convert(cls, obj):
        if 'version' in obj.__dict__.keys() and obj.version == cls.version:
            return obj
        new_obj = cls(obj.graph)
        return new_obj


    @classmethod
    def batch_compose_all(cls, tip_paths, batch_size=10000, verbose=True):
        graphs = []
        for i, tp in enumerate(tqdm(tip_paths, disable=not verbose)):
            graphs.append(AstroGraph.path_to_graph(tp))
            if i % batch_size == 0:
                gx_all = nx.compose_all(graphs)
                graphs = [gx_all]
        return cls(nx.compose_all(graphs))


    @staticmethod
    def path_to_graph(path):
        "Converts an ordered list of points (path) into a directed graph"
        g = nx.DiGraph()

        root = tuple(path[-1])
        visited = set()
        for k,p in enumerate(path):
            tp = tuple(p)
            if not tp in visited:
                g.add_node(tp, root=root)
                if k > 0:
                    g.add_edge(tp, tprev, weight=1)
                tprev = tp
                visited.add(tp)
        return g

    @property
    def type(self):
        return type(self.graph)

    @property
    def nodes(self, data=False):
        return self.graph.nodes(data=data)

    # def nodes(self):
    #     return self.graph.nodes()

    @property
    def edges(self, data=False):
        return self.graph.edges(data=data)

    # @property
    # def _node(self):


    def predecessors(self, node):
        return self.graph.predecessors(node)

    def successors(self, node):
        return self.graph.successors(node)

    @property
    def tips(self):
        return {n for n in self.nodes if len(list(self.successors(n))) == 0}

    @property
    def roots(self):
        return {n for n in self.nodes if len(list(self.predecessors(n))) < 1}


    def get_sorted_roots(self):
        return sorted(self.roots,
                  key=lambda r: len(self.filter_graph(lambda n: n['root']==r)),
                  reverse=True,)

    @property
    def branches(self):
        branches = {}
        for root in self.roots:
            branches[root] = AstroGraph(self.filter_graph(lambda node: node['root'] == root))
        return branches


    def get_branch_points(self):
        return {n for n in self.nodes if len(list(self.successors(n))) > 1}


    def get_processors(self):
        raise Exception('ERROR!')


    def get_attrs_by_nodes(self, arr, func=None):
        nodesG = np.array(self.nodes())
        attrs = arr[nodesG[:,0], nodesG[:,1], nodesG[:,2]]
        if func is not None:
            func_vect = np.vectorize(func)
            attrs = func_vect(attrs)
        return {tuple(node): attr for node, attr in zip(nodesG, attrs)}


    def subgraph(self, nodes):
        return self.graph.subgraph(nodes)


    def add_edge(self, start, end, **attr):
        self.graph.add_edge(start, end, **attr)


    def add_node(self, node, **attr):
        self.graph.add_node(node, **attr)


    def check_for_cycles(self, verbose=False):
        try:
            cycle = nx.find_cycle(self.graph)
            if verbose:
                print('Found a cycle:', cycle)
            return cycle
        except nx.exception.NetworkXNoCycle:
            if verbose:
                print('No cycles!')
            return None


    def filter_graph(self, func = lambda node: True):
        "returns a view on graph for the nodes satisfying the condition defined by func(node)"
        good_nodes = (node for node in self.graph if func(self.nodes[node]))
        return self.subgraph(good_nodes)


    def get_bunches(self, min_dist=4):
        bunches = []
        roots = self.roots
        roots_arr = np.array(sorted(list(roots)))

        rvecs = {tuple(root): np.array([*self.successors(tuple(root))]) - np.array(root) for root in roots_arr}
        rvecs_arr = np.array([np.array([*self.successors(tuple(root))][0]) - np.array(root) for root in roots_arr]) # May be more than 1 root successor but we ignore that and choosed first

        for root, rvec in rvecs.items():

            def cosine_arr(vec):
                norm_root = np.sum(rvec**2)**0.5
                norm_vec = np.sum(vec**2)**0.5

                normprod = norm_root*norm_vec
                dprod = np.sum(rvec*vec)

                return dprod/normprod

            cos_dist = np.apply_along_axis(cosine_arr, 1, rvecs_arr)
            roots_dists = np.linalg.norm(np.array(root) - roots_arr, axis=-1)
            neighbours = set([tuple(r) for r in roots_arr[(roots_dists < min_dist)*(cos_dist > 0.99)]])
            for bunch in bunches:
                if bunch & neighbours:
                    bunch.update(neighbours)
                    break
            else:
                bunches.append(set(neighbours))

        set2del = []

        for i, cur_bunch in enumerate(bunches[:-1]):
            for node in cur_bunch:
                for bunch in bunches[i+1:]:
                    if node in bunch:
                        bunch.update(cur_bunch)
                        set2del.append(i)
                        break
                if i in set2del:
                    break

        if set2del:
            bunches.pop(*set2del)

        return bunches


    def get_sector(self, point):
        selected_nodes = set([point])
        to_visit = set(self.successors(point))
        while to_visit:
            node = to_visit.pop()
            selected_nodes.add(node)
            to_visit.update(set(self.successors(node)))
        return selected_nodes


    def cut_branches(self, nodes):
        if type(nodes) is tuple:
            nodes = [nodes]
        for node in nodes:
            sector_nodes = self.get_sector(node)
            self.graph.remove_nodes_from(sector_nodes)


    def remove_parallels(self, min_dist=4):
        bunches = self.get_bunches(min_dist)
        branches = self.branches
        pos = {node: node for node in self.nodes}

        for bunch in bunches:
            main_branch_root, main_branch = choose_main(bunch, branches, lambda x: len(x.nodes()))
            main_branch_lines = AstroGraph.make_lines(main_branch, main_branch_root)
            main_branch_line_tip, (main_branch_line, main_branch_line_mass) = choose_main(main_branch.tips, main_branch_lines)
            main_branch_points = draw_nodes(pos, main_branch_line)

            # mr, mb = choose_main(bunch, branches, lambda x: len(x.nodes()))
            # main_branch = Branch(mb, mr)

            for branch_root in tqdm(bunch):
                # Can be commented if need to remove parallels from branch itself (NOT WORKING FOR NOW)
                if branch_root == main_branch_root:
                    continue
                branch = branches[branch_root]
                nx.set_node_attributes(self.graph, {p: main_branch_root for p in branch.nodes()}, name='root')


                for line, line_mass in AstroGraph.make_lines(branch, branch_root).values():
                    points = draw_nodes(pos, line)

            #         branch_paths = list(branch.graph_to_paths().values())
            #         for path in branch_paths[0]:
            #             path = [branch_root] + path
            #             points = draw_nodes(pos, path)

                    count = min(len(points), len(main_branch_points))
                    dists = np.linalg.norm(points[:count] - main_branch_points[:count], axis=-1)
                    self.clear_line(points[:count], main_branch_points[:count], dists, min_dist)
        self.check_roots()


    def clear_line(self, points, main_points, dists, min_dist=4):
        for p, mbp, d in zip(points, main_points, dists):
            point = p
            mb_point = mbp

            if tuple(p) not in self.graph or tuple(p) == tuple(mbp):
                continue
            elif self.graph.nodes[tuple(p)]['sigma_mask'] == self.graph.nodes[tuple(mbp)]['sigma_mask'] \
                or d <= min_dist//2:
#                 min(data.graph.nodes[tuple(mbp)]['sigma_opt'], data.graph.nodes[tuple(p)]['sigma_opt']):
                self.graph.remove_node(tuple(p))
            else:
                break

        else:
            point = mb_point

        # print('start_point: {}, end_point: {}'.format(mb_point, point))
        self.connect_points(mb_point, point)


    def connect_points(self, start_point, end_point):
        cur_p = start_point
        prev_p = start_point
        end_p = end_point
        azi = np.array([*np.sign(end_p - cur_p)])

        root = self.nodes[tuple(start_point)]['root']

        while tuple(cur_p) != tuple(end_p):
            cur_p = np.clip(cur_p + azi, np.min([start_point, end_point], axis=0), np.max([start_point, end_point], axis=0))

            if self.graph.has_edge(tuple(prev_p), tuple(cur_p)) or self.graph.has_edge(tuple(cur_p), tuple(prev_p)):
                prev_p = cur_p
                continue
            self.graph.add_node(tuple(cur_p), root=root) #Add another parameters
            # print('prev_p: {}, cur_p: {}'.format(prev_p, cur_p))
            self.graph.add_edge(tuple(prev_p), tuple(cur_p))
            prev_p = cur_p


    #### VIZUALIZATIONS


    def view_graph_as_shapes(self, viewer, color=None, kind='points', name=None):
        """
        display nodes of graph g in napari viewer as points or as lines
        """
        if color is None:
            color = np.random.rand(3)
        pts = np.array(self.nodes)

        kw = dict(face_color=color, edge_color=color, blending='translucent_no_depth', name=name)
        #kw = dict(face_color=color, edge_color=color,  name=name)
        if kind == 'points':
            viewer.add_points(pts, size=1, symbol='square', **kw)
        elif kind == 'path':
            viewer.add_shapes(pts, edge_width=0.5, shape_type='path', **kw)

    def view_graph_as_colored_image(self, shape,
                                    viewer=None, name=None,
                                    root_chooser=lambda r: True,
                                    change_color_at_branchpoints=False):
        """
        Convert a graph to a colored 3D stack image and add it to a napari viewer.
        if the viewer instance is None, just return the colored 3D stack
        """
        paths = self.graph_to_paths(root_chooser=root_chooser)
        stack = self.paths_to_colored_stack(paths, shape, change_color_at_branchpoints)
        if viewer is not None:
            viewer.add_image(stack, channel_axis=3, colormap=['red','green','blue'], name=name)
            return viewer
        else:
            return stack

    def graph_to_paths(self, min_path_length=1, root_chooser=lambda r:True):
        """
        given a directed graph, return a list of a lists of nodes, collected
        as unbranched segments of the graph
        """

        roots = self.roots

        def _acc_segment(root, segm, accx):
            if segm is None:
                segm = []
            if accx is None:
                accx = []
            children = list(self.successors(root))

            if len(children) < 1:
                accx.append(segm)
                return

            elif len(children) == 1:
                c = children[0]
                segm.append(c)
                _acc_segment(c, segm, accx)

            if len(children) > 1:
                #segm.append(root)
                accx.append(segm)
                for c in children:
                    _acc_segment(c, [root, c], accx)

        acc = {}
        for root in roots:
            if root_chooser(root):
                px = []
                _acc_segment(root, [], px)
                acc[root] = [s for s in px if len(s) >= min_path_length]
        return acc


    def __str__(self):
        return str(self.graph)


    def __add__(self, other):
        new_graph =  AstroGraph(nx.compose(self.graph, other.graph))
        new_graph.check_roots()
        return new_graph

    def __radd__(self, other):
        new_graph = AstroGraph(nx.compose(other.graph, self.graph))
        new_graph.check_roots()
        return new_graph

    def __iadd__(self, other):
        self.gaph.update(other.graph)
        self.check_roots()


    def check_roots(self):
        for root in self.roots:
            try:
                nodes = self.get_sector(root)
            except:
                continue
            nx.set_node_attributes(self.graph, dict.fromkeys(nodes, root), 'root')


    def related_tips(self, root):
        
        # # collect tree nodes
            # coords = [i[0] for i in self.graph.nodes.data()]
            # all_roots = [i[1]["root"] for i in self.graph.nodes.data()]

            # #create root-specialized mask
            # x, y, z = root
            # root_mask = (np.array(all_roots)[:,0]==x) & (np.array(all_roots)[:,1]==y) & (np.array(all_roots)[:,2]==z)
            # root_nodes = [tuple(i) for i in np.array(coords)[root_mask]]

            # #get all tips
            # my_tips = np.array(list(self.tips))

            # #filter tips
            # root_tips = []

            # for tip in my_tips:
            #     tip = tuple(tip)

            #     if tip in root_nodes:
            #         root_tips.append(tip)

            # return root_tips
        
        try:
            # root_nodes = self.get_sector(root)
            tips = self.tips
            root_tips = [tip for tip in tips if tip['root'] == root]
            return root_tips

        except:
            nodes = list(self.nodes.data())
            root_nodes = [i for i,j in nodes if j['root'] == root]
            root_tips = [tip for tip in self.tips if tip in root_nodes]
            return root_tips


    def root_travel(self, root):
        root_path = {}
        root_path[root] = (1, -1)
        count = 2
        
        tips = self.related_tips(root)

        for tip in tips:
            for n in list(nx.shortest_path(self.graph, source=root, target=tip))[1:]:
                if n in root_path:
                    continue

                else:
                    num = count
                    #parent name
                    #return list with name of parent node
                    p_name = nx.predecessor(self.graph, root, n)
                    parent = root_path[p_name[0]][0]
                    root_path[n] = (num, parent)
                    count+=1

        return root_path

    def swc(self, center=None):

        roots  = self.roots
        collection = []

        if center is None:
            #connect all roots for continuous structure
            convergence = {AstroGraph.roots_convergence(roots): (1, -1)}
            collection.append(convergence)
        else:
            collection.append({center:(1, -1)})


        for r in tqdm(roots):
            visit = self.root_travel(r)

            #write first root
            if not collection:
                collection.append(visit)

            #write subsequent roots with updated vals
            else:
                value = max(collection[-1].values())[0]

                for i in visit.items():

                    #check if current node is root
                    if i[0] is not r:
                        new_pos = i[1][0] + value
                        new_par = i[1][1] + value
                        visit[i[0]] = (new_pos, new_par)

                    else:
                        new_pos = i[1][0] + value
                        # new_par = i[1][1]
                        new_par = 1
                        visit[i[0]] = (new_pos, new_par)

                collection.append(visit)

        return collection



    ## USEFUL FUNCTIONS


    @staticmethod
    def count_points_paths(paths):
        acc = defaultdict(int)
        for path in paths:
            for n in path:
                acc[n] += 1
        return acc


    @staticmethod
    def paths_to_colored_stack(paths, shape, change_color_at_branchpoints=False):
        #colors = np.random.randint(0,255,size=(len(paths),3))
        stack = np.zeros(shape + (3,), np.uint8)
        for root in paths:
            color =  np.random.randint(0,255, size=3)
            for kc,pc in enumerate(paths[root]):
                if change_color_at_branchpoints:
                    color = np.random.randint(0,255, size=3)
                for k,p in enumerate(pc):
                    #print(k, p)
                    stack[tuple(p)] = color
        return stack


    @staticmethod
    def find_paths(graph, stack_shape, targets, sources=None, min_count=1, min_path_length=10):
        length_dict, paths_dict = nx.multi_source_dijkstra(graph, targets, sources)

        #reverse order of points in paths, so that they start at tips
        if type(paths_dict) == list:
            if len(paths_dict) >= min_path_length:
                paths_dict = {paths_dict[-1]:paths_dict[::-1]}
            else:
                paths_dict = {}

            qstack = np.zeros(stack_shape)  #Это встречаемость точек в путях
            for p in list(paths_dict.values())[0]:
                qstack[p] = 1
            return qstack, paths_dict
        else:
            paths_dict = {path[-1]:path[::-1] for path in paths_dict.values() if len(path) >= min_path_length}
            paths = list(paths_dict.values())

            points = AstroGraph.count_points_paths(paths)

            qstack = np.zeros(stack_shape)  #Это встречаемость точек в путях
            for p, val in points.items():
                if val >= min_count:
                    qstack[p] = np.log(val)
            return qstack, paths_dict

    @staticmethod
    def roots_convergence(roots):
        roots = [r for r in roots]
        x = list((map(lambda x: x[0], roots)))
        y = list((map(lambda x: x[1], roots)))
        z = list((map(lambda x: x[2], roots)))

        return (np.average(x), np.average(y), np.average(z))


    @staticmethod
    def make_lines(branch, root):
        lines = {}
        for tip in branch.tips:
            lines[tip] = nx.shortest_path(branch.graph, root, tip), nx.shortest_path_length(branch.graph, root, tip)
        return lines