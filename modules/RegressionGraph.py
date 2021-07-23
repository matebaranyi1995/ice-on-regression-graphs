import os
import tempfile
from copy import deepcopy

import matplotlib.pyplot as plt
import jsonpickle as jp
import networkx as nx
import pandas as pd
from IPython.display import Image
from IPython import get_ipython
from networkx import Graph
from networkx import DiGraph
from networkx.readwrite import json_graph


class RegressionGraph:

    def __init__(self, incoming_digraph=None, incoming_ungraph=None,
                 boxes=None, types=None, reversed_input=False):

        if types is None:
            types = {}
        if boxes is None:
            boxes = {}
        self.directed = DiGraph(incoming_digraph)
        if incoming_digraph is None and boxes is None:
            print("Empty graph object is created, you should fill it up.")
        if reversed_input:
            edge_data = deepcopy(self.directed.edges(data=True))
            for u, v, d in edge_data:
                self.directed.remove_edge(u, v)
                self.directed.add_edge(v, u)
                self.directed[v][u].update(d)
        if boxes == {}:
            print("Boxes were not provided.")
            self.boxes = {'context': list(self.directed.nodes())}
        else:
            self.boxes = boxes
            if incoming_digraph is None:
                print("Boxes are provided, but no graph object.")
                self.directed.add_nodes_from([i for j in list(self.boxes.values()) for i in j])
        for i in self.boxes.keys():
            for j in self.boxes[i]:
                self.directed.nodes[j]['box'] = i
        for i in list(self.directed.nodes()):
            if i in types.keys():
                self.directed.nodes[i]['type'] = types[i]
            else:
                self.directed.nodes[i]['type'] = 'c'
        self.undirected = Graph(incoming_ungraph)
        edge_data = deepcopy(self.directed.edges(data=True))
        for u, v, d in edge_data:
            if self.directed.nodes[u]['box'] == self.directed.nodes[v]['box']:
                self.directed.remove_edge(u, v)
                self.undirected.add_edge(u, v)
                self.undirected[u][v].update(d)

    def reggraph_to_dot(self, labels, splinestyle):
        union_graph = nx.compose(self.directed, self.undirected)
        union_graph = nx.nx_agraph.to_agraph(union_graph)
        union_graph.node_attr['shape'] = 'box'
        union_graph.graph_attr['rankdir'] = 'RL'
        union_graph.graph_attr['splines'] = splinestyle
        for i in self.boxes:
            union_graph.add_subgraph(self.boxes[i], rank='same', name='cluster_' + i, rankdir='RL', label=i)
            # ToDo: check if s = ... is needed or not
        for s in union_graph.subgraphs_iter():
            if s.get_name() == 'cluster_context':
                for i in s.nodes_iter():
                    i.attr['style'] = "filled"
                    i.attr['color'] = "lightgrey"
            for e in s.edges_iter():
                e.attr['dir'] = 'none'
                e.attr['constraint'] = 'false'
                if s.get_name() != 'cluster_context':
                    e.attr['style'] = 'dashed'
                if e in s.reverse().edges():
                    # e.attr['style']='invis'
                    union_graph.delete_edges_from([e])
        if labels:
            for j in union_graph.nodes_iter():
                j.attr['label'] = labels[j.get_name()]

        return union_graph

    def draw_it(self, formating='png', prog='dot', splinestyle="spline", labels=None, *args, **kwargs):
        # print(self.reggraph_to_dot())
        g = self.reggraph_to_dot(splinestyle=splinestyle, labels=labels)
        if get_ipython().__class__.__name__ in ("NoneType", "PyDevTerminalInteractiveShell"):
            tempf = tempfile.NamedTemporaryFile(suffix='.png')
            tempf.close()
            g.draw(path=tempf.name, format=formating, prog=prog, *args, ** kwargs)
            im = plt.imread(tempf.name)
            plt.imshow(im)
            plt.axis('off')
            plt.draw()
            plt.show()
            os.remove(tempf.name)
            return True
        else:
            return Image(g.draw(format=formating, prog=prog, *args, **kwargs))

    def save_it(self, fp='regressiongraph.png', format='png', prog='dot', splinestyle="spline", labels=None,
                *args, **kwargs):
        g = self.reggraph_to_dot(splinestyle=splinestyle, labels=labels)
        return g.draw(path=fp, format=format, prog=prog, *args, **kwargs)

    def serialize(self):
        graph_cp = deepcopy(self)
        graph_cp.undirected = json_graph.node_link_data(graph_cp.undirected)
        graph_cp.directed = json_graph.node_link_data(graph_cp.directed)
        return jp.encode(graph_cp)

    @staticmethod
    def deserialize(serialized_reggraph):
        reggraph = jp.decode(serialized_reggraph)
        reggraph.undirected = json_graph.node_link_graph(reggraph.undirected)
        reggraph.directed = json_graph.node_link_graph(reggraph.directed)
        return reggraph


def build_reggraph_by_r_script(dataframe, boxes, types):
    # the output of this should be the arg incoming_digraph
    # with reversed_input=False
    # the first key of boxes should be the 'context'

    if dataframe.shape[0] > 1000:
        dataframe.sample(n=1000).to_json("data.json", orient="records")
    else:
        dataframe.to_json("data.json", orient="records")

    rb = '~'.join(['+'.join(boxes[key]) for key in sorted(boxes)])
    # print(rb)
    rt = ""
    for box in [boxes[key] for key in sorted(boxes)]:
        print(box)
        for v in box:
            if types[v] == 'c':
                rt += 'cont,'
            if types[v] == 'o':
                rt += 'count,'
    rt = rt[:-1]
    # print(rt)

    import subprocess
    subprocess.call(["Rscript", "graph_builder.R", "data.json", rb, rt])

    adja = pd.read_csv('reggraph_adjmat.csv', sep=',')
    ggg = nx.convert_matrix.from_numpy_matrix(adja.iloc[:, 1:].values, create_using=nx.DiGraph)
    # print(graph.nodes)
    keys = list(ggg.nodes)
    values = adja.iloc[:, 0]
    ggg = nx.relabel_nodes(ggg, dict(zip(keys, values)))
    return ggg
