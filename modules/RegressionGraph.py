import os
import tempfile
import subprocess
import warnings
from copy import deepcopy
from pathlib import Path

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
    """
    Class for handling the structure of a regression graph. 
    """

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
        if incoming_ungraph is None:
            self.undirected.add_nodes_from(self.directed)
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


def build_reggraph_by_r_script(dataframe, boxes, types=None):
    """ 
    It learns the structure of a MVR chain graph based on an R script.
    The edges of the 'context' box should be re-evaluated because MVR 
    chain graphs use covariance graphs here too. 
    However, it has no effect on the ICE algorithm, 
    so this step can be skipped.

    Notes:
      * The output of this should be the arg incoming_digraph of 
      the RegressionGraph class with reversed_input=False.
      * The first key of boxes should be named 'context'.
    """

    Path("tmp").mkdir(parents=True, exist_ok=True)

    if dataframe.shape[0] > 1000:
        dataframe.sample(n=1000).to_json("tmp/data.json", orient="records")
    else:
        dataframe.to_json("tmp/data.json", orient="records")


    variables = list(boxes.values())
    if list(boxes.keys())[0] == "context":
        variables.reverse()
    elif list(boxes.keys())[-1] == "context":
        pass
    else:
        warnings.warn("""
        The first or last box should be named `context` to work with the 
        `ICERegression` class.""")

    rb = '~'.join(['+'.join(variabs) for variabs in variables])
    print(rb)

    # # the variable types are handled by the R parts instead
    # rt = ""
    # for box in [boxes[key] for key in sorted(boxes)]:
    #     print(box)
    #     for v in box:
    #         if dataframe[v].nunique() == 2:
    #             rt += 'bin,'
    #         elif types[v] == 'c':
    #             rt += 'cont,'
    #         elif types[v] == 'o':
    #             rt += 'ord,'
    #         elif types[v] == 'u':
    #             rt += 'count,'
    # rt = rt[:-1]
    # print(rt)

    subprocess.call(["Rscript", "modules/graph_builder.R", "tmp/data.json", rb])#, rt])

    adja = pd.read_csv('tmp/reggraph_adjmat.csv', sep=',', index_col=0)
    print(adja)
    dir_graph = nx.convert_matrix.from_numpy_matrix(adja.values, create_using=nx.DiGraph)
    # print(graph.nodes)
    keys = list(dir_graph.nodes)
    values = list(adja)
    dir_graph = nx.relabel_nodes(dir_graph, dict(zip(keys, values)))

    reg_graph = RegressionGraph(incoming_digraph=dir_graph, types=types,
                            boxes=boxes, reversed_input=False)
    return reg_graph, dir_graph
