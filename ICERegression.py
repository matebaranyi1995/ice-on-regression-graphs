import warnings
import inspect
from copy import deepcopy

import cloudpickle as cp
import jsonpickle as jp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx import topological_sort

from modules.RegressionGraph import RegressionGraph


class ICERegression:

    def __init__(self, reggraph=None,
                 dummify=False, rounding=False, boxparents=False, orderedascat=False,
                 num_mod_class=None, num_mod_args=None,
                 cat_mod_class=None, cat_mod_args=None,
                 num_mod_sel_class=None, num_mod_sel_args=None,
                 cat_mod_sel_class=None, cat_mod_sel_args=None):

        # Node names should be strings, not numbers.
        if reggraph is None:
            self.reggraph = RegressionGraph()
        else:
            self.reggraph = reggraph

        self.variables = list(self.reggraph.directed.nodes())
        # ToDo: implement consistency check for the two input graphs

        self.dummify = dummify
        self.rounding = rounding
        self.boxparents = boxparents
        self.orderedascat = orderedascat

        self.models = dict()

        # model arguments as dictionaries
        if num_mod_args is None and cat_mod_args is None:
            self.num_mod_args = {}
            self.cat_mod_args = {}
        elif num_mod_args and (cat_mod_args is None):
            self.num_mod_args = num_mod_args
            self.cat_mod_args = num_mod_args
        else:
            self.num_mod_args = num_mod_args
            self.cat_mod_args = cat_mod_args

        self.num_mod_class = num_mod_class
        if num_mod_class and (cat_mod_class is None):
            self.cat_mod_class = num_mod_class
        else:
            self.cat_mod_class = cat_mod_class

        self.cat_mod_sel_class = cat_mod_sel_class

        if cat_mod_sel_args is not None:
            self.cat_mod_sel_args = cat_mod_sel_args
        else:
            self.cat_mod_sel_args = {}

        self.num_mod_sel_class = num_mod_sel_class

        if num_mod_sel_args is not None:
            self.num_mod_sel_args = num_mod_sel_args
        else:
            self.num_mod_sel_args = {}

        self.dummified_vars = []

    # run this before learning on the data for compatibility
    def check_data(self, data):
        if type(data) == pd.DataFrame and self.variables:
            if set(list(data)) == set(self.variables):
                return data
            else:
                ICERegression.user_warnings("inconsdata")
                raise TypeError
        if type(data) == np.ndarray and self.variables:
            data = pd.DataFrame(data=data, columns=self.variables)
            ICERegression.user_warnings("numpy")
            return data
        else:
            ICERegression.user_warnings("unknowndata")
            raise TypeError

    # run this before learning, to use dummy variables in the place of categoricals    
    def var_dummify(self, data):
        try:
            data = self.check_data(data)
        except TypeError:
            return False

        for i in [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] != 'context']:
            if self.reggraph.directed.nodes[i]['type'] == 'u':
                dummm = pd.get_dummies(data[i].astype('category'), prefix=i)
                in_g = deepcopy(self.reggraph.directed.in_edges(i, data=True))
                ou_g = deepcopy(self.reggraph.directed.out_edges(i, data=True))
                for j in dummm.columns.tolist():
                    self.reggraph.directed.add_nodes_from([(j, self.reggraph.directed.nodes[i])])
                    self.reggraph.directed.add_edges_from((n1, j, d.copy()) for (n1, n2, d) in in_g)
                    self.reggraph.directed.add_edges_from((j, n2, d.copy()) for (n1, n2, d) in ou_g)
                    self.reggraph.directed.nodes[j]['type'] = 'c'
                self.reggraph.directed.remove_node(i)
                self.dummified_vars.append(i)
                data = data.drop(i, axis=1).join(dummm)
        return data

    def parents(self, i):
        if self.boxparents:
            return list(self.reggraph.directed.predecessors(i)) + list(self.reggraph.undirected.neighbors(i))
        else:
            return [x for x in self.reggraph.directed.predecessors(i)
                    if self.reggraph.directed.nodes[x]['box'] != self.reggraph.directed.nodes[i]['box']]

    def add_vtype_args(self, inputs, catmod=False):
        vtypes = ''.join([self.reggraph.directed.nodes[inp]['type'] for inp in inputs])
        if catmod:
            self.cat_mod_args['var_type'] = vtypes
        else:
            self.num_mod_args['var_type'] = vtypes

    # Fitting

    def fit(self, data):
        try:
            data = self.check_data(data)
        except TypeError:
            return False

        if self.dummify:
            data = self.var_dummify(data)

        noncontext_nodes = [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] != 'context']

        def fit_step_step(xlearn, ylearn, catornum):
            if catornum == "cat":
                mod = self.cat_mod_class(**self.cat_mod_args)
                if self.cat_mod_sel_class is None:
                    mod.fit(xlearn, ylearn)
                else:
                    mod_sel = self.cat_mod_sel_class(mod, **self.cat_mod_sel_args)
                    mod_sel.fit(xlearn, ylearn)
                    mod = mod_sel.best_estimator_
            elif catornum == "num":
                mod = self.num_mod_class(**self.num_mod_args)
                if self.num_mod_sel_class is None:
                    mod.fit(xlearn, ylearn)
                else:
                    mod_sel = self.num_mod_sel_class(mod, **self.num_mod_sel_args)
                    mod_sel.fit(xlearn, ylearn)
                    mod = mod_sel.best_estimator_
            return mod

        def fit_step(output, inputs):

            # Take the appropriate subsets
            y_learn = data[output].to_numpy(dtype='float32', copy=True)
            x_learn = data[inputs].to_numpy(dtype='float32', copy=True)

            # Fit regression models
            if 'var_type' in inspect.signature(self.num_mod_class).parameters.keys():
                self.add_vtype_args(inputs, catmod=False)
            if 'var_type' in inspect.signature(self.cat_mod_class).parameters.keys():
                self.add_vtype_args(inputs, catmod=True)

            if self.reggraph.directed.nodes[output]['type'] == 'u':
                mod = fit_step_step(x_learn, y_learn, "cat")
            elif self.reggraph.directed.nodes[output]['type'] == 'c':
                mod = fit_step_step(x_learn, y_learn, "num")
            elif self.reggraph.directed.nodes[output]['type'] == 'o':
                if self.orderedascat:
                    mod = fit_step_step(x_learn, y_learn, "cat")
                else:
                    mod = fit_step_step(x_learn, y_learn, "num")
            else:
                raise KeyError("Response variable without proper type")

            self.models[output] = mod
            print('Parameters of ' + output + ' are learned.')

        for i in noncontext_nodes:
            fit_step(i, self.parents(i))

    # Prediction

    def predict(self, testdata, traindata=None, plot_steps=False):

        # testdata as df with the variables as column names
        # ToDo: implement something if the test data is sane
        testdata = deepcopy(testdata)

        nodes_notcontext = [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] != 'context']

        if self.dummify:
            for i in nodes_notcontext:
                if (i in self.dummified_vars) & (i in testdata.keys().tolist()):
                    # ToDo: check if this works
                    dummm = pd.get_dummies(testdata[i].astype('category'), prefix=i)
                    testdata = testdata.drop(i, axis=1).join(dummm)

        def predict_step(output, inputs, test_data):

            x_test = test_data[inputs].to_numpy(dtype='float32', copy=True)
            if ((self.reggraph.directed.nodes[output]['type'] == 'o') | (output in self.dummified_vars))\
                    & self.rounding:
                test_data.loc[:, output] = np.around(self.models[output].predict(x_test))
            else:
                test_data.loc[:, output] = self.models[output].predict(x_test)

        def plot_step(output, inputs, test_data, train_data):

            # dist = np.square((y_test_pred[0,:]-np.transpose(y_test))/(np.amax(y_test)-np.amin(y_test)))[0]
            # dist = np.square(test_data[output]-np.transpose(y_test))[0]
            for inp in inputs:
                if train_data is not None:
                    plt.scatter(train_data[inp], train_data[output],
                                c='r', alpha=0.1, label='Training Data')
                plt.scatter(test_data[inp], test_data[output],
                            c='b', alpha=0.1, label='Kernel Regression')
                # plt.colorbar()
                plt.xlabel(inp)
                plt.ylabel(output)
                plt.title(output + ' over its parents')
                plt.legend()
                plt.show()

        for i in [j for j in topological_sort(self.reggraph.directed)]:
            if (i in nodes_notcontext) & (i not in testdata.keys().tolist()):
                parents = self.parents(i)
                predict_step(i, parents, testdata)
                if plot_steps:
                    plot_step(i, parents, testdata, traindata)
        return testdata

    def test_on_train(self, traindata, plot_steps):

        try:
            traindata = self.check_data(traindata)
        except TypeError:
            return False

        nodes_context = [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] == 'context']
        contextdata = traindata[nodes_context]
        pred_on_con = self.predict(testdata=contextdata, traindata=traindata, plot_steps=plot_steps)
        return pred_on_con

    def r_squared(self, data, output):

        try:
            data = self.check_data(data)
        except TypeError:
            return False

        y = data[self.reggraph.directed.nodes[output]]
        y_hat = self.test_on_train(data, False)[self.reggraph.directed.nodes[output]]
        y_bar = np.mean(y_hat)
        r2_numer = (((y - y_bar) * (y_hat - y_bar)).sum()) ** 2
        r2_denom = ((y - y_bar) ** 2).sum(axis=0) * \
                   ((y_hat - y_bar) ** 2).sum(axis=0)
        return r2_numer / r2_denom

    def __repr__(self):
        """Provide something sane to print."""
        st = ''
        for i in self.reggraph.directed.node(data=True):
            st += str(i)
            st += '\n'
        if self.models:
            for k, v in self.models.items():
                st += 'model of ' + str(k) + '\n'
                st += str(v) + '\n'
        else:
            st += 'Models are not trained yet.'
        return st

    # Serialization

    def serialize(self):
        mod_cp = deepcopy(self)
        mod_cp.data = None
        for k in list(mod_cp.models):
            mod_cp.models[k] = cp.dumps(mod_cp.models[k])
        mod_cp.reggraph = mod_cp.reggraph.serialize()
        return jp.encode(mod_cp)

    @staticmethod
    def deserialize(serialized_ice, data=None):
        ice = jp.decode(serialized_ice)
        ice.reggraph = RegressionGraph.deserialize(ice.reggraph)
        for k in list(ice.models):
            ice.models[k] = cp.loads(ice.models[k])
        if type(data) == pd.DataFrame:
            ice.data = data
        elif type(data) == np.ndarray and ice.cols:
            ice.data = pd.DataFrame(data=data, columns=ice.cols)
        return ice

    @staticmethod
    def user_warnings(message):
        msg = "Unspecified warning"
        if message == "numpy":
            msg = """
                    Note that the provided training data is a numpy array.
                    Consistency with the variable names provided by the regression graph object
                    is not ensured, and the order of the variables in the created data frame will
                    be determined by the directed part of the graph.\n
                    """
        if message == "unkowndata":
            msg = """
                    Note that the provided training data is not a supported type,
                    and/or variable names are undefined by the graph objects.\n
                    """
        if message == "inconsdata":
            msg = """
                    Note that the provided training data is a pandas data frame,
                    but the column names and the variable names provided by the graph objects
                    are different.\n
                    """
        warnings.warn(msg)
