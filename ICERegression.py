from copy import deepcopy

import cloudpickle as cp
import jsonpickle as jp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx import topological_sort
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.kernel_density import EstimatorSettings

from CategoricalKernelReg import CategoricalKernelReg
from RegressionGraph import RegressionGraph


class ICERegression:

    def __init__(self, reggraph=None, data=None, cols=None, bw=None,
                 modemax=False, rounding=False, boxparents=False,
                 estset=EstimatorSettings(efficient=True)):

        # Node names should be strings, not numbers.
        if reggraph is None:
            self.reggraph = RegressionGraph()
        else:
            self.reggraph = reggraph
        if type(data) == pd.DataFrame:
            self.data = data
            self.cols = list(data.columns)
        if type(data) == np.ndarray and cols:
            self.data = pd.DataFrame(data=data, columns=cols)
            self.cols = cols
        if type(data) == np.ndarray and cols is None:
            self.cols = list(self.reggraph.directed.nodes())
            self.data = pd.DataFrame(data=self.data, columns=self.cols)
            print(
                """
                Note that the provided training data is of type numpy array.
                Consistency with the variable names provided by the regression graph object
                is not ensured, and the order of the variables in the created dataframe will
                be determined by the directed part of the graph.\n
                """
            )
        self.bw = bw
        self.modemax = modemax
        self.rounding = rounding
        self.boxparents = boxparents
        self.estset = estset
        self.models = dict()

    def check_for_data(self):
        if type(self.data) == pd.DataFrame:
            return True
        if type(self.data) == np.ndarray and self.cols:
            self.data = pd.DataFrame(data=self.data, columns=self.cols)
            print(
                """Note that the provided training data is of type numpy array.
                Consistency with the variable names provided by the regression graph object
                is not ensured, and the order of the variables in the created data frame will
                be determined by the directed part of the graph.\n"""
            )
            return True
        if type(self.data) == np.ndarray and self.cols is None:
            self.data = pd.DataFrame(data=self.data, columns=list(self.reggraph.directed.nodes()))
            print(
                """Note that the provided training data is of type numpy array.
                Consistency with the variable names provided by the regression graph object
                is not ensured, and the order of the variables in the created data frame will
                be determined by the directed part of the graph.\n"""
            )
            return True
        else:
            print("There was no training data provided, process is terminated.")
            return False

    # run this before learning, to use dummy variables in the place of categoricals    
    def var_dummify(self):
        if self.check_for_data():
            pass
        else:
            return False

        for i in [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] != 'context']:
            if self.reggraph.directed.nodes[i]['type'] == 'u':
                dummm = pd.get_dummies(self.data[i].astype('category'), prefix=i)
                in_g = deepcopy(self.reggraph.directed.in_edges(i, data=True))
                ou_g = deepcopy(self.reggraph.directed.out_edges(i, data=True))
                for j in dummm.columns.tolist():
                    self.reggraph.directed.add_nodes_from([(j, self.reggraph.directed.nodes[i])])
                    self.reggraph.directed.add_edges_from((n1, j, d.copy()) for (n1, n2, d) in in_g)
                    self.reggraph.directed.add_edges_from((j, n2, d.copy()) for (n1, n2, d) in ou_g)
                    self.reggraph.directed.nodes[j]['type'] = 'c'
                self.reggraph.directed.remove_node(i)
                self.data = self.data.drop(i, axis=1).join(dummm)

    @staticmethod
    def scottbw(x_learn):
        h0 = 1.06 * np.std(x_learn, axis=0) * np.shape(x_learn)[0] ** (- 1. / (4 + np.shape(x_learn)[1]))
        return h0

    def parents(self, i):
        if self.boxparents:
            return list(self.reggraph.directed.predecessors(i)) + list(self.reggraph.undirected.neighbors(i))
        else:
            return [x for x in self.reggraph.directed.predecessors(i)
                    if self.reggraph.directed.nodes[x]['box'] != self.reggraph.directed.nodes[i]['box']]

    # Fitting

    def learn(self):

        if self.check_for_data():
            pass
        else:
            return False

        if not self.modemax:
            self.var_dummify()

        nodes_notcontext = [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] != 'context']

        def learn_step(output, inputs):
            # We need a 0 if we do not want intercept
            # inputstring = "0 + " + " + ".join(inputs)
            # inputstring = " + ".join(inputs)

            # Take the appropirate subsets
            # Note: categorical variables should be coded with numbers
            # in order for CategoricalKernelReg to work

            # y_learn, x_learn = dmatrices( output + "~" + inputstring, data=self.data, return_type='dataframe' )

            y_learn = self.data[output].to_numpy(dtype='float32', copy=True)
            x_learn = self.data[inputs].to_numpy(dtype='float32', copy=True)

            # Fit regression models
            settings = self.estset
            vtypes = ''.join([self.reggraph.directed.nodes[inp]['type'] for inp in inputs])
            if self.bw == 'scott':
                if self.reggraph.directed.nodes[output]['type'] == 'u':
                    mod = CategoricalKernelReg(endog=y_learn, exog=x_learn, var_type=vtypes,
                                               bw=ICERegression.scottbw(x_learn), reg_type='lc', defaults=settings)
                else:
                    mod = KernelReg(y_learn, x_learn, var_type=vtypes,
                                    bw=ICERegression.scottbw(x_learn), reg_type='lc', defaults=settings)
            else:
                if self.reggraph.directed.nodes[output]['type'] == 'u':
                    mod = CategoricalKernelReg(endog=y_learn, exog=x_learn, var_type=vtypes,
                                               bw=self.bw, reg_type='lc', defaults=settings)
                else:
                    mod = KernelReg(y_learn, x_learn, var_type=vtypes,
                                    bw=self.bw, reg_type='lc', defaults=settings)
            self.models[output] = mod
            print('Parameters of ' + output + ' are learned.')

        for i in nodes_notcontext:
            learn_step(i, self.parents(i))

    # Prediction

    def predict(self, testdata_orig, plot_steps=False):

        testdata = deepcopy(testdata_orig)

        # testdata as df with the variables as column names
        nodes_notcontext = [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] != 'context']

        if not self.modemax:
            for i in nodes_notcontext:
                if (self.reggraph.directed.nodes[i]['type'] == 'u') & (i in testdata.keys().tolist()):
                    dummm = pd.get_dummies(testdata[i].astype('category'), prefix=i)
                    testdata = testdata.drop(i, axis=1).join(dummm)

        def predict_step(output, inputs, test_data):

            # inputstring = '0 + ' + " + ".join(inputs)
            # x_test = dmatrix( inputstring, data=testdata, return_type='matrix' )

            x_test = test_data[inputs].to_numpy(dtype='float32', copy=True)
            if (self.reggraph.directed.nodes[output]['type'] == 'o') & self.rounding:
                test_data.loc[:, output] = np.around(self.models[output].fit(x_test)[0])
            else:
                test_data.loc[:, output] = self.models[output].fit(x_test)[0]

        def plot_step(output, inputs, test_data):

            # dist = np.square((y_test_pred[0,:]-np.transpose(y_test))/(np.amax(y_test)-np.amin(y_test)))[0]
            # dist = np.square(test_data[output]-np.transpose(y_test))[0]
            for inp in inputs:
                plt.scatter(self.data[inp], self.data[output],
                            c='r', alpha=0.1, label='Learning Data')
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
                    if self.check_for_data():
                        pass
                    else:
                        return False
                    plot_step(i, parents, testdata)
        return testdata

    def test_on_train(self, plot_steps):

        if self.check_for_data():
            pass
        else:
            return False

        nodes_context = [x for x, y in self.reggraph.directed.nodes(data=True) if y['box'] == 'context']
        contextdata = self.data[nodes_context]
        pred_on_con = self.predict(contextdata, plot_steps)
        return pred_on_con

    def r_squared(self, output):

        if self.check_for_data():
            pass
        else:
            return False

        y = self.data[self.reggraph.directed.nodes[output]]
        y_hat = self.test_on_train(False)[self.reggraph.directed.nodes[output]]
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
