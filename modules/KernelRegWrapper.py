import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.bandwidths import bandwidth_funcs

from modules.CategoricalKernelReg import CategoricalKernelReg


class KernelRegWrapper(BaseEstimator, RegressorMixin):
    """ 
    A sklearn-style wrapper for statsmodels' KernelReg implementation.
    Categorical outputs are handled by the CategoricalKernelReg class.
    """

    def __init__(self, var_type=None, y_type="numeric", bw='cv_ls', reg_type='lc', settings=None):
        if y_type[:5] == "categ":
            self.model_class = CategoricalKernelReg
        else:
            self.model_class = KernelReg
        self.var_type = var_type
        self.bw = bw
        self.reg_type = reg_type
        self.settings = settings
        self.fitted_model = None

    def fit(self, X, y):
        if self.var_type is None:
            self.var_type = 'c' * X.shape[1]
        if self.bw in bandwidth_funcs.keys():
            self.bw = bandwidth_funcs[self.bw](X)
        elif self.bw == "scott2":
            self.bw = KernelRegWrapper.scott_bw(X)
        elif isinstance(self.bw, int):
            self.bw = [self.bw] * len(self.var_type)
        self.fitted_model = self.model_class(endog=y, exog=X, var_type=self.var_type,
                                             bw=self.bw, reg_type=self.reg_type, defaults=self.settings)
        return self

    def predict(self, X):
        return self.fitted_model.fit(data_predict=X)[0]

    @staticmethod
    def scott_bw(x_learn):
        h0 = 1.06 * np.std(x_learn, axis=0) * x_learn.shape[0] ** (- 1. / (4 + x_learn.shape[1]))
        return h0

