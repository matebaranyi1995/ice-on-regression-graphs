import pandas as pd
import numpy as np

from statsmodels.nonparametric.kernel_regression import KernelReg


class CategoricalKernelReg:
    """ 
    Kernel regression implementation with categorical output.
    The maximally probable class will be predicted based on the 
    kernel regression of the one-hot encoded dummies.
    Notes: 
      * Categorical variables should be encoded with numbers
      in order for this class to work.
      * It works based on statsmodels' KernelReg implementation.
      * It is supported by the sklearn-like KernelRegWrapper. 
    """
    
    def __init__(self, endog, exog, var_type, reg_type, bw, *args, **kwargs):
        endog_c = endog.astype('category')
        dummy_out = pd.get_dummies(endog_c, prefix=[''], prefix_sep=[''])
        self.categs = list(dummy_out.columns)
        mods = {}
        for col in dummy_out:
            dum = dummy_out[col]
            mods[col] = KernelReg(dum, exog, var_type, reg_type, bw, *args, **kwargs)
        self.models = mods
        
        # for repr purposes
        self.var_type = var_type
        self.reg_type = reg_type
        self.k_vars = len(self.var_type)
        self.nobs = np.shape(endog)[0]
        if not isinstance(bw, str):
            self._bw_method = "user-specified"
        else:
            self._bw_method = bw
        
    def fit(self, data_predict=None):
        preds = pd.DataFrame()
        for k, v in self.models.items():
            preds[k] = v.fit(data_predict=data_predict)[0]
        mean = preds.idxmax(axis=1)
        return mean
    
    def __repr__(self):
        """Provide something sane to print."""
        rpr = "CategoricalKernelReg instance\n"
        rpr += "Number of variables: k_vars = " + str(self.k_vars) + "\n"
        rpr += "Number of samples:   N = " + str(self.nobs) + "\n"
        rpr += "Variable types:      " + self.var_type + "\n"
        rpr += "BW selection method: " + self._bw_method + "\n"
        rpr += "Estimator type: " + self.reg_type + "\n"
        return rpr
