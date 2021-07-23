import numpy as np
import scipy as sp
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from modules.KernelRegWrapper import KernelRegWrapper


class DependentKernelReg(BaseEstimator, RegressorMixin):
    """
    A sklearn-style NW kernel regression with dependent bandwidth matrix
    and multi-variate normal kernel.
    """

    def __init__(self, kernel="exp", bw_init="scott"):
        self.outp = None
        self.regressors = None
        self.kernel = kernel
        self.bw_init = bw_init
        self.kernel_params = None
        self.params = []

    def fit(self, X, y):
        self.regressors = X
        self.outp = y

        cov_mat = np.cov(X, rowvar=False)
        n = X.shape[0]
        d = X.shape[1]

        H_init = 1
        if self.bw_init == "scott":
            H_init = 1.06 * n ** (-1. / (d + 4))
        elif self.bw_init == "silverman":
            H_init = (n * (d+2) / 4.) ** (-1. / (d + 4))

        if self.regressors.shape[1] == 1:
            H = H_init * np.sqrt(cov_mat)
        else:
            H = H_init * sp.linalg.sqrtm(cov_mat)

        self.params = {"sample_size": n, "no_of_regressors": d, "bw_matrix": H}

        if self.kernel == "exp":
            if self.regressors.shape[1] == 1:
                H_inv = H**(-1)
                H_det = H
            else:
                H_inv = np.linalg.inv(H)
                H_det = np.linalg.det(H)
            H_const = 1. / ((np.sqrt(2 * np.pi) ** d) * H_det) #0.5))
            self.kernel_params = {"H": H, "invH": H_inv, "detH": H_det,
                                  "dim": d, "const": H_const}

        return self

    def exp_kernel(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.regressors.shape[1] == 1:
            powa = -0.5 * (self.kernel_params["invH"] ** 2) * (X * X).sum(-1)
        else:
            xtimesH = np.matmul(X, self.kernel_params["invH"])
            powa = -0.5 * (xtimesH * xtimesH).sum(-1)  # equiv. to (invH * X)^T (invH *X)

        # powa = np.expand_dims(powa, axis=0)
        weight = self.kernel_params["const"] * np.exp(powa)

        return weight

    def predict(self, X):
        # preds = np.empty(X.shape[0])
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        def predict_step(i):
            x_shifted = np.subtract(self.regressors, i)
            k_weights = self.exp_kernel(x_shifted)
            pred = np.matmul(k_weights, np.squeeze(self.outp))
            pred /= np.sum(k_weights)
            return pred

        preds = np.apply_along_axis(predict_step, axis=1, arr=X)

        if isinstance(self.outp, pd.Series):
            preds = pd.Series(preds, name=self.outp.name)
        elif isinstance(self.outp, pd.DataFrame):
            preds = pd.DataFrame(preds, columns=self.outp.columns)

        return preds


#
# data = pd.read_excel("/home/baranyim/Documents/Adatok/Fatma_data/fatma_data_clean.xlsx")
# Fvarnames = {'Age_Woman': 'W_Age',
#              'Wealth_Index': 'Fam_Wealth',
#              'NumOfBornChildren': 'Num_Births',
#              'Age_Woman_atFristMar': 'W_Age_aFM',
#              'Age_Husband': 'H_Age',
#              'IdealNumOfChildren': 'Ideal_NoC',
#              'SchoolingYears_Woman': 'W_SchYears',
#              'SchoolingYears_Husband': 'H_SchYears',
#              'Contraception_Years': 'CM_Years'}
#
# data = data.rename(columns=Fvarnames)
#
# model = DependentKernelReg()
#
# XX = data[['W_Age', 'H_Age', 'W_SchYears', 'H_SchYears']]
# yy = data[['Fam_Wealth']]
#
# model.fit(XX, yy.to_numpy())
# # model.fit(XX, yy.squeeze())
# # model.fit(XX, yy)
#
# ps = model.predict(XX)
#
# print(yy.head(10))
# print(ps.shape, type(ps), ps)
#
# from sklearn.metrics import r2_score, mean_squared_error
#
# model2 = KernelRegWrapper(bw="scott")
# model2.fit(XX, yy)
# ps2 = model2.predict(XX)
#
# print("with_bw_matrix_r2: ", r2_score(yy, ps))
# print("indep_kernel_r2: ", r2_score(yy, ps2))
#
# print("with_bw_matrix_mse: ", mean_squared_error(yy, ps))
# print("indep_kernel_mse: ", mean_squared_error(yy, ps2))
