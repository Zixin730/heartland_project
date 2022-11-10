import pandas as pd
import numpy as np
from cmath import nan
from scipy import linalg

def solve_sym(xtx, xty):
    mat = linalg.cholesky(xtx) 
    return linalg.lapack.dpotrs(mat, xty)[0]

def m_ols(x, y, inter=True, r2 = False):
    n, _ = np.shape(x)
    X = np.copy(x)
    if inter:
        X = np.c_[np.ones((n, 1)), X] 
    xt = X.T
    xtx = np.dot(xt, X)
    xty = np.dot(xt, y)
    coe = solve_sym(xtx, xty)
    # if inter:
    #     return coe[0], coe[1:]
    # else:
    if r2:
        y_pred = np.sum(coe*X, 1)
        r_square = 1- ((y - y_pred)** 2).sum()/((y - y.mean()) ** 2).sum()
        return np.r_[coe, r_square]
    else:
        return coe

def cal_rolling_coe(x, y, rolling_window=30):
    if len(y) <= x.shape[1]:
        return np.NaN
    elif len(y) <= rolling_window:
        return m_ols(x, y)
    else:
        n = len(y) - rolling_window +1
        coe_ls = m_ols(x[:rolling_window,:], y[:rolling_window])
        for i in range(1, n):
            coe_ls = np.r_[coe_ls, m_ols(x[: rolling_window+i, :], y[: rolling_window+i])]
        coe_arr = np.reshape(coe_ls, (n, x.shape[1]+1))
        return coe_arr

def reg_func(ret_array, trading_date, rolling_window=30, method="TM"):
    r = ret_array[:,0]
    r_f = ret_array[:,1]
    r_m = ret_array[:, 2]
    y = r - r_f

    if method == "TM":
        x = np.c_[r_m-r_f, np.square(r_m-r_f)]
    
    dummy = np.zeros((len(y)))
    dummy[r_m-r_f > 0] = 1
    if method == "HM":
        x = np.c_[r_m-r_f, dummy*(r_m-r_f)]
    elif method == "CL":
        x = np.c_[dummy*(r_m-r_f), (1-dummy)*(r_m-r_f)]

    if len(trading_date) > rolling_window:
        result = cal_rolling_coe(x, y, rolling_window)
        index = trading_date[rolling_window-1: ]
        df = pd.DataFrame(result, columns=["alpha", "beta_1", "beta_2"], index=index)
        return df
    else:
        return np.nan

def _reg_func(ret_array, rolling_window=30, method="TM"):
    r = ret_array[:,0]
    r_f = ret_array[:,1]
    r_m = ret_array[:, 2]
    y = r - r_f

    if method == "TM":
        x = np.c_[r_m-r_f, np.square(r_m-r_f)]
    
    dummy = np.zeros((len(y)))
    dummy[r_m-r_f > 0] = 1
    if method == "HM":
        x = np.c_[r_m-r_f, dummy*(r_m-r_f)]
    elif method == "CL":
        x = np.c_[dummy*(r_m-r_f), (1-dummy)*(r_m-r_f)]

    if len(ret_array) > rolling_window:
        return cal_rolling_coe(x, y, rolling_window)
    else:
        return np.nan

def method_eva(ret, begin_date, end_date, rolling_window=[30, 60, 120], method="TM", ty=1):
    ret = ret.dropna()
    ret = ret[(ret['TradingDay']>=begin_date) & (ret['TradingDay']<=end_date)]
    result = {}
    if ty == 1:
        for rw in rolling_window:
            reg_result = _reg_func(np.array(ret.iloc[:, 1:4]), rw, method)
            if type(reg_result) != float:
                result[rw] = np.mean(reg_result, 0)
    elif ty == 2:
        for rw in rolling_window:
            reg_result = _reg_func(np.array(ret.iloc[:, [1,2,4]]), rw, method)
            if type(reg_result) != float:
                result[rw] = np.mean(reg_result, 0)       
    # alpha = [_res[0] for _res in result.values()]
    # if method == "CL":
    #     beta = [(_res[1]-_res[2]) for _res in result.values()]
    # else:
    #     beta = [_res[2] for _res in result.values()]
    # return (alpha, beta, np.mean(alpha), np.mean(beta))
    return(result)


