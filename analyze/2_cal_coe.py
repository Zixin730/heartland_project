import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from statsmodels.regression.rolling import RollingOLS
import os


def cal_beta(ret_df, mkt_ret, group_col="Ticker", excess=True, cal_window=60):
    # close_price_u = stk_excess_ret[[
    # "Ticker", "excess_ret", "Monthly Price Date"]].copy()
    # close_price_u = close_price_u.drop_duplicates(
    #     subset=["Monthly Price Date", "Ticker"], keep="first")
    beta_df = pd.DataFrame()
    ls = []

    cp_grouped = ret_df.groupby(group_col)
    for group in cp_grouped:
        name = group[0]
        y = group[1].copy()
        xy = y.merge(mkt_ret, left_index=True, right_index=True, how="inner")
        if xy.shape[0] > cal_window:
            rols = RollingOLS(xy.iloc[:, 0], xy.iloc[:, 1], window=cal_window).fit()
            params = rols.params
            params.columns = ["beta"]
            params[group_col] = name
            ls.append(params)
    beta_df = pd.concat(ls)
    return beta_df

def df_to_panel(df, group_col, variable_name):
    grouped = df.groupby(group_col)
    date_ls = df.index.drop_duplicates().sort_values()
    panel = pd.DataFrame(index=date_ls)
    for t, group in grouped:
        group = group[[variable_name]]
        group.columns = [t]
        panel = panel.merge(
            group, left_index=True, right_index=True, how="left")
    return panel

def cal_panel_avg(panel, rolling_window, expanding=True):
    if expanding:
        df = panel.rolling(rolling_window, min_periods=1).apply(lambda x : np.nanmean(x))
    else:
        df = panel.rolling(rolling_window).apply(lambda x : np.nanmean(x))
    return df


#%%
# Read data
ROOTPATH = os.getcwd()
print(ROOTPATH)

# Stocks close price
print("Read stock data...")
close_price = pd.read_excel(ROOTPATH+"/data/stock_returns.xlsx")  
close_price.index = close_price["Monthly Price Date"]

# Risk free rate data
print("Read rf data...")
rf = pd.read_excel(ROOTPATH+"/data/US 10 year yields.xlsx", sheet_name=1)
rf["rf"] = rf["GT10 Govt"]/100

# Market return data
print("Read market return data...")
R3000 = pd.read_excel(ROOTPATH+"/data/Russell 3000 Price.xlsx", sheet_name=0)
R3000["return"] = R3000["RAY Index"].pct_change()

# Annualize return data
close_price["ann_ret_s"] = close_price["Monthly Total Return"] * 12
R3000["ann_ret"] = R3000["return"] * 12

# Calculate market excess return
mkt_all = R3000.merge(rf, left_on="DATES", right_on="DATES")
mkt_all["mkt_excess"] = mkt_all["ann_ret"] - mkt_all["rf"]
mkt = mkt_all[["DATES", "mkt_excess"]]
mkt.index = mkt.DATES

close_all = close_price.merge(rf, left_index=True, right_on="DATES")
close_all["excess_ret"] = close_all["ann_ret_s"] - close_all["rf"]
stk_excess_ret = close_all[["Ticker", "Monthly Price Date", "excess_ret"]]
stk_excess_ret.index = stk_excess_ret["Monthly Price Date"]


#%%
close_price_u = stk_excess_ret[[
    "Ticker", "excess_ret", "Monthly Price Date"]].copy()
close_price_u = close_price_u.drop_duplicates(
    subset=["Monthly Price Date", "Ticker"], keep="first")

ECoC = cal_beta(close_price_u, mkt["mkt_excess"], group_col="Ticker", excess=True, cal_window=60)
beta_panel = df_to_panel(df=ECoC, group_col="Ticker", variable_name="beta")

ECoC.to_excel("./clean_data/beta_1208_dropna.xlsx")
beta_panel.to_excel("./clean_data/beta_panel_1208.xlsx")

ECoC = ECoC.merge(rf, left_index=True, right_on="DATES")
ECoC["coe"] = ECoC["beta"] * 0.06 + ECoC["rf"]
ECoC.to_excel("./clean_data/ECoC_1208_dropna.xlsx")

grouped_ECoC = ECoC.groupby("name")
# ECoC.index = ECoC["DATES"]
ecoc_panel = df_to_panel(df=ECoC, group_col="Ticker", variable_name="coe")

