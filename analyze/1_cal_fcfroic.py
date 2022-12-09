import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from statsmodels.regression.rolling import RollingOLS
import wrds

# https://wrds-www.wharton.upenn.edu/data-dictionary/factsamp_all/wrds_fund_qf_int/

def fillna(df, col_ls):
    df[col_ls] = df[col_ls].fillna(0)
    return df


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

PATH = "../data/raw_1203.xlsx"
raw = pd.read_excel(PATH)


#%% Calculate Free Cash Flow.
# Calculate Free Cash Flow.
# For companies which do not have Capital Expenditures: nan --> 0
raw["FCF"] = raw['Operating Activities - Net Cash Flow'] - raw['Capital Expenditures']

raw['Capital Expenditures'] = raw['Capital Expenditures'].fillna(0)
raw["FCF_fillna"] = raw['Operating Activities - Net Cash Flow'] - \
    raw['Capital Expenditures']
raw["FCF_diff"] = raw["FCF_fillna"].diff()
raw["FCF_annualized"] = raw["FCF_fillna"]*4


# Calculate invested capital
col_ls = ["Debt in Current Liabilities", "Long-Term Debt - Total", 'Common/Ordinary Equity - Total', \
    'Preferred Stock At Carrying Value - Utility', 'Cash and Short-Term Investments']
raw = fillna(raw, col_ls)

raw["IC_cal"] = raw["Debt in Current Liabilities"] + \
    raw["Long-Term Debt - Total"] + raw['Common/Ordinary Equity - Total'] + raw["Preferred Stock At Carrying Value - Utility"]- \
    raw['Cash and Short-Term Investments']
raw["IC_cal"] = raw["IC_cal"].replace(0.0, np.nan)

# Calculate FCFROIC
raw["pre_IC_cal"] = raw["IC_cal"].shift()
raw["avg_IC_cal"] = (raw["IC_cal"] + raw["pre_IC_cal"])/2
raw["avg_IC_cal"] = raw["avg_IC_cal"].replace(0.0, np.nan)

raw["FCFROIC_cal"] = raw["FCF_annualized"]/raw["avg_IC_cal"]  

# For those finance service companies
raw["ave_equity"] = (raw["Stockholders Equity - Total"] +
                     raw["Stockholders Equity - Total"].shift()) * 1/2
raw["ave_asset"] = (raw['Assets - Total'] +
                    raw['Assets - Total'].shift()) * 1/2

raw["ROA"] = raw["Net Income (Loss)"]/raw["ave_asset"]
raw["ROE"] = raw["Net Income (Loss)"]/raw["ave_equity"]

raw['SIC'] = raw['Standard Industry Classification Code'].astype(int)
raw["FCFROIC_ADJ"] = raw["FCFROIC_cal"].copy()
raw.loc[(raw['SIC'] >= 6000) & (raw["SIC"] <= 6411), "FCFROIC_ADJ"] = raw.loc[(
    raw['SIC'] >= 6000) & (raw["SIC"] <= 6411), "ROE"]


#%%

grouped = raw.groupby('Global Company Key')
FCFROIC_ADJ = pd.DataFrame(columns=["Global Company Key", "Data Date",
                                    "Fiscal Year", "Fiscal Quarter", 'Ticker Symbol', "FCFROIC",
                                    "ROA", "ROE", "FCFROIC_cal", "avg_fcfroic", "avg_fcfroic_expanding"])
ls = []
for group in grouped:
    group = group[1].iloc[1:][["Global Company Key", "Data Date",
                               "Fiscal Year", "Fiscal Quarter", 'Ticker Symbol', "FCFROIC_cal"]].copy()
    group["avg_fcfroic"] = group[["FCFROIC_cal"]].rolling(
        40).mean().fillna(method="ffill")
    group["avg_fcfroic_expanding"] = group[["FCFROIC_cal"]].expanding(
        40).mean().fillna(method="ffill")
    ls.append(group)
    # FCFROIC_ADJ = FCFROIC_ADJ.append(group, ignore_index=True)
FCFROIC_ADJ = pd.concat(ls, ignore_index=True)

FCFROIC_ADJ[["avg_fcfroic", "avg_fcfroic_expanding"]].describe()

