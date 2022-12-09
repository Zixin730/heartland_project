import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from statsmodels.regression.rolling import RollingOLS
import os
import wrds

# dictionary: Keys: fiscal quarter; Values: list of tickers
low_q_dict = {}
high_q_dict = {}
missing_dict = {}
# Function gives list of index members, find which are in the high bucket, which are in the low bucket and which are not in the database


# Giving quarterly bucket, stock monthly retrns, calculate portfolio cumulative quarterly returns. 
# in-sample/out-of-sample
# weight: value weighted/equal weighted


# Giving quarterly bucket, stock monthly retrns, calculate their n year returns(use cumulative monthly returns). 
# In sample returns/Out of sample returns: calculatte the returns hold till today/calculate the next n year returns
# n: year of holding
# weight: value weighted/equal weighted

