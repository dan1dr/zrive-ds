## Module 5: Analyse, diagnose and improve a model​

In the excercise of this week you will be working with financial data in order to (hopefully) find a portfolio of equities which outperform SP500. The data that you are gonna work with has two main sources: 
* Financial data from the companies extracted from the quarterly company reports (mostly extracted from [macrotrends](https://www.macrotrends.net/) so you can use this website to understand better the data and get insights on the features, for example [this](https://www.macrotrends.net/stocks/charts/AAPL/apple/revenue) is the one corresponding to APPLE)
* Stock prices, mostly extracted from [morningstar](https://indexes.morningstar.com/page/morningstar-indexes-empowering-investor-success?utm_source=google&utm_medium=cpc&utm_campaign=MORNI%3AG%3ASearch%3ABrand%3ACore%3AUK%20MORNI%3ABrand%3ACore%3ABroad&utm_content=engine%3Agoogle%7Ccampaignid%3A18471962329%7Cadid%3A625249340069&utm_term=morningstar%20index&gclid=CjwKCAjws9ipBhB1EiwAccEi1Fu6i20XHVcxFxuSEtJGF0If-kq5-uKnZ3rov3eRkXXFfI5j8QBtBBoCayEQAvD_BwE), which basically tell us how the stock price is evolving so we can use it both as past features and the target to predict).

Before going to the problem that we want to solve, let's comment some of the columns of the dataset:


* `Ticker`: a [short name](https://en.wikipedia.org/wiki/Ticker_symbol) to identify the equity (that you can use to search in macrotrends)
* `date`: the date of the company report (normally we are gonna have 1 every quarter). This is for informative purposes but you can ignore it when modeling.
* `execution date`: the date when we would had executed the algorithm for that equity. We want to execute the algorithm once per quarter to create the portfolio, but the release `date`s of all the different company reports don't always match for the quarter, so we just take a common `execution_date` for all of them.
* `stock_change_div_365`: what is the % change of the stock price (with dividens) in the FOLLOWING year after `execution date`. 
* `sp500_change_365`: what is the % change of the SP500 in the FOLLOWING year after `execution date`.
* `close_0`: what is the price at the moment of `execution date`
* `stock_change__minus_120` what is the % change of the stock price in the last 120 days
* `stock_change__minus_730`: what is the % change of the stock price in the last 730 days

The rest of the features can be divided beteween financial features (the ones coming from the reports) and technical features (coming from the stock price). We leave the technical features here as a reference: 


```python
technical_features = [
    "close_0",
    "close_sp500_0",
    "close_365",
    "close_sp500_365",
    "close__minus_120",
    "close_sp500__minus_120",
    "close__minus_365",
    "close_sp500__minus_365",
    "close__minus_730",
    "close_sp500__minus_730",
    "stock_change_365",
    "stock_change_div_365",
    "sp500_change_365",
    "stock_change__minus_120",
    "sp500_change__minus_120",
    "stock_change__minus_365",
    "sp500_change__minus_365",
    "stock_change__minus_730",
    "sp500_change__minus_730",
    "std__minus_365",
    "std__minus_730",
    "std__minus_120",
]
```

The problem that we want to solve is basically find a portfolio of `top_n` tickers (initially set to 10) to invest every `execution date` (basically once per quarter) and the goal is to have a better return than `SP500` in the following year. The initial way to model this is to have a binary target which is 1 when `stock_change_div_365` - `sp500_change_365` (the difference between the return of the equity and the SP500 in the following year) is positive or 0 otherwise. So we try to predict the probability of an equity of improving SP500 in the following year, we take the `top_n` equities and compute their final return.


```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.colors
from plotnine import (
    ggplot,
    geom_histogram,
    aes,
    geom_col,
    coord_flip,
    geom_bar,
    scale_x_discrete,
    geom_point,
    theme,
    element_text,
    geom_boxplot,
    scale_color_manual,
    scale_fill_gradient,
    coord_flip,
    scale_fill_brewer,
    labs,
    guides,
    scale_fill_manual,
)
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from typing import Tuple, Optional, Dict, List
from sklearn.inspection import permutation_importance
from datetime import datetime
```


```python
# number of trees in lightgbm
n_trees = 40
minimum_number_of_tickers = 1500
# Number of the quarters in the past to train
n_train_quarters = 36
# number of tickers to make the portfolio
top_n = 10
```


```python
data_set = pd.read_feather("/home/dan1dr/data/financials_against_return.feather")
```


```python
print(data_set.shape)
data_set.head()
```

    (170483, 144)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>date</th>
      <th>AssetTurnover</th>
      <th>CashFlowFromFinancialActivities</th>
      <th>CashFlowFromInvestingActivities</th>
      <th>CashFlowFromOperatingActivities</th>
      <th>CashOnHand</th>
      <th>ChangeInAccountsPayable</th>
      <th>ChangeInAccountsReceivable</th>
      <th>ChangeInAssetsLiabilities</th>
      <th>...</th>
      <th>EBIT_change_2_years</th>
      <th>Revenue_change_1_years</th>
      <th>Revenue_change_2_years</th>
      <th>NetCashFlow_change_1_years</th>
      <th>NetCashFlow_change_2_years</th>
      <th>CurrentRatio_change_1_years</th>
      <th>CurrentRatio_change_2_years</th>
      <th>Market_cap__minus_365</th>
      <th>Market_cap__minus_730</th>
      <th>diff_ch_sp500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>2005-01-31</td>
      <td>0.1695</td>
      <td>81.000</td>
      <td>-57.000</td>
      <td>137.000</td>
      <td>2483.0000</td>
      <td>5.000</td>
      <td>44.000</td>
      <td>-5.000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.304773</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NDSN</td>
      <td>2005-01-31</td>
      <td>0.2248</td>
      <td>-3.366</td>
      <td>10.663</td>
      <td>7.700</td>
      <td>62.6220</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-21.145</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.387846</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HURC</td>
      <td>2005-01-31</td>
      <td>0.3782</td>
      <td>0.483</td>
      <td>-0.400</td>
      <td>2.866</td>
      <td>11.3030</td>
      <td>0.156</td>
      <td>0.854</td>
      <td>-0.027</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.543440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NRT</td>
      <td>2005-01-31</td>
      <td>1.0517</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.9015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.331322</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HRL</td>
      <td>2005-01-31</td>
      <td>0.4880</td>
      <td>-12.075</td>
      <td>-113.077</td>
      <td>83.476</td>
      <td>145.2050</td>
      <td>NaN</td>
      <td>17.084</td>
      <td>3.539</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.218482</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 144 columns</p>
</div>




```python
data_set[data_set["Ticker"] == "AAPL"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>date</th>
      <th>AssetTurnover</th>
      <th>CashFlowFromFinancialActivities</th>
      <th>CashFlowFromInvestingActivities</th>
      <th>CashFlowFromOperatingActivities</th>
      <th>CashOnHand</th>
      <th>ChangeInAccountsPayable</th>
      <th>ChangeInAccountsReceivable</th>
      <th>ChangeInAssetsLiabilities</th>
      <th>...</th>
      <th>EBIT_change_2_years</th>
      <th>Revenue_change_1_years</th>
      <th>Revenue_change_2_years</th>
      <th>NetCashFlow_change_1_years</th>
      <th>NetCashFlow_change_2_years</th>
      <th>CurrentRatio_change_1_years</th>
      <th>CurrentRatio_change_2_years</th>
      <th>Market_cap__minus_365</th>
      <th>Market_cap__minus_730</th>
      <th>diff_ch_sp500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>447</th>
      <td>AAPL</td>
      <td>2005-03-31</td>
      <td>0.3207</td>
      <td>406.0</td>
      <td>-2432.0</td>
      <td>1311.0</td>
      <td>7057.0</td>
      <td>322.0</td>
      <td>-114.0</td>
      <td>187.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.489618</td>
    </tr>
    <tr>
      <th>2852</th>
      <td>AAPL</td>
      <td>2005-06-30</td>
      <td>0.3356</td>
      <td>63.0</td>
      <td>305.0</td>
      <td>472.0</td>
      <td>7526.0</td>
      <td>-243.0</td>
      <td>61.0</td>
      <td>312.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.348821</td>
    </tr>
    <tr>
      <th>5409</th>
      <td>AAPL</td>
      <td>2005-09-30</td>
      <td>0.3194</td>
      <td>74.0</td>
      <td>-429.0</td>
      <td>752.0</td>
      <td>8261.0</td>
      <td>249.0</td>
      <td>-68.0</td>
      <td>-150.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.043944</td>
    </tr>
    <tr>
      <th>10118</th>
      <td>AAPL</td>
      <td>2005-12-31</td>
      <td>0.4054</td>
      <td>283.0</td>
      <td>93.0</td>
      <td>283.0</td>
      <td>8707.0</td>
      <td>1117.0</td>
      <td>-436.0</td>
      <td>-1050.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.384008</td>
    </tr>
    <tr>
      <th>12004</th>
      <td>AAPL</td>
      <td>2006-03-31</td>
      <td>0.3133</td>
      <td>-141.0</td>
      <td>2462.0</td>
      <td>-125.0</td>
      <td>8226.0</td>
      <td>-788.0</td>
      <td>470.0</td>
      <td>-365.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.334104</td>
      <td>NaN</td>
      <td>2.430769</td>
      <td>NaN</td>
      <td>-0.057423</td>
      <td>NaN</td>
      <td>3.154658e+04</td>
      <td>NaN</td>
      <td>0.947410</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>207280</th>
      <td>AAPL</td>
      <td>2019-12-31</td>
      <td>0.2696</td>
      <td>-25407.0</td>
      <td>-13668.0</td>
      <td>30516.0</td>
      <td>107162.0</td>
      <td>-1089.0</td>
      <td>2015.0</td>
      <td>3347.0</td>
      <td>...</td>
      <td>0.029474</td>
      <td>0.023206</td>
      <td>0.119188</td>
      <td>-1.179745</td>
      <td>-1.279317</td>
      <td>0.228510</td>
      <td>0.286473</td>
      <td>9.128368e+05</td>
      <td>8.653736e+05</td>
      <td>0.393320</td>
    </tr>
    <tr>
      <th>210430</th>
      <td>AAPL</td>
      <td>2020-03-31</td>
      <td>0.1820</td>
      <td>-20940.0</td>
      <td>9013.0</td>
      <td>13311.0</td>
      <td>94051.0</td>
      <td>-12431.0</td>
      <td>5269.0</td>
      <td>4433.0</td>
      <td>...</td>
      <td>-0.007039</td>
      <td>0.036717</td>
      <td>0.083115</td>
      <td>1.616559</td>
      <td>-0.891914</td>
      <td>0.137297</td>
      <td>0.027473</td>
      <td>9.474150e+05</td>
      <td>9.382286e+05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>215365</th>
      <td>AAPL</td>
      <td>2020-06-30</td>
      <td>0.1881</td>
      <td>-19116.0</td>
      <td>-5165.0</td>
      <td>16271.0</td>
      <td>93025.0</td>
      <td>2733.0</td>
      <td>-2135.0</td>
      <td>-339.0</td>
      <td>...</td>
      <td>-0.011222</td>
      <td>0.057224</td>
      <td>0.072796</td>
      <td>-1.847968</td>
      <td>-2.277015</td>
      <td>-0.023328</td>
      <td>0.124073</td>
      <td>1.030571e+06</td>
      <td>1.119621e+06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>217685</th>
      <td>AAPL</td>
      <td>2020-09-30</td>
      <td>0.1998</td>
      <td>-21357.0</td>
      <td>5531.0</td>
      <td>20576.0</td>
      <td>90943.0</td>
      <td>6725.0</td>
      <td>1768.0</td>
      <td>-4479.0</td>
      <td>...</td>
      <td>-0.065023</td>
      <td>0.055121</td>
      <td>0.033585</td>
      <td>-1.429230</td>
      <td>-2.855441</td>
      <td>-0.114603</td>
      <td>0.203637</td>
      <td>1.355251e+06</td>
      <td>7.887174e+05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>225047</th>
      <td>AAPL</td>
      <td>2020-12-31</td>
      <td>0.3148</td>
      <td>-32249.0</td>
      <td>-8584.0</td>
      <td>38763.0</td>
      <td>76826.0</td>
      <td>21670.0</td>
      <td>-10945.0</td>
      <td>-4420.0</td>
      <td>...</td>
      <td>0.092438</td>
      <td>0.098818</td>
      <td>0.124318</td>
      <td>-0.270444</td>
      <td>-1.228356</td>
      <td>-0.272124</td>
      <td>-0.105797</td>
      <td>1.132762e+06</td>
      <td>9.128368e+05</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>64 rows × 144 columns</p>
</div>



Remove these quarters which have less than `minimum_number_of_tickers` tickers:


```python
df_quarter_lengths = (
    data_set.groupby(["execution_date"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
data_set = pd.merge(data_set, df_quarter_lengths, on=["execution_date"])
data_set = data_set[data_set["count"] >= minimum_number_of_tickers]
```


```python
data_set.shape
```




    (170483, 145)




```python
data_set.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>date</th>
      <th>AssetTurnover</th>
      <th>CashFlowFromFinancialActivities</th>
      <th>CashFlowFromInvestingActivities</th>
      <th>CashFlowFromOperatingActivities</th>
      <th>CashOnHand</th>
      <th>ChangeInAccountsPayable</th>
      <th>ChangeInAccountsReceivable</th>
      <th>ChangeInAssetsLiabilities</th>
      <th>...</th>
      <th>Revenue_change_1_years</th>
      <th>Revenue_change_2_years</th>
      <th>NetCashFlow_change_1_years</th>
      <th>NetCashFlow_change_2_years</th>
      <th>CurrentRatio_change_1_years</th>
      <th>CurrentRatio_change_2_years</th>
      <th>Market_cap__minus_365</th>
      <th>Market_cap__minus_730</th>
      <th>diff_ch_sp500</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>2005-01-31</td>
      <td>0.1695</td>
      <td>81.000</td>
      <td>-57.0000</td>
      <td>137.0000</td>
      <td>2483.0000</td>
      <td>5.000</td>
      <td>44.0000</td>
      <td>-5.0000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.304773</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NDSN</td>
      <td>2005-01-31</td>
      <td>0.2248</td>
      <td>-3.366</td>
      <td>10.6630</td>
      <td>7.7000</td>
      <td>62.6220</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-21.1450</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.387846</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HURC</td>
      <td>2005-01-31</td>
      <td>0.3782</td>
      <td>0.483</td>
      <td>-0.4000</td>
      <td>2.8660</td>
      <td>11.3030</td>
      <td>0.156</td>
      <td>0.8540</td>
      <td>-0.0270</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.543440</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NRT</td>
      <td>2005-01-31</td>
      <td>1.0517</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.9015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.331322</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HRL</td>
      <td>2005-01-31</td>
      <td>0.4880</td>
      <td>-12.075</td>
      <td>-113.0770</td>
      <td>83.4760</td>
      <td>145.2050</td>
      <td>NaN</td>
      <td>17.0840</td>
      <td>3.5390</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.218482</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HRB</td>
      <td>2005-01-31</td>
      <td>0.1687</td>
      <td>1122.307</td>
      <td>-54.0290</td>
      <td>-1564.4330</td>
      <td>1111.4640</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>-4.5610</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.231248</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NTAP</td>
      <td>2005-01-31</td>
      <td>0.1839</td>
      <td>20.424</td>
      <td>-277.7090</td>
      <td>344.6350</td>
      <td>1106.4340</td>
      <td>15.355</td>
      <td>-40.0650</td>
      <td>113.5450</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.182464</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HPQ</td>
      <td>2005-01-31</td>
      <td>0.2855</td>
      <td>-508.000</td>
      <td>-411.0000</td>
      <td>1558.0000</td>
      <td>13600.0000</td>
      <td>-1143.000</td>
      <td>1598.0000</td>
      <td>-374.0000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.311283</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HOV</td>
      <td>2005-01-31</td>
      <td>0.3181</td>
      <td>127.135</td>
      <td>-19.5990</td>
      <td>-92.2810</td>
      <td>93.2790</td>
      <td>-35.960</td>
      <td>52.6650</td>
      <td>-26.3790</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.604859</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NVDA</td>
      <td>2005-01-31</td>
      <td>0.3478</td>
      <td>13.843</td>
      <td>-151.9530</td>
      <td>132.2000</td>
      <td>670.0450</td>
      <td>52.941</td>
      <td>-110.3120</td>
      <td>-7.0270</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.527354</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NX</td>
      <td>2005-01-31</td>
      <td>0.4091</td>
      <td>171.155</td>
      <td>-195.1320</td>
      <td>10.4150</td>
      <td>28.1910</td>
      <td>-12.411</td>
      <td>-10.0520</td>
      <td>-2.9280</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.165162</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>11</th>
      <td>HEI.A</td>
      <td>2005-01-31</td>
      <td>0.1496</td>
      <td>12.684</td>
      <td>-12.3380</td>
      <td>3.9570</td>
      <td>4.5170</td>
      <td>-2.836</td>
      <td>2.8930</td>
      <td>-0.4280</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.251519</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>12</th>
      <td>HEI</td>
      <td>2005-01-31</td>
      <td>0.1496</td>
      <td>12.684</td>
      <td>-12.3380</td>
      <td>3.9570</td>
      <td>4.5170</td>
      <td>-2.836</td>
      <td>2.8930</td>
      <td>-0.4280</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.147588</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HD</td>
      <td>2005-01-31</td>
      <td>0.4309</td>
      <td>-2783.000</td>
      <td>-4479.0000</td>
      <td>6632.0000</td>
      <td>2165.0000</td>
      <td>0.000</td>
      <td>-266.0000</td>
      <td>336.0000</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.133304</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>14</th>
      <td>OCC</td>
      <td>2005-01-31</td>
      <td>0.3521</td>
      <td>0.000</td>
      <td>-0.5616</td>
      <td>-0.2618</td>
      <td>3.5185</td>
      <td>NaN</td>
      <td>0.6006</td>
      <td>-0.0894</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.260866</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MOV</td>
      <td>2005-01-31</td>
      <td>0.2515</td>
      <td>2.737</td>
      <td>-59.4720</td>
      <td>31.0140</td>
      <td>63.7820</td>
      <td>11.248</td>
      <td>1.4220</td>
      <td>6.5350</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.160486</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ODC</td>
      <td>2005-01-31</td>
      <td>0.3892</td>
      <td>-4.683</td>
      <td>-2.4050</td>
      <td>5.6590</td>
      <td>20.0710</td>
      <td>0.021</td>
      <td>-0.6140</td>
      <td>0.1760</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.054700</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>17</th>
      <td>JW.B</td>
      <td>2005-01-31</td>
      <td>0.2363</td>
      <td>-55.181</td>
      <td>-90.9300</td>
      <td>188.8350</td>
      <td>139.8410</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>9.8320</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.231372</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HIBB</td>
      <td>2005-01-31</td>
      <td>0.5298</td>
      <td>-17.118</td>
      <td>-12.6260</td>
      <td>46.1230</td>
      <td>58.3420</td>
      <td>12.212</td>
      <td>-1.2630</td>
      <td>3.6810</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.118799</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>19</th>
      <td>IMBI</td>
      <td>2005-01-31</td>
      <td>0.4667</td>
      <td>1.981</td>
      <td>-2.3040</td>
      <td>-18.0700</td>
      <td>100.5810</td>
      <td>0.000</td>
      <td>-8.2390</td>
      <td>-3.4170</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.147808</td>
      <td>1962</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 145 columns</p>
</div>



Create the target:


```python
data_set["diff_ch_sp500"] = (
    data_set["stock_change_div_365"] - data_set["sp500_change_365"]
)

data_set.loc[data_set["diff_ch_sp500"] > 0, "target"] = 1
data_set.loc[data_set["diff_ch_sp500"] < 0, "target"] = 0

data_set["target"].value_counts()
```




    target
    0.0    82437
    1.0    73829
    Name: count, dtype: int64



This function computes the main metric that we want to optimize: given a prediction where we have probabilities for each equity, we sort the equities in descending order of probability, we pick the `top_n` ones, and we we weight the returned `diff_ch_sp500` by the probability:


```python
def get_weighted_performance_of_stocks(df, metric):
    df["norm_prob"] = 1 / len(df)
    return np.sum(df["norm_prob"] * df[metric])


def get_top_tickers_per_prob(preds):
    if len(preds) == len(train_set):
        data_set = train_set.copy()
    elif len(preds) == len(test_set):
        data_set = test_set.copy()
    else:
        assert "Not matching train/test"
    data_set["prob"] = preds
    data_set = data_set.sort_values(["prob"], ascending=False)
    data_set = data_set.head(top_n)
    return data_set


# main metric to evaluate: average diff_ch_sp500 of the top_n stocks
def top_wt_performance(preds, train_data):
    top_dataset = get_top_tickers_per_prob(preds)
    return (
        "weighted-return",
        get_weighted_performance_of_stocks(top_dataset, "diff_ch_sp500"),
        True,
    )
```

We have created for you a function to make the `train` and `test` split based on a `execution_date`:


```python
def split_train_test_by_period(
    data_set, test_execution_date, include_nulls_in_test=False
):
    # we train with everything happening at least one year before the test execution date
    train_set = data_set.loc[
        data_set["execution_date"]
        <= pd.to_datetime(test_execution_date) - pd.Timedelta(350, unit="day")
    ]
    # remove those rows where the target is null
    train_set = train_set[~pd.isna(train_set["diff_ch_sp500"])]
    execution_dates = train_set.sort_values("execution_date")["execution_date"].unique()
    # Pick only the last n_train_quarters
    if n_train_quarters != None:
        train_set = train_set[
            train_set["execution_date"].isin(execution_dates[-n_train_quarters:])
        ]

    # the test set are the rows happening in the execution date with the concrete frequency
    test_set = data_set.loc[(data_set["execution_date"] == test_execution_date)]
    if not include_nulls_in_test:
        test_set = test_set[~pd.isna(test_set["diff_ch_sp500"])]
    test_set = test_set.sort_values("date", ascending=False).drop_duplicates(
        "Ticker", keep="first"
    )

    return train_set, test_set
```

Ensure that we don't include features which are irrelevant or related to the target:


```python
def get_columns_to_remove():
    columns_to_remove = [
        "date",
        "improve_sp500",
        "Ticker",
        "freq",
        "set",
        "close_sp500_365",
        "close_365",
        "stock_change_365",
        "sp500_change_365",
        "stock_change_div_365",
        "stock_change_730",
        "sp500_change_365",
        "stock_change_div_730",
        "diff_ch_sp500",
        "diff_ch_avg_500",
        "execution_date",
        "target",
        "index",
        "quarter",
        "std_730",
        "count",
    ]

    return columns_to_remove
```

This is the main modeling function, it receives a train test and a test set and trains a `lightgbm` in classification mode. We don't recommend to change the main algorithm for this excercise but we suggest to play with its hyperparameters:


```python
import warnings

warnings.filterwarnings("ignore")


def train_model(train_set, test_set, learning_rate, path_smooth, n_estimators=300):
    columns_to_remove = get_columns_to_remove()

    X_train = train_set.drop(columns=columns_to_remove, errors="ignore")
    X_test = test_set.drop(columns=columns_to_remove, errors="ignore")

    y_train = train_set["target"]
    y_test = test_set["target"]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    eval_result = {}

    objective = "binary"
    metric = "binary_logloss"
    params = {
        "random_state": 1,
        "verbosity": -1,
        "n_jobs": 10,
        "path_smooth": path_smooth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "objective": objective,
        "metric": metric,
    }

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_test, lgb_train],
        feval=[top_wt_performance],
        callbacks=[lgb.record_evaluation(eval_result=eval_result)],
    )
    return model, eval_result, X_train, X_test
```

This is the function which receives an `execution_date` and splits the dataset between train and test, trains the models and evaluates the model in test. It returns a dictionary with the different evaluation metrics in train and test:


```python
def run_model_for_execution_date(
    execution_date,
    all_results,
    all_predicted_tickers_list,
    all_models,
    n_estimators,
    learning_rate=0.1,
    path_smooth=0,
    include_nulls_in_test=False,
):
    global train_set
    global test_set
    # split the dataset between train and test
    train_set, test_set = split_train_test_by_period(
        data_set, execution_date, include_nulls_in_test=include_nulls_in_test
    )
    train_size, _ = train_set.shape
    test_size, _ = test_set.shape
    model = None
    X_train = None
    X_test = None

    # if both train and test are not empty
    if train_size > 0 and test_size > 0:
        model, evals_result, X_train, X_test = train_model(
            train_set, test_set, learning_rate, path_smooth, n_estimators
        )

        test_set["prob"] = model.predict(X_test)
        predicted_tickers = test_set.sort_values("prob", ascending=False)
        predicted_tickers["execution_date"] = execution_date
        all_results[(execution_date)] = evals_result
        all_models[(execution_date)] = model
        all_predicted_tickers_list.append(predicted_tickers)
    return all_results, all_predicted_tickers_list, all_models, model, X_train, X_test


execution_dates = np.sort(data_set["execution_date"].unique())
```

This is the main training loop: it goes through each different `execution_date` and calls `run_model_for_execution_date`. All the results are stored in `all_results` and the predictions in `all_predicted_tickers_list`.


```python
all_results = {}
all_predicted_tickers_list = []
all_models = {}

for execution_date in execution_dates:
    print(execution_date)
    (
        all_results,
        all_predicted_tickers_list,
        all_models,
        model,
        X_train,
        X_test,
    ) = run_model_for_execution_date(
        execution_date,
        all_results,
        all_predicted_tickers_list,
        all_models,
        n_trees,
        include_nulls_in_test=False,
    )
all_predicted_tickers = pd.concat(all_predicted_tickers_list)
```

    2005-06-30T00:00:00.000000000
    2005-09-30T00:00:00.000000000
    2005-12-30T00:00:00.000000000
    2006-03-31T00:00:00.000000000
    2006-06-30T00:00:00.000000000
    2006-09-30T00:00:00.000000000
    2006-12-30T00:00:00.000000000
    2007-03-31T00:00:00.000000000
    2007-06-30T00:00:00.000000000
    2007-09-30T00:00:00.000000000
    2007-12-30T00:00:00.000000000
    2008-03-31T00:00:00.000000000
    2008-06-30T00:00:00.000000000
    2008-09-30T00:00:00.000000000
    2008-12-30T00:00:00.000000000
    2009-03-31T00:00:00.000000000
    2009-06-30T00:00:00.000000000
    2009-09-30T00:00:00.000000000
    2009-12-30T00:00:00.000000000
    2010-03-31T00:00:00.000000000
    2010-06-30T00:00:00.000000000
    2010-09-30T00:00:00.000000000
    2010-12-30T00:00:00.000000000
    2011-03-31T00:00:00.000000000
    2011-06-30T00:00:00.000000000
    2011-09-30T00:00:00.000000000
    2011-12-30T00:00:00.000000000
    2012-03-31T00:00:00.000000000
    2012-06-30T00:00:00.000000000
    2012-09-30T00:00:00.000000000
    2012-12-30T00:00:00.000000000
    2013-03-31T00:00:00.000000000
    2013-06-30T00:00:00.000000000
    2013-09-30T00:00:00.000000000
    2013-12-30T00:00:00.000000000
    2014-03-31T00:00:00.000000000
    2014-06-30T00:00:00.000000000
    2014-09-30T00:00:00.000000000
    2014-12-30T00:00:00.000000000
    2015-03-31T00:00:00.000000000
    2015-06-30T00:00:00.000000000
    2015-09-30T00:00:00.000000000
    2015-12-30T00:00:00.000000000
    2016-03-31T00:00:00.000000000
    2016-06-30T00:00:00.000000000
    2016-09-30T00:00:00.000000000
    2016-12-30T00:00:00.000000000
    2017-03-31T00:00:00.000000000
    2017-06-30T00:00:00.000000000
    2017-09-30T00:00:00.000000000
    2017-12-30T00:00:00.000000000
    2018-03-31T00:00:00.000000000
    2018-06-30T00:00:00.000000000
    2018-09-30T00:00:00.000000000
    2018-12-30T00:00:00.000000000
    2019-03-31T00:00:00.000000000
    2019-06-30T00:00:00.000000000
    2019-09-30T00:00:00.000000000
    2019-12-30T00:00:00.000000000
    2020-03-31T00:00:00.000000000
    2020-06-30T00:00:00.000000000
    2020-09-30T00:00:00.000000000
    2020-12-30T00:00:00.000000000
    2021-03-27T00:00:00.000000000



```python
def parse_results_into_df(set_):
    df = pd.DataFrame()
    for date in all_results:
        df_tmp = pd.DataFrame(all_results[(date)][set_])
        df_tmp["n_trees"] = list(range(len(df_tmp)))
        df_tmp["execution_date"] = date
        df = pd.concat([df, df_tmp])

    df["execution_date"] = df["execution_date"].astype(str)

    return df
```


```python
test_results = parse_results_into_df("valid_0")
train_results = parse_results_into_df("training")
```


```python
train_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>binary_logloss</th>
      <th>weighted-return</th>
      <th>n_trees</th>
      <th>execution_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.657505</td>
      <td>0.267845</td>
      <td>0</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.639193</td>
      <td>0.483940</td>
      <td>1</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.619754</td>
      <td>0.218716</td>
      <td>2</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.601840</td>
      <td>0.247316</td>
      <td>3</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.585715</td>
      <td>0.250948</td>
      <td>4</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.621253</td>
      <td>0.431394</td>
      <td>35</td>
      <td>2020-03-31</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.620597</td>
      <td>0.487939</td>
      <td>36</td>
      <td>2020-03-31</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.619887</td>
      <td>0.487939</td>
      <td>37</td>
      <td>2020-03-31</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.619180</td>
      <td>0.487939</td>
      <td>38</td>
      <td>2020-03-31</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.618439</td>
      <td>0.487939</td>
      <td>39</td>
      <td>2020-03-31</td>
    </tr>
  </tbody>
</table>
<p>2240 rows × 4 columns</p>
</div>




```python
test_results_final_tree = test_results.sort_values(
    ["execution_date", "n_trees"]
).drop_duplicates("execution_date", keep="last")
train_results_final_tree = train_results.sort_values(
    ["execution_date", "n_trees"]
).drop_duplicates("execution_date", keep="last")
```

And this are the results:


```python
(
    ggplot(test_results_final_tree)
    + geom_point(aes(x="execution_date", y="weighted-return"))
    + theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1))
)
```


    
![png](module5_files/module5_32_0.png)
    





    <Figure Size: (640 x 480)>




```python
(
    ggplot(train_results_final_tree)
    + geom_point(aes(x="execution_date", y="weighted-return"))
    + theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1))
)
```


    
![png](module5_files/module5_33_0.png)
    





    <Figure Size: (640 x 480)>



We have trained the first models for all the periods for you, but there are a lot of things which may be wrong or can be improved. Some ideas where you can start:
* Try to see if there is any kind of data leakage or suspicious features
* If the training part is very slow, try to see how you can modify it to execute faster tests
* Try to understand if the algorithm is learning correctly
* We are using a very high level metric to evaluate the algorithm so you maybe need to use some more low level ones
* Try to see if there is overfitting
* Try to see if there is a lot of noise between different trainings
* To simplify, why if you only keep the first tickers in terms of Market Cap?
* Change the number of quarters to train in the past

This function can be useful to compute the feature importance:


```python
def draw_feature_importance(model, top=15):
    fi = model.feature_importance()
    fn = model.feature_name()
    feature_importance = pd.DataFrame(
        [{"feature": fn[i], "imp": fi[i]} for i in range(len(fi))]
    )
    feature_importance = feature_importance.sort_values("imp", ascending=False).head(
        top
    )
    feature_importance = feature_importance.sort_values("imp", ascending=True)
    plot = (
        ggplot(feature_importance, aes(x="feature", y="imp"))
        + geom_col(fill="lightblue")
        + coord_flip()
        + scale_x_discrete(limits=feature_importance["feature"])
    )
    return plot
```

# Solution

The solution propose will encompass the following structure:
1. Define baseline: we will create a baseline model for performance comparison, which would be the 500 largest market cap tickers. We could even decide that    the SP500 itself might be the benchmark, but will use the same baseline as the proposed in the solution.

2. Model learning and generalization assessment: evaluate if the model is learning efectively and not overfitting. For that, we will make use of low-levels metrics to optimize the problem and achieve greater top_10 weighted return (top-level). This includes plotting learning curves, adjust hyperparameters and observe the changes.

3. Feature Importance and Data Leakage Inspection: identify key features using SHAP library and check for any potential leakage. Look for anomalies, outliers and perform some feature engineering if needed. Then, re-train the model and re-assess performance.

4. Address additional points for improvement: tackle specific issues and suggestions made in the problem statement, as well as any potential areas for improvement.

## 1. Define baseline

We will use an equal weighted baseline for our approach, i.e. select the top-n market cap and extract the avg return. Another feasible strategy it is to use a market-cap weighted baseline, which will return a number very similar to the sp500 index itself (there are some minor differences if you want to exactly replicate it); even we could just get the sp500 as our baseline as it is our main benchmark here, and it wouldn't make sense if we are not even able to beat the index.

Anyway, as a first approach, we will define the top 500 largest companies by market cap.


```python
test_results_final_tree = test_results.sort_values(
    ["execution_date", "n_trees"]
).drop_duplicates("execution_date", keep="last")
train_results_final_tree = train_results.sort_values(
    ["execution_date", "n_trees"]
).drop_duplicates("execution_date", keep="last")
```


```python
train_results_final_tree.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>binary_logloss</th>
      <th>weighted-return</th>
      <th>n_trees</th>
      <th>execution_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>0.276797</td>
      <td>0.277172</td>
      <td>39</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.393084</td>
      <td>0.236828</td>
      <td>39</td>
      <td>2006-09-30</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.446241</td>
      <td>0.288516</td>
      <td>39</td>
      <td>2006-12-30</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.479211</td>
      <td>0.209679</td>
      <td>39</td>
      <td>2007-03-31</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.496369</td>
      <td>0.222281</td>
      <td>39</td>
      <td>2007-06-30</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.set_option("display.max_columns", None)
all_predicted_tickers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>date</th>
      <th>AssetTurnover</th>
      <th>CashFlowFromFinancialActivities</th>
      <th>CashFlowFromInvestingActivities</th>
      <th>CashFlowFromOperatingActivities</th>
      <th>CashOnHand</th>
      <th>ChangeInAccountsPayable</th>
      <th>ChangeInAccountsReceivable</th>
      <th>ChangeInAssetsLiabilities</th>
      <th>ChangeInInventories</th>
      <th>CommonStockDividendsPaid</th>
      <th>CommonStockNet</th>
      <th>ComprehensiveIncome</th>
      <th>CostOfGoodsSold</th>
      <th>CurrentRatio</th>
      <th>DaysSalesInReceivables</th>
      <th>DebtIssuanceRetirementNet_minus_Total</th>
      <th>DebtEquityRatio</th>
      <th>EBIT</th>
      <th>EBITMargin</th>
      <th>EBITDA</th>
      <th>EBITDAMargin</th>
      <th>FinancialActivities_minus_Other</th>
      <th>GoodwillAndIntangibleAssets</th>
      <th>GrossMargin</th>
      <th>GrossProfit</th>
      <th>IncomeAfterTaxes</th>
      <th>IncomeFromContinuousOperations</th>
      <th>IncomeFromDiscontinuedOperations</th>
      <th>IncomeTaxes</th>
      <th>Inventory</th>
      <th>InventoryTurnoverRatio</th>
      <th>InvestingActivities_minus_Other</th>
      <th>LongTermDebt</th>
      <th>Long_minus_TermInvestments</th>
      <th>Long_minus_termDebtCapital</th>
      <th>NetAcquisitionsDivestitures</th>
      <th>NetCashFlow</th>
      <th>NetChangeInIntangibleAssets</th>
      <th>NetChangeInInvestments_minus_Total</th>
      <th>NetChangeInLong_minus_TermInvestments</th>
      <th>NetChangeInPropertyPlantAndEquipment</th>
      <th>NetChangeInShort_minus_termInvestments</th>
      <th>NetCommonEquityIssuedRepurchased</th>
      <th>NetCurrentDebt</th>
      <th>NetIncome</th>
      <th>NetIncomeLoss</th>
      <th>NetLong_minus_TermDebt</th>
      <th>NetProfitMargin</th>
      <th>NetTotalEquityIssuedRepurchased</th>
      <th>OperatingExpenses</th>
      <th>OperatingIncome</th>
      <th>OperatingMargin</th>
      <th>OtherCurrentAssets</th>
      <th>OtherIncome</th>
      <th>OtherLong_minus_TermAssets</th>
      <th>OtherNon_minus_CashItems</th>
      <th>OtherNon_minus_CurrentLiabilities</th>
      <th>OtherOperatingIncomeOrExpenses</th>
      <th>OtherShareHoldersEquity</th>
      <th>Pre_minus_PaidExpenses</th>
      <th>Pre_minus_TaxIncome</th>
      <th>Pre_minus_TaxProfitMargin</th>
      <th>PropertyPlantAndEquipment</th>
      <th>ROA_minus_ReturnOnAssets</th>
      <th>ROE_minus_ReturnOnEquity</th>
      <th>ROI_minus_ReturnOnInvestment</th>
      <th>Receivables</th>
      <th>ReceiveableTurnover</th>
      <th>ResearchAndDevelopmentExpenses</th>
      <th>RetainedEarningsAccumulatedDeficit</th>
      <th>ReturnOnTangibleEquity</th>
      <th>Revenue</th>
      <th>SGAExpenses</th>
      <th>ShareHolderEquity</th>
      <th>Stock_minus_BasedCompensation</th>
      <th>TotalAssets</th>
      <th>TotalChangeInAssetsLiabilities</th>
      <th>TotalCommonAndPreferredStockDividendsPaid</th>
      <th>TotalCurrentAssets</th>
      <th>TotalCurrentLiabilities</th>
      <th>TotalDepreciationAndAmortization_minus_CashFlow</th>
      <th>TotalLiabilities</th>
      <th>TotalLiabilitiesAndShareHoldersEquity</th>
      <th>TotalLongTermLiabilities</th>
      <th>TotalLong_minus_TermAssets</th>
      <th>TotalNon_minus_CashItems</th>
      <th>TotalNon_minus_OperatingIncomeExpense</th>
      <th>execution_date</th>
      <th>close_0</th>
      <th>close_sp500_0</th>
      <th>stock_change_365</th>
      <th>stock_change_div_365</th>
      <th>sp500_change_365</th>
      <th>stock_change_730</th>
      <th>stock_change_div_730</th>
      <th>sp500_change_730</th>
      <th>stock_change__minus_120</th>
      <th>stock_change_div__minus_120</th>
      <th>sp500_change__minus_120</th>
      <th>stock_change__minus_365</th>
      <th>stock_change_div__minus_365</th>
      <th>sp500_change__minus_365</th>
      <th>stock_change__minus_730</th>
      <th>stock_change_div__minus_730</th>
      <th>sp500_change__minus_730</th>
      <th>std_730</th>
      <th>std__minus_120</th>
      <th>std__minus_365</th>
      <th>std__minus_730</th>
      <th>Market_cap</th>
      <th>n_finan_prev_year</th>
      <th>Enterprisevalue</th>
      <th>EBITDAEV</th>
      <th>EBITEV</th>
      <th>RevenueEV</th>
      <th>CashOnHandEV</th>
      <th>PFCF</th>
      <th>PE</th>
      <th>PB</th>
      <th>RDEV</th>
      <th>WorkingCapital</th>
      <th>ROC</th>
      <th>DividendYieldLastYear</th>
      <th>EPS_minus_EarningsPerShare_change_1_years</th>
      <th>EPS_minus_EarningsPerShare_change_2_years</th>
      <th>FreeCashFlowPerShare_change_1_years</th>
      <th>FreeCashFlowPerShare_change_2_years</th>
      <th>OperatingCashFlowPerShare_change_1_years</th>
      <th>OperatingCashFlowPerShare_change_2_years</th>
      <th>EBITDA_change_1_years</th>
      <th>EBITDA_change_2_years</th>
      <th>EBIT_change_1_years</th>
      <th>EBIT_change_2_years</th>
      <th>Revenue_change_1_years</th>
      <th>Revenue_change_2_years</th>
      <th>NetCashFlow_change_1_years</th>
      <th>NetCashFlow_change_2_years</th>
      <th>CurrentRatio_change_1_years</th>
      <th>CurrentRatio_change_2_years</th>
      <th>Market_cap__minus_365</th>
      <th>Market_cap__minus_730</th>
      <th>diff_ch_sp500</th>
      <th>count</th>
      <th>target</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8051</th>
      <td>TD</td>
      <td>2006-01-31</td>
      <td>0.0105</td>
      <td>8049.80</td>
      <td>-952.120</td>
      <td>-6645.170</td>
      <td>97038.950</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5154.850</td>
      <td>NaN</td>
      <td>1871.680</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>2.4956</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4285.000</td>
      <td>8270.90</td>
      <td>NaN</td>
      <td>4231.866</td>
      <td>2008.800</td>
      <td>1977.099</td>
      <td>0.0</td>
      <td>188.540</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>730.160</td>
      <td>6191.820</td>
      <td>42595.47</td>
      <td>0.2812</td>
      <td>-701.880</td>
      <td>385.200</td>
      <td>NaN</td>
      <td>-916.13</td>
      <td>NaN</td>
      <td>-64.270</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3541.120</td>
      <td>1977.099</td>
      <td>1977.090</td>
      <td>NaN</td>
      <td>56.9069</td>
      <td>485.060</td>
      <td>3931.900</td>
      <td>2171.646</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>47248.120</td>
      <td>NaN</td>
      <td>16314.7</td>
      <td>NaN</td>
      <td>-570.76</td>
      <td>NaN</td>
      <td>2197.340</td>
      <td>63.2461</td>
      <td>NaN</td>
      <td>0.6002</td>
      <td>12.7826</td>
      <td>8.9774</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10842.760</td>
      <td>26.1504</td>
      <td>18000.780</td>
      <td>1682.290</td>
      <td>15831.36</td>
      <td>NaN</td>
      <td>329411.100</td>
      <td>-7594.730</td>
      <td>-261.38</td>
      <td>229838.800</td>
      <td>288633.300</td>
      <td>182.540</td>
      <td>313579.700</td>
      <td>329411.100</td>
      <td>24946.390</td>
      <td>98114.480</td>
      <td>182.540</td>
      <td>NaN</td>
      <td>2006-06-30</td>
      <td>25.483320</td>
      <td>1270.20438</td>
      <td>0.343890</td>
      <td>0.373124</td>
      <td>0.183549</td>
      <td>0.242549</td>
      <td>0.316127</td>
      <td>0.007713</td>
      <td>0.132029</td>
      <td>0.114763</td>
      <td>0.014908</td>
      <td>-0.124695</td>
      <td>-0.158442</td>
      <td>-0.062098</td>
      <td>-0.372490</td>
      <td>-0.436061</td>
      <td>-0.101849</td>
      <td>0.013489</td>
      <td>0.011074</td>
      <td>0.009943</td>
      <td>0.009740</td>
      <td>36619.530618</td>
      <td>4.0</td>
      <td>253160.280618</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.071104</td>
      <td>0.383310</td>
      <td>10.072458</td>
      <td>11.103843</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-58794.500</td>
      <td>NaN</td>
      <td>-0.033748</td>
      <td>0.490260</td>
      <td>NaN</td>
      <td>1.136004</td>
      <td>NaN</td>
      <td>1.163775</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.265095</td>
      <td>NaN</td>
      <td>-0.617174</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29528.262573</td>
      <td>NaN</td>
      <td>0.189576</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.955844</td>
    </tr>
    <tr>
      <th>8085</th>
      <td>RY</td>
      <td>2006-01-31</td>
      <td>0.0122</td>
      <td>13705.14</td>
      <td>-10458.820</td>
      <td>-3469.990</td>
      <td>126104.100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6160.970</td>
      <td>NaN</td>
      <td>2824.670</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>0.4011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14515.000</td>
      <td>4165.87</td>
      <td>NaN</td>
      <td>3707.380</td>
      <td>1009.550</td>
      <td>1002.690</td>
      <td>0.0</td>
      <td>284.520</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-8798.810</td>
      <td>6955.410</td>
      <td>29229.69</td>
      <td>0.2863</td>
      <td>-149.110</td>
      <td>318.860</td>
      <td>NaN</td>
      <td>-1409.76</td>
      <td>NaN</td>
      <td>-101.120</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-271.660</td>
      <td>1002.690</td>
      <td>1004.400</td>
      <td>NaN</td>
      <td>19.6705</td>
      <td>-117.400</td>
      <td>5237.970</td>
      <td>1294.080</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91927.800</td>
      <td>11.140</td>
      <td>114354.6</td>
      <td>NaN</td>
      <td>-1743.99</td>
      <td>NaN</td>
      <td>1294.070</td>
      <td>25.3867</td>
      <td>NaN</td>
      <td>0.2398</td>
      <td>5.9883</td>
      <td>4.1268</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12241.380</td>
      <td>7.6102</td>
      <td>23758.959</td>
      <td>2401.310</td>
      <td>17341.39</td>
      <td>NaN</td>
      <td>418108.000</td>
      <td>-4596.940</td>
      <td>-420.78</td>
      <td>291296.800</td>
      <td>276659.300</td>
      <td>95.980</td>
      <td>400766.600</td>
      <td>418108.000</td>
      <td>124107.300</td>
      <td>125323.400</td>
      <td>107.120</td>
      <td>NaN</td>
      <td>2006-06-30</td>
      <td>40.751731</td>
      <td>1270.20438</td>
      <td>0.306123</td>
      <td>0.345876</td>
      <td>0.183549</td>
      <td>0.108486</td>
      <td>0.196335</td>
      <td>0.007713</td>
      <td>0.028775</td>
      <td>0.019941</td>
      <td>0.014908</td>
      <td>-0.239673</td>
      <td>-0.271696</td>
      <td>-0.062098</td>
      <td>-0.458892</td>
      <td>-0.517172</td>
      <td>-0.101849</td>
      <td>0.013912</td>
      <td>0.012474</td>
      <td>0.011025</td>
      <td>0.011109</td>
      <td>53572.470030</td>
      <td>4.0</td>
      <td>328234.970030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.072384</td>
      <td>0.384188</td>
      <td>-2.567151</td>
      <td>18.111880</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14637.500</td>
      <td>NaN</td>
      <td>-0.032023</td>
      <td>-0.141221</td>
      <td>NaN</td>
      <td>0.348613</td>
      <td>NaN</td>
      <td>0.349572</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.124467</td>
      <td>NaN</td>
      <td>-0.838614</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40256.249510</td>
      <td>NaN</td>
      <td>0.162328</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.951718</td>
    </tr>
    <tr>
      <th>8177</th>
      <td>GS</td>
      <td>2006-02-28</td>
      <td>0.0137</td>
      <td>16873.00</td>
      <td>-920.000</td>
      <td>-19643.000</td>
      <td>363986.000</td>
      <td>-5282.000</td>
      <td>-2400.000</td>
      <td>-313.000</td>
      <td>-14689.000</td>
      <td>-148.0</td>
      <td>6.000</td>
      <td>15.0</td>
      <td>418.000</td>
      <td>0.7275</td>
      <td>721.3026</td>
      <td>18166.000</td>
      <td>17.8806</td>
      <td>9817.0000</td>
      <td>35.3590</td>
      <td>10692.0000</td>
      <td>NaN</td>
      <td>788.000</td>
      <td>NaN</td>
      <td>95.9935</td>
      <td>10015.000</td>
      <td>2479.000</td>
      <td>2479.000</td>
      <td>NaN</td>
      <td>1210.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>114651.000</td>
      <td>292278.00</td>
      <td>0.7986</td>
      <td>-270.000</td>
      <td>1040.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-650.000</td>
      <td>NaN</td>
      <td>-1933.000</td>
      <td>3938.000</td>
      <td>2453.000</td>
      <td>2479.000</td>
      <td>14228.000</td>
      <td>23.5119</td>
      <td>-1933.000</td>
      <td>6744.000</td>
      <td>3689.000</td>
      <td>35.3590</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18942.000</td>
      <td>343.000</td>
      <td>NaN</td>
      <td>-327.0</td>
      <td>3305.00</td>
      <td>NaN</td>
      <td>3689.000</td>
      <td>35.3590</td>
      <td>NaN</td>
      <td>0.3267</td>
      <td>9.1257</td>
      <td>1.7267</td>
      <td>83615.000</td>
      <td>0.1248</td>
      <td>396.0</td>
      <td>21416.000</td>
      <td>8.5734</td>
      <td>29266.000</td>
      <td>5740.000</td>
      <td>28915.00</td>
      <td>343.000</td>
      <td>758821.000</td>
      <td>-22684.000</td>
      <td>-148.00</td>
      <td>447601.000</td>
      <td>615255.000</td>
      <td>219.000</td>
      <td>729906.000</td>
      <td>758821.000</td>
      <td>114651.000</td>
      <td>311220.000</td>
      <td>562.000</td>
      <td>NaN</td>
      <td>2006-06-30</td>
      <td>150.430000</td>
      <td>1270.20438</td>
      <td>0.440870</td>
      <td>0.450176</td>
      <td>0.183549</td>
      <td>0.162667</td>
      <td>0.181280</td>
      <td>0.007713</td>
      <td>-0.042744</td>
      <td>-0.045071</td>
      <td>0.014908</td>
      <td>-0.321811</td>
      <td>-0.329123</td>
      <td>-0.062098</td>
      <td>-0.374061</td>
      <td>-0.388021</td>
      <td>-0.101849</td>
      <td>0.022554</td>
      <td>0.017813</td>
      <td>0.013878</td>
      <td>0.012802</td>
      <td>72702.819000</td>
      <td>4.0</td>
      <td>438622.819000</td>
      <td>0.024376</td>
      <td>0.022381</td>
      <td>0.066722</td>
      <td>0.829838</td>
      <td>-2.672012</td>
      <td>11.268165</td>
      <td>2.243627</td>
      <td>0.000903</td>
      <td>-167654.000</td>
      <td>NaN</td>
      <td>-0.007312</td>
      <td>0.135204</td>
      <td>NaN</td>
      <td>-0.166129</td>
      <td>NaN</td>
      <td>-0.226284</td>
      <td>NaN</td>
      <td>0.140845</td>
      <td>NaN</td>
      <td>0.144172</td>
      <td>NaN</td>
      <td>0.142311</td>
      <td>NaN</td>
      <td>-0.777015</td>
      <td>NaN</td>
      <td>-0.032194</td>
      <td>NaN</td>
      <td>52550.502000</td>
      <td>NaN</td>
      <td>0.266628</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.949645</td>
    </tr>
    <tr>
      <th>9460</th>
      <td>JPM</td>
      <td>2006-03-31</td>
      <td>0.0113</td>
      <td>53821.00</td>
      <td>-34501.000</td>
      <td>-19141.000</td>
      <td>513228.000</td>
      <td>NaN</td>
      <td>-125.000</td>
      <td>-9752.000</td>
      <td>-9330.000</td>
      <td>-1215.0</td>
      <td>3645.000</td>
      <td>-1017.0</td>
      <td>8243.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29062.000</td>
      <td>4.2880</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27011.000</td>
      <td>59513.00</td>
      <td>NaN</td>
      <td>15175.000</td>
      <td>3027.000</td>
      <td>3027.000</td>
      <td>54.0</td>
      <td>1537.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5368.000</td>
      <td>137513.000</td>
      <td>166905.00</td>
      <td>0.5593</td>
      <td>-663.000</td>
      <td>-690.000</td>
      <td>NaN</td>
      <td>-28470.00</td>
      <td>-20101.0</td>
      <td>NaN</td>
      <td>-8369.0</td>
      <td>-898.000</td>
      <td>26024.000</td>
      <td>3077.000</td>
      <td>3081.000</td>
      <td>3038.000</td>
      <td>21.4515</td>
      <td>-1037.000</td>
      <td>18783.000</td>
      <td>4635.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78188.000</td>
      <td>2256.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4564.000</td>
      <td>31.8182</td>
      <td>8985.000</td>
      <td>0.2377</td>
      <td>2.7941</td>
      <td>1.2312</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35892.000</td>
      <td>6.1998</td>
      <td>84132.000</td>
      <td>10185.000</td>
      <td>108337.00</td>
      <td>839.000</td>
      <td>1273282.000</td>
      <td>-25537.000</td>
      <td>-1215.00</td>
      <td>959691.000</td>
      <td>985195.000</td>
      <td>837.000</td>
      <td>1164945.000</td>
      <td>1273282.000</td>
      <td>179750.000</td>
      <td>313591.000</td>
      <td>3093.000</td>
      <td>-71.000</td>
      <td>2006-06-30</td>
      <td>42.000000</td>
      <td>1270.20438</td>
      <td>0.153571</td>
      <td>0.185952</td>
      <td>0.183549</td>
      <td>-0.183095</td>
      <td>-0.114524</td>
      <td>0.007713</td>
      <td>-0.008095</td>
      <td>-0.016190</td>
      <td>0.014908</td>
      <td>-0.159048</td>
      <td>-0.191429</td>
      <td>-0.062098</td>
      <td>-0.076905</td>
      <td>-0.141667</td>
      <td>-0.101849</td>
      <td>0.020474</td>
      <td>0.012135</td>
      <td>0.010052</td>
      <td>0.009992</td>
      <td>149973.600000</td>
      <td>4.0</td>
      <td>801690.600000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.104943</td>
      <td>0.640182</td>
      <td>-3.951268</td>
      <td>16.091954</td>
      <td>1.346391</td>
      <td>NaN</td>
      <td>-25504.000</td>
      <td>NaN</td>
      <td>-0.032381</td>
      <td>0.035714</td>
      <td>NaN</td>
      <td>0.177614</td>
      <td>NaN</td>
      <td>0.177614</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.103863</td>
      <td>NaN</td>
      <td>-1.071134</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>126085.336000</td>
      <td>NaN</td>
      <td>0.002404</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.943300</td>
    </tr>
    <tr>
      <th>9370</th>
      <td>RUSHB</td>
      <td>2006-03-31</td>
      <td>0.5860</td>
      <td>-9.08</td>
      <td>-30.061</td>
      <td>27.377</td>
      <td>121.305</td>
      <td>3.425</td>
      <td>9.754</td>
      <td>0.189</td>
      <td>-6.348</td>
      <td>NaN</td>
      <td>0.248</td>
      <td>NaN</td>
      <td>416.285</td>
      <td>1.3010</td>
      <td>9.7096</td>
      <td>-10.247</td>
      <td>1.5966</td>
      <td>90.6479</td>
      <td>4.4259</td>
      <td>112.9829</td>
      <td>NaN</td>
      <td>0.605</td>
      <td>NaN</td>
      <td>16.3893</td>
      <td>81.600</td>
      <td>11.577</td>
      <td>11.577</td>
      <td>NaN</td>
      <td>6.946</td>
      <td>355.626</td>
      <td>1.1706</td>
      <td>-0.268</td>
      <td>125.721</td>
      <td>NaN</td>
      <td>0.3044</td>
      <td>-21.986</td>
      <td>23.028</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-7.807</td>
      <td>NaN</td>
      <td>0.562</td>
      <td>-8.466</td>
      <td>11.577</td>
      <td>11.577</td>
      <td>-1.781</td>
      <td>2.3252</td>
      <td>0.562</td>
      <td>475.849</td>
      <td>22.036</td>
      <td>4.4259</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>112.591</td>
      <td>1.334</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.64</td>
      <td>18.523</td>
      <td>3.7203</td>
      <td>200.525</td>
      <td>1.3626</td>
      <td>4.0292</td>
      <td>2.8028</td>
      <td>53.714</td>
      <td>9.2692</td>
      <td>NaN</td>
      <td>122.347</td>
      <td>4.0292</td>
      <td>1960.612</td>
      <td>56.656</td>
      <td>287.33</td>
      <td>0.964</td>
      <td>849.621</td>
      <td>8.448</td>
      <td>NaN</td>
      <td>536.505</td>
      <td>412.367</td>
      <td>6.018</td>
      <td>562.291</td>
      <td>849.621</td>
      <td>149.924</td>
      <td>313.116</td>
      <td>7.352</td>
      <td>-3.513</td>
      <td>2006-06-30</td>
      <td>7.511111</td>
      <td>1270.20438</td>
      <td>0.239053</td>
      <td>0.239053</td>
      <td>0.183549</td>
      <td>-0.036095</td>
      <td>-0.036095</td>
      <td>0.007713</td>
      <td>0.035503</td>
      <td>0.035503</td>
      <td>0.014908</td>
      <td>-0.206509</td>
      <td>-0.206509</td>
      <td>-0.062098</td>
      <td>-0.233136</td>
      <td>-0.233136</td>
      <td>-0.101849</td>
      <td>0.025671</td>
      <td>0.024161</td>
      <td>0.020176</td>
      <td>0.022789</td>
      <td>423.345000</td>
      <td>4.0</td>
      <td>864.331000</td>
      <td>0.130717</td>
      <td>0.104876</td>
      <td>2.268358</td>
      <td>0.140346</td>
      <td>-26.559799</td>
      <td>8.733850</td>
      <td>1.450778</td>
      <td>NaN</td>
      <td>124.138</td>
      <td>0.279206</td>
      <td>0.000000</td>
      <td>0.560232</td>
      <td>NaN</td>
      <td>0.928744</td>
      <td>NaN</td>
      <td>1.119601</td>
      <td>NaN</td>
      <td>0.439713</td>
      <td>NaN</td>
      <td>0.518289</td>
      <td>NaN</td>
      <td>0.219156</td>
      <td>NaN</td>
      <td>1.096113</td>
      <td>NaN</td>
      <td>0.045820</td>
      <td>NaN</td>
      <td>332.501248</td>
      <td>NaN</td>
      <td>0.055505</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.940366</td>
    </tr>
  </tbody>
</table>
</div>




```python
def merge_against_benchmark(
    metrics_df: pd.DataFrame,
    all_predicted_tickers: pd.DataFrame,
    top_n_market_cap: int = 500,
) -> pd.DataFrame:
    """
    Merges a given metrics DataFrame with a baseline created from the top N tickers by market cap.

    Parameters:
    - metrics_df (pd.DataFrame): A DataFrame containing various metrics including 'execution_date'.
    - all_predicted_tickers (pd.DataFrame): A DataFrame with predicted tickers, including their
      'execution_date', 'Market_cap', and performance relative to the S&P 500 ('diff_ch_sp500').
    - top_n_market_cap (int): The number of top tickers to consider based on market cap (default is 500).

    Returns:
    - pd.DataFrame: The merged DataFrame containing both the original metrics and the baseline for comparison.
    """

    # Sorting and ranking the tickers based on market cap
    all_predicted_tickers = all_predicted_tickers.sort_values(
        ["execution_date", "Market_cap"], ascending=False
    )  # = False applies to both cols
    all_predicted_tickers["rank"] = all_predicted_tickers.groupby(
        ["execution_date"]
    ).cumcount()
    top_tickers = all_predicted_tickers[
        all_predicted_tickers["rank"] <= top_n_market_cap
    ]

    # Calculate the baseline
    baseline = (
        top_tickers.groupby(["execution_date"])["diff_ch_sp500"].mean().reset_index()
    )
    baseline = baseline.rename(columns={"diff_ch_sp500": "diff_ch_sp500_baseline"})
    baseline["execution_date"] = baseline["execution_date"].astype(str)

    metrics_df = pd.merge(metrics_df, baseline, on="execution_date")

    return metrics_df
```


```python
test_results_final_tree = merge_against_benchmark(
    test_results_final_tree, all_predicted_tickers
)
test_results_final_tree.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>binary_logloss</th>
      <th>weighted-return</th>
      <th>n_trees</th>
      <th>execution_date</th>
      <th>diff_ch_sp500_baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.759610</td>
      <td>0.099404</td>
      <td>39</td>
      <td>2006-06-30</td>
      <td>0.049213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.730303</td>
      <td>0.035528</td>
      <td>39</td>
      <td>2006-09-30</td>
      <td>0.067796</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.715078</td>
      <td>-0.052195</td>
      <td>39</td>
      <td>2006-12-30</td>
      <td>0.068473</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.710833</td>
      <td>-0.067471</td>
      <td>39</td>
      <td>2007-03-31</td>
      <td>0.048029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.711701</td>
      <td>-0.045395</td>
      <td>39</td>
      <td>2007-06-30</td>
      <td>0.077166</td>
    </tr>
  </tbody>
</table>
</div>




```python
(
    ggplot(
        test_results_final_tree[test_results_final_tree["weighted-return"] < 10],
        aes(x="execution_date"),
    )
    + geom_point(aes(y="weighted-return"), colour="blue")
    + geom_point(aes(y="diff_ch_sp500_baseline"), colour="red")
    + theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1))
)
```


    
![png](module5_files/module5_46_0.png)
    





    <Figure Size: (640 x 480)>




```python
print(f"Mean for model: {test_results_final_tree['weighted-return'].mean()}")
print(
    f"Mean for baseline: {test_results_final_tree['diff_ch_sp500_baseline'].mean()}\n"
)

print(f"Median for model: {test_results_final_tree['weighted-return'].median()}")
print(
    f"Median for baseline: {test_results_final_tree['diff_ch_sp500_baseline'].median()}\n"
)
```

    Mean for model: 4.029023035251342
    Mean for baseline: 0.022159133577893696
    
    Median for model: 0.10471844503860958
    Median for baseline: 0.015525563344158869
    


Our model does 38 pts better in average and the median is 9 points better, maybe too optimistic to be true. Let's explore what could be happening more deeply:

## 2. Model learning and generalization assessment

In machine learning, the ultimate goal for the model is to try to reduce the error function as much as possible. While this might be positive, sometimes the metric being optimized is not necessarily our best metric based on the business problem we are tackling. In this light, there exists a range of different metrics closer to the model or business, which might be interesting to keep in mind and be diligent to know in which area we are trying to optimize. We can denominate that "pyramid," referring to how close we are with respect to the model, e.g., low-level if we are closer to the model, or high-level if we abstract enough from the model and we are closer to the business.

- Low-Level Metrics (Closest to the Model):
    - Log Loss (Binary Cross-Entropy): A standard metric for binary classification models. It measures the model's ability to assign high probabilities to the correct class. It's sensitive to the confidence of the predictions. This sensitivity is crucial in probabilistic models like yours.
    - Alternatives to Log Loss:
        - Mean Squared Error (MSE): Typically used for regression but can be adapted for classification. It penalizes large errors more heavily.
        - Hinge Loss: Common in SVM models; it's more about margin maximization, less sensitive to probabilistic outputs.
- Mid-Level Metrics (Balancing Model and Business Objectives):
    - Accuracy: The proportion of correctly predicted instances. It's intuitive but can be misleading in imbalanced datasets.
    - Precision and Recall: Precision measures the correctness achieved in positive prediction, while recall measures how many actual positives were correctly identified.
    - F1-Score: The harmonic mean of precision and recall, useful when you need a balance between precision and recall.
- High-Level Metrics (Direct Business Impact):
    - Top 10 Weighted Return: Your primary metric, directly aligned with your business objective (portfolio performance).
    - Sharpe Ratio: Measures risk-adjusted return; it could be a supplementary metric for evaluating the effectiveness of the investment strategy.

For visualization, I've illustrated a basic graph the following:

<img src="images/pyramid_metrics.png" width="800" height="550">


Let's try to evaluate if the model is learning correctly plotting the curves for each tree. We want to calculate the percentage difference of the logloss metric at each tree compared to the first one, to understand if model is improving, overfitting, or underperforming as more trees are added.


```python
def compute_learning_rate(set_: str, all_results: dict) -> pd.DataFrame:
    """
    Calculate the normalized learning rates for the model

    Parameters:
    - set_ (str): The dataset type ('training' or 'valid_0').
    - all_results (dict): Dictionary containing training results for each execution date.

    Returns:
    - pd.DataFrame: DataFrame with normalized learning rates, including execution dates and tree categories.
    """

    df = pd.DataFrame()
    for date in all_results:
        df_tmp = pd.DataFrame(all_results[date][set_])
        df_tmp["n_trees"] = range(len(df_tmp))
        df_tmp["execution_date"] = date
        df = pd.concat([df, df_tmp])

    # Calculate the % diff respect to first tree
    df["first_tree_logloss"] = df.groupby(["execution_date"])[
        "binary_logloss"
    ].transform("first")
    df[f"normalized_learning_{set_}"] = (
        df["binary_logloss"] - df["first_tree_logloss"]
    ) / df["first_tree_logloss"]
    df = df.drop(columns="first_tree_logloss")

    return df
```


```python
learning_rates_train = compute_learning_rate("training", all_results)
learning_rates_test = compute_learning_rate("valid_0", all_results)

# Convert 'n_trees' to a categorical variable for better plotting
learning_rates_train["n_trees_cat_train"] = pd.Categorical(
    learning_rates_train["n_trees"],
    categories=sorted(learning_rates_train["n_trees"].unique()),
)
learning_rates_test["n_trees_cat_test"] = pd.Categorical(
    learning_rates_test["n_trees"],
    categories=sorted(learning_rates_test["n_trees"].unique()),
)

# Merging training and validation learning rates for comparison
learning_rates = pd.merge(
    learning_rates_test[
        ["n_trees", "n_trees_cat_test", "normalized_learning_valid_0", "execution_date"]
    ],
    learning_rates_train[
        [
            "n_trees",
            "n_trees_cat_train",
            "normalized_learning_training",
            "execution_date",
        ]
    ],
    on=["execution_date", "n_trees"],
)
ggplot(learning_rates, aes(x="n_trees_cat_train")) + geom_boxplot(
    aes(y="normalized_learning_training")
)
```


    
![png](module5_files/module5_54_0.png)
    





    <Figure Size: (640 x 480)>



Model learning OK for each additional n_tree passed. Let's compare in validation:


```python
ggplot(learning_rates, aes(x="n_trees_cat_test")) + geom_boxplot(
    aes(y="normalized_learning_valid_0")
)
```


    
![png](module5_files/module5_56_0.png)
    





    <Figure Size: (640 x 480)>



We see the model seems to have a slightly decreasing tren until tree 20th, since there the logloss start increasing again. Let's zoom in for only first 20 n_trees then:


```python
ggplot(
    learning_rates[learning_rates["n_trees"] <= 20], aes(x="n_trees_cat_test")
) + geom_boxplot(aes(y="normalized_learning_valid_0"))
```


    
![png](module5_files/module5_58_0.png)
    





    <Figure Size: (640 x 480)>



Minimum loss obtained around tree = 10. From that point, it again increases. That might be happening for several reason, first let's try to play a bit with hyperparameters:

- learning_rate: let's reduce it in order to make smaller steps
- n_estimators: n_trees
- path_smoth: introduces regularization term (smoother boundaries)
- num_leaves: max. leaves per tree


```python
import warnings

warnings.filterwarnings("ignore")


def train_model(train_set, test_set, params, n_estimators=300):
    columns_to_remove = get_columns_to_remove()

    X_train = train_set.drop(columns=columns_to_remove, errors="ignore")
    X_test = test_set.drop(columns=columns_to_remove, errors="ignore")

    y_train = train_set["target"]
    y_test = test_set["target"]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    eval_result = {}

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_test, lgb_train],
        feval=[top_wt_performance],
        callbacks=[lgb.record_evaluation(eval_result=eval_result)],
    )
    return model, eval_result, X_train, X_test
```


```python
params = {
    "random_state": 1,
    "verbosity": -1,
    "n_jobs": 10,
    "n_estimators": 20,
    "learning_rate": 0.05,
    "path_smooth": 0.1,
    "objective": "binary",
    "metric": "binary_logloss",
}

all_results = {}
all_predicted_tickers_list = []
all_models = {}


for execution_date in execution_dates:
    print(execution_date)
    (
        all_results,
        all_predicted_tickers_list,
        all_models,
        model,
        X_train,
        X_test,
    ) = run_model_for_execution_date(
        execution_date,
        all_results,
        all_predicted_tickers_list,
        all_models,
        n_trees,
        params,
        False,
    )
all_predicted_tickers = pd.concat(all_predicted_tickers_list)
```

    2005-06-30T00:00:00.000000000
    2005-09-30T00:00:00.000000000
    2005-12-30T00:00:00.000000000
    2006-03-31T00:00:00.000000000
    2006-06-30T00:00:00.000000000
    2006-09-30T00:00:00.000000000
    2006-12-30T00:00:00.000000000
    2007-03-31T00:00:00.000000000
    2007-06-30T00:00:00.000000000
    2007-09-30T00:00:00.000000000
    2007-12-30T00:00:00.000000000
    2008-03-31T00:00:00.000000000
    2008-06-30T00:00:00.000000000
    2008-09-30T00:00:00.000000000
    2008-12-30T00:00:00.000000000
    2009-03-31T00:00:00.000000000
    2009-06-30T00:00:00.000000000
    2009-09-30T00:00:00.000000000
    2009-12-30T00:00:00.000000000
    2010-03-31T00:00:00.000000000
    2010-06-30T00:00:00.000000000
    2010-09-30T00:00:00.000000000
    2010-12-30T00:00:00.000000000
    2011-03-31T00:00:00.000000000
    2011-06-30T00:00:00.000000000
    2011-09-30T00:00:00.000000000
    2011-12-30T00:00:00.000000000
    2012-03-31T00:00:00.000000000
    2012-06-30T00:00:00.000000000
    2012-09-30T00:00:00.000000000
    2012-12-30T00:00:00.000000000
    2013-03-31T00:00:00.000000000
    2013-06-30T00:00:00.000000000
    2013-09-30T00:00:00.000000000
    2013-12-30T00:00:00.000000000
    2014-03-31T00:00:00.000000000
    2014-06-30T00:00:00.000000000
    2014-09-30T00:00:00.000000000
    2014-12-30T00:00:00.000000000
    2015-03-31T00:00:00.000000000
    2015-06-30T00:00:00.000000000
    2015-09-30T00:00:00.000000000
    2015-12-30T00:00:00.000000000
    2016-03-31T00:00:00.000000000
    2016-06-30T00:00:00.000000000
    2016-09-30T00:00:00.000000000
    2016-12-30T00:00:00.000000000
    2017-03-31T00:00:00.000000000
    2017-06-30T00:00:00.000000000
    2017-09-30T00:00:00.000000000
    2017-12-30T00:00:00.000000000
    2018-03-31T00:00:00.000000000
    2018-06-30T00:00:00.000000000
    2018-09-30T00:00:00.000000000
    2018-12-30T00:00:00.000000000
    2019-03-31T00:00:00.000000000
    2019-06-30T00:00:00.000000000
    2019-09-30T00:00:00.000000000
    2019-12-30T00:00:00.000000000
    2020-03-31T00:00:00.000000000
    2020-06-30T00:00:00.000000000
    2020-09-30T00:00:00.000000000
    2020-12-30T00:00:00.000000000
    2021-03-27T00:00:00.000000000



```python
learning_rates_train = compute_learning_rate("training", all_results)
learning_rates_test = compute_learning_rate("valid_0", all_results)

# Filter for n_trees <= 20
learning_rates_train = learning_rates_train[learning_rates_train["n_trees"] <= 20]
learning_rates_test = learning_rates_test[learning_rates_test["n_trees"] <= 20]

# Convert 'n_trees' to a categorical variable for better plotting
learning_rates_train = learning_rates_train[
    ["n_trees", "normalized_learning_training", "execution_date"]
]
learning_rates_test = learning_rates_test[
    ["n_trees", "normalized_learning_valid_0", "execution_date"]
]

# Rename columns for merging
learning_rates_train.rename(
    columns={"normalized_learning_training": "Normalized Learning"}, inplace=True
)
learning_rates_test.rename(
    columns={"normalized_learning_valid_0": "Normalized Learning"}, inplace=True
)

learning_rates_train["Set"] = "Training"
learning_rates_test["Set"] = "Validation"

learning_rates_combined = pd.concat([learning_rates_train, learning_rates_test])


learning_rates_combined["n_trees_cat"] = pd.Categorical(
    learning_rates_combined["n_trees"],
    categories=sorted(learning_rates_combined["n_trees"].unique()),
)

# Plotting
plot = (
    ggplot(
        learning_rates_combined,
        aes(x="n_trees_cat", y="Normalized Learning", color="Set"),
    )
    + geom_boxplot()
    + scale_color_manual(values=["blue", "green"])
)

print(plot)
```


    
![png](module5_files/module5_62_0.png)
    


    


Seems to improve, let's try with an stronger regularization also:


```python
params = {
    "random_state": 1,
    "verbosity": -1,
    "n_jobs": 10,
    "n_estimators": 20,
    "learning_rate": 0.01,
    "path_smooth": 0.3,
    "objective": "binary",
    "metric": "binary_logloss",
}

all_results = {}
all_predicted_tickers_list = []
all_models = {}

for execution_date in execution_dates:
    print(execution_date)
    (
        all_results,
        all_predicted_tickers_list,
        all_models,
        model,
        X_train,
        X_test,
    ) = run_model_for_execution_date(
        execution_date,
        all_results,
        all_predicted_tickers_list,
        all_models,
        n_trees,
        params,
        False,
    )
all_predicted_tickers = pd.concat(all_predicted_tickers_list)
```

    2005-06-30T00:00:00.000000000
    2005-09-30T00:00:00.000000000
    2005-12-30T00:00:00.000000000
    2006-03-31T00:00:00.000000000
    2006-06-30T00:00:00.000000000
    2006-09-30T00:00:00.000000000
    2006-12-30T00:00:00.000000000
    2007-03-31T00:00:00.000000000
    2007-06-30T00:00:00.000000000
    2007-09-30T00:00:00.000000000
    2007-12-30T00:00:00.000000000
    2008-03-31T00:00:00.000000000
    2008-06-30T00:00:00.000000000
    2008-09-30T00:00:00.000000000
    2008-12-30T00:00:00.000000000
    2009-03-31T00:00:00.000000000
    2009-06-30T00:00:00.000000000
    2009-09-30T00:00:00.000000000
    2009-12-30T00:00:00.000000000
    2010-03-31T00:00:00.000000000
    2010-06-30T00:00:00.000000000
    2010-09-30T00:00:00.000000000
    2010-12-30T00:00:00.000000000
    2011-03-31T00:00:00.000000000
    2011-06-30T00:00:00.000000000
    2011-09-30T00:00:00.000000000
    2011-12-30T00:00:00.000000000
    2012-03-31T00:00:00.000000000
    2012-06-30T00:00:00.000000000
    2012-09-30T00:00:00.000000000
    2012-12-30T00:00:00.000000000
    2013-03-31T00:00:00.000000000
    2013-06-30T00:00:00.000000000
    2013-09-30T00:00:00.000000000
    2013-12-30T00:00:00.000000000
    2014-03-31T00:00:00.000000000
    2014-06-30T00:00:00.000000000
    2014-09-30T00:00:00.000000000
    2014-12-30T00:00:00.000000000
    2015-03-31T00:00:00.000000000
    2015-06-30T00:00:00.000000000
    2015-09-30T00:00:00.000000000
    2015-12-30T00:00:00.000000000
    2016-03-31T00:00:00.000000000
    2016-06-30T00:00:00.000000000
    2016-09-30T00:00:00.000000000
    2016-12-30T00:00:00.000000000
    2017-03-31T00:00:00.000000000
    2017-06-30T00:00:00.000000000
    2017-09-30T00:00:00.000000000
    2017-12-30T00:00:00.000000000
    2018-03-31T00:00:00.000000000
    2018-06-30T00:00:00.000000000
    2018-09-30T00:00:00.000000000
    2018-12-30T00:00:00.000000000
    2019-03-31T00:00:00.000000000
    2019-06-30T00:00:00.000000000
    2019-09-30T00:00:00.000000000
    2019-12-30T00:00:00.000000000
    2020-03-31T00:00:00.000000000
    2020-06-30T00:00:00.000000000
    2020-09-30T00:00:00.000000000
    2020-12-30T00:00:00.000000000
    2021-03-27T00:00:00.000000000



```python
learning_rates_train = compute_learning_rate("training", all_results)
learning_rates_test = compute_learning_rate("valid_0", all_results)

learning_rates_train = learning_rates_train[learning_rates_train["n_trees"] <= 20]
learning_rates_test = learning_rates_test[learning_rates_test["n_trees"] <= 20]

learning_rates_train = learning_rates_train[
    ["n_trees", "normalized_learning_training", "execution_date"]
]
learning_rates_test = learning_rates_test[
    ["n_trees", "normalized_learning_valid_0", "execution_date"]
]

learning_rates_train.rename(
    columns={"normalized_learning_training": "Normalized Learning"}, inplace=True
)
learning_rates_test.rename(
    columns={"normalized_learning_valid_0": "Normalized Learning"}, inplace=True
)

learning_rates_train["Set"] = "Training"
learning_rates_test["Set"] = "Validation"

learning_rates_combined = pd.concat([learning_rates_train, learning_rates_test])

learning_rates_combined["n_trees_cat"] = pd.Categorical(
    learning_rates_combined["n_trees"],
    categories=sorted(learning_rates_combined["n_trees"].unique()),
)

# Plotting
plot = (
    ggplot(
        learning_rates_combined,
        aes(x="n_trees_cat", y="Normalized Learning", color="Set"),
    )
    + geom_boxplot()
    + scale_color_manual(values=["blue", "red"])
)

print(plot)
```


    
![png](module5_files/module5_65_0.png)
    


    


Yep, even better. Enough fine-tuning as we do not want to overfit on training set.
Keep and eye on lambda rank and ndcg

## 3. Feature importance and Data Leakage


```python
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Optional, Dict, List
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance


def train_model(
    train_set: pd.DataFrame, test_set: pd.DataFrame, params: Dict
) -> Tuple[LGBMClassifier, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Trains the LightGBM model and computes feature importance.

    Parameters:
    - train_set (pd.DataFrame): Training dataset.
    - test_set (pd.DataFrame): Testing dataset.
    - params (Dict): Parameters for the LightGBM model.

    Returns:
    - Tuple[LGBMClassifier, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Trained model, evaluation results, training features, testing features, feature importance dataframe.
    """
    columns_to_remove = get_columns_to_remove()

    X_train = train_set.drop(columns=columns_to_remove, errors="ignore")
    X_test = test_set.drop(columns=columns_to_remove, errors="ignore")

    y_train = train_set["target"]
    y_test = test_set["target"]

    clf = LGBMClassifier(**params)
    model = clf.fit(
        X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="binary_logloss"
    )

    # Compute feature importance
    pi = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0)
    feature_names = X_test.columns
    df_feature_importance = pd.DataFrame(
        {"importance": pi.importances_mean, "feature": feature_names}
    )

    return model, X_train, X_test, df_feature_importance
```


```python
def run_model_for_execution_date(
    execution_date: str,
    all_results: Dict,
    all_predicted_tickers_list: List[pd.DataFrame],
    all_models: Dict,
    all_feature_importance: pd.DataFrame,
    params: Dict,
    include_nulls_in_test: bool = False,
) -> Tuple[
    Dict,
    List[pd.DataFrame],
    Dict,
    LGBMClassifier,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Runs the model for a specific execution date and computes feature importance.

    Parameters:
    - execution_date (str): The execution date for the model.
    - all_results (Dict): Dictionary to store all results.
    - all_predicted_tickers_list (List[pd.DataFrame]): List to store all predicted tickers.
    - all_models (Dict): Dictionary to store all models.
    - all_feature_importance (pd.DataFrame): DataFrame to store all feature importances.
    - params (Dict): Parameters for the LightGBM model.
    - include_nulls_in_test (bool): Flag to include nulls in the test set.

    Returns:
    - Tuple[Dict, List[pd.DataFrame], Dict, LGBMClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated results, predicted tickers list, models, trained model, training features, testing features, feature importance dataframe.
    """
    global train_set
    global test_set

    train_set, test_set = split_train_test_by_period(
        data_set, execution_date, include_nulls_in_test=include_nulls_in_test
    )
    train_size, _ = train_set.shape
    test_size, _ = test_set.shape

    model = None
    X_train = None
    X_test = None

    if train_size > 0 and test_size > 0:
        model, X_train, X_test, df_feature_importance = train_model(
            train_set, test_set, params
        )

        test_set["prob"] = model.predict_proba(X_test)[:, 1]
        predicted_tickers = test_set.sort_values("prob", ascending=False)
        predicted_tickers["execution_date"] = execution_date
        all_results[execution_date] = model.evals_result_
        all_models[execution_date] = model
        all_predicted_tickers_list.append(predicted_tickers)

        # Append feature importance for this execution date
        df_feature_importance["execution_date"] = execution_date
        all_feature_importance = pd.concat(
            [all_feature_importance, df_feature_importance]
        )

    return (
        all_results,
        all_predicted_tickers_list,
        all_models,
        X_train,
        X_test,
        all_feature_importance,
    )
```


```python
period_frequency = 4
execution_dates = np.sort(data_set["execution_date"].unique())

all_results = {}
all_predicted_tickers_list = []
all_models = {}
all_feature_importance = pd.DataFrame()

for i, execution_date in enumerate(execution_dates):
    if i % period_frequency == 0:
        print(f"Training for execution date: {execution_date}")
        (
            all_results,
            all_predicted_tickers_list,
            all_models,
            X_train,
            X_test,
            all_feature_importance,
        ) = run_model_for_execution_date(
            execution_date,
            all_results,
            all_predicted_tickers_list,
            all_models,
            all_feature_importance,
            params,
        )
```

    Training for execution date: 2005-06-30T00:00:00.000000000
    Training for execution date: 2006-06-30T00:00:00.000000000
    Training for execution date: 2007-06-30T00:00:00.000000000
    Training for execution date: 2008-06-30T00:00:00.000000000
    Training for execution date: 2009-06-30T00:00:00.000000000
    Training for execution date: 2010-06-30T00:00:00.000000000
    Training for execution date: 2011-06-30T00:00:00.000000000
    Training for execution date: 2012-06-30T00:00:00.000000000
    Training for execution date: 2013-06-30T00:00:00.000000000
    Training for execution date: 2014-06-30T00:00:00.000000000
    Training for execution date: 2015-06-30T00:00:00.000000000
    Training for execution date: 2016-06-30T00:00:00.000000000
    Training for execution date: 2017-06-30T00:00:00.000000000
    Training for execution date: 2018-06-30T00:00:00.000000000
    Training for execution date: 2019-06-30T00:00:00.000000000
    Training for execution date: 2020-06-30T00:00:00.000000000



```python
all_feature_importance = all_feature_importance.sort_values(
    ["execution_date", "importance"], ascending=False
)
all_feature_importance_mean = (
    all_feature_importance.groupby("feature")["importance"].mean().reset_index()
)
all_feature_importance_mean = all_feature_importance_mean.sort_values(
    "importance", ascending=False
)
all_feature_importance_mean_importants = all_feature_importance_mean.head(10)
```


```python
# Convert 'feature' to a categorical type with ordered levels
all_feature_importance_mean_importants["feature"] = pd.Categorical(
    all_feature_importance_mean_importants["feature"],
    categories=all_feature_importance_mean_importants["feature"],
    ordered=True,
)

plot = (
    ggplot(
        all_feature_importance_mean_importants,
        aes(x="feature", y="importance", fill="importance"),
    )
    + geom_col()
    + scale_fill_gradient(low="lightblue", high="darkblue")
    + coord_flip()
)  # This will make the plot horizontal

print(plot)
```


    
![png](module5_files/module5_72_0.png)
    


    


Anormal importance given to close_0 (actual price). This means that my actual price will determine very strongly if I'll beat the index. We could argue that this might be somehow explained by momentum, but we saw in previous lessons that reverse stock splitting are causing an important data leakages for these cases.

Let's examine the weights for each period taken, to see how predominant are these features over time:


```python
# Get unique features
all_feature_importance["rank"] = all_feature_importance.groupby("execution_date")[
    "importance"
].rank(ascending=False, method="first")
all_feature_importance_year = all_feature_importance[
    all_feature_importance["rank"] <= 2
]
unique_features = all_feature_importance_year["feature"].unique()

# Create palette
color_palette = plt.cm.tab20(range(len(unique_features)))
hex_color_palette = [matplotlib.colors.to_hex(c) for c in color_palette]

# Map features to cols
color_map = dict(zip(unique_features, hex_color_palette))

# Create the plot
plot = (
    ggplot(
        all_feature_importance_year,
        aes(x="execution_date", y="importance", fill="feature"),
    )
    + geom_col()
    + theme(
        axis_text_x=element_text(angle=45, hjust=1),
        legend_position="right",
        figure_size=(12, 6),
    )
    + scale_fill_manual(values=color_map)
    + labs(
        title="Top Feature Importance Over Time",
        x="Execution Date",
        y="Importance",
        fill="Feature",
    )
)
print(plot)
```


    
![png](module5_files/module5_74_0.png)
    


    


`close_o` presumably very present in half of periods taken, specially in 2009 and 2012. Let's take 2009 as and example


```python
all_feature_importance_year[
    all_feature_importance_year["execution_date"] == "2009-06-30"
]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
      <th>feature</th>
      <th>execution_date</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87</th>
      <td>0.026532</td>
      <td>close_0</td>
      <td>2009-06-30</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.008233</td>
      <td>EBIT</td>
      <td>2009-06-30</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
stocks = all_predicted_tickers[all_predicted_tickers["execution_date"] == "2009-06-30"]
stocks.sort_values("prob", ascending=False).head(10)[["Ticker", "close_0", "prob"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>close_0</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34587</th>
      <td>EBTC</td>
      <td>10.976029</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>35124</th>
      <td>WNEB</td>
      <td>9.060000</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>34748</th>
      <td>PKBK</td>
      <td>4.273207</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>34793</th>
      <td>CAC</td>
      <td>22.686667</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>34877</th>
      <td>AROW</td>
      <td>19.690890</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>34226</th>
      <td>FLIC</td>
      <td>10.284444</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>35626</th>
      <td>PBHC</td>
      <td>3.642545</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>35820</th>
      <td>NWFL</td>
      <td>19.006060</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>35789</th>
      <td>FNLC</td>
      <td>19.470000</td>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>35834</th>
      <td>SONA</td>
      <td>8.200000</td>
      <td>0.547425</td>
    </tr>
  </tbody>
</table>
</div>




```python
stocks.sort_values("prob", ascending=False).tail(25)[["Ticker", "close_0", "prob"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>close_0</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35476</th>
      <td>TGH</td>
      <td>11.490000</td>
      <td>0.458347</td>
    </tr>
    <tr>
      <th>35148</th>
      <td>WINT</td>
      <td>13356.000000</td>
      <td>0.458231</td>
    </tr>
    <tr>
      <th>35156</th>
      <td>OLED</td>
      <td>9.780000</td>
      <td>0.458231</td>
    </tr>
    <tr>
      <th>35069</th>
      <td>HRTX</td>
      <td>18.800000</td>
      <td>0.458231</td>
    </tr>
    <tr>
      <th>35241</th>
      <td>IEC</td>
      <td>3.840000</td>
      <td>0.457656</td>
    </tr>
    <tr>
      <th>35166</th>
      <td>UHT</td>
      <td>31.520000</td>
      <td>0.457321</td>
    </tr>
    <tr>
      <th>35804</th>
      <td>HIG</td>
      <td>11.870000</td>
      <td>0.457259</td>
    </tr>
    <tr>
      <th>35217</th>
      <td>MS</td>
      <td>28.510000</td>
      <td>0.456835</td>
    </tr>
    <tr>
      <th>34849</th>
      <td>MFG</td>
      <td>4.610000</td>
      <td>0.456532</td>
    </tr>
    <tr>
      <th>34906</th>
      <td>AGM</td>
      <td>4.830000</td>
      <td>0.456270</td>
    </tr>
    <tr>
      <th>34915</th>
      <td>SITC</td>
      <td>8.056089</td>
      <td>0.455972</td>
    </tr>
    <tr>
      <th>34559</th>
      <td>BXP</td>
      <td>47.700000</td>
      <td>0.453912</td>
    </tr>
    <tr>
      <th>36109</th>
      <td>INT</td>
      <td>20.615000</td>
      <td>0.453908</td>
    </tr>
    <tr>
      <th>35733</th>
      <td>AGEN</td>
      <td>12.540000</td>
      <td>0.453680</td>
    </tr>
    <tr>
      <th>35041</th>
      <td>CLBS</td>
      <td>190.000000</td>
      <td>0.453316</td>
    </tr>
    <tr>
      <th>36032</th>
      <td>MVIS</td>
      <td>24.560000</td>
      <td>0.452988</td>
    </tr>
    <tr>
      <th>35793</th>
      <td>NLY</td>
      <td>15.140000</td>
      <td>0.452546</td>
    </tr>
    <tr>
      <th>35123</th>
      <td>NYMX</td>
      <td>5.000000</td>
      <td>0.451938</td>
    </tr>
    <tr>
      <th>36119</th>
      <td>EIG</td>
      <td>13.550000</td>
      <td>0.450245</td>
    </tr>
    <tr>
      <th>34879</th>
      <td>CGEN</td>
      <td>2.010000</td>
      <td>0.448631</td>
    </tr>
    <tr>
      <th>35339</th>
      <td>HALO</td>
      <td>6.980000</td>
      <td>0.448631</td>
    </tr>
    <tr>
      <th>35811</th>
      <td>DXCM</td>
      <td>6.190000</td>
      <td>0.448631</td>
    </tr>
    <tr>
      <th>33969</th>
      <td>CDMO</td>
      <td>29.400000</td>
      <td>0.448631</td>
    </tr>
    <tr>
      <th>33982</th>
      <td>TD</td>
      <td>25.870754</td>
      <td>0.445386</td>
    </tr>
    <tr>
      <th>35739</th>
      <td>FMBI</td>
      <td>7.310000</td>
      <td>0.443811</td>
    </tr>
  </tbody>
</table>
</div>




```python
# all_predicted_tickers[all_predicted_tickers['Ticker'] == 'TTNP'][['execution_date','close_0', 'prob']]
```

$13356 for WINT looks strange. Let's check online:

''Vestas Wind Systems stock (symbol: VWS.CO) underwent a total of 1 stock split.
The stock split occured on April 27th, 2021.
One VWS.CO share bought prior to April 27th, 2021 would equal to 5 VWS.CO shares today.''

Assessing individually with SHAP library, the idea it is that the close_0 feature is heavily lowering the probability for this case.


```python
def get_shap_values(execution_date: str, ticker: str = None):
    """
    Generate SHAP values for a given execution date and optionally for a specific ticker.

    Parameters:
    - execution_date (str): Execution date for which to generate SHAP values.
    - all_models (dict): Dictionary containing trained models with dates as keys.
    - all_predicted_tickers (pd.DataFrame): DataFrame containing predicted tickers with their features.
    - ticker (str, optional): Specific ticker to generate SHAP values for. If None, SHAP values for all tickers are generated.

    Returns:
    - shap.Explanation: SHAP values for the specified execution date and ticker.
    """
    date = np.datetime64(execution_date)
    model = all_models.get(date)
    if model is None:
        raise ValueError(f"No model found for the date {execution_date}")

    # Filter X_test for the specified execution date
    X_test = all_predicted_tickers[all_predicted_tickers["execution_date"] == date]

    feature_names = model.booster_.feature_name()

    # Filter for a specific ticker if provided
    if ticker is not None:
        X_test = X_test[X_test["Ticker"] == ticker]
        X_test = X_test.sort_values("Ticker")
        X_test["Ticker"] = X_test["Ticker"].astype("category")

        explainer = shap.Explainer(model)
        shap_values = explainer(X_test[feature_names])
        shap_values = shap_values[..., 1]

    else:
        explainer = shap.Explainer(model, X_test[feature_names])
        shap_values = explainer(X_test[feature_names])

    return shap_values


# Example usage:
sv = get_shap_values("2009-06-30T00:00:00.000000000")
fig = plt.gcf()
ax = plt.gca()
ax.set_position([0.3, 0.1, 0.65, 0.8])
shap.plots.bar(sv, max_display=10)
plt.show()
```


    
![png](module5_files/module5_82_0.png)
    



```python
ticker_top = all_predicted_tickers[
    all_predicted_tickers["execution_date"] == "2009-06-30"
].sort_values("prob", ascending=True)
ticker_top[["Ticker", "prob", "close_0"]].head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>prob</th>
      <th>close_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35739</th>
      <td>FMBI</td>
      <td>0.443811</td>
      <td>7.310000</td>
    </tr>
    <tr>
      <th>33982</th>
      <td>TD</td>
      <td>0.445386</td>
      <td>25.870754</td>
    </tr>
    <tr>
      <th>35811</th>
      <td>DXCM</td>
      <td>0.448631</td>
      <td>6.190000</td>
    </tr>
    <tr>
      <th>34879</th>
      <td>CGEN</td>
      <td>0.448631</td>
      <td>2.010000</td>
    </tr>
    <tr>
      <th>33969</th>
      <td>CDMO</td>
      <td>0.448631</td>
      <td>29.400000</td>
    </tr>
    <tr>
      <th>35339</th>
      <td>HALO</td>
      <td>0.448631</td>
      <td>6.980000</td>
    </tr>
    <tr>
      <th>36119</th>
      <td>EIG</td>
      <td>0.450245</td>
      <td>13.550000</td>
    </tr>
    <tr>
      <th>35123</th>
      <td>NYMX</td>
      <td>0.451938</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>35793</th>
      <td>NLY</td>
      <td>0.452546</td>
      <td>15.140000</td>
    </tr>
    <tr>
      <th>36032</th>
      <td>MVIS</td>
      <td>0.452988</td>
      <td>24.560000</td>
    </tr>
    <tr>
      <th>35041</th>
      <td>CLBS</td>
      <td>0.453316</td>
      <td>190.000000</td>
    </tr>
    <tr>
      <th>35733</th>
      <td>AGEN</td>
      <td>0.453680</td>
      <td>12.540000</td>
    </tr>
    <tr>
      <th>36109</th>
      <td>INT</td>
      <td>0.453908</td>
      <td>20.615000</td>
    </tr>
    <tr>
      <th>34559</th>
      <td>BXP</td>
      <td>0.453912</td>
      <td>47.700000</td>
    </tr>
    <tr>
      <th>34915</th>
      <td>SITC</td>
      <td>0.455972</td>
      <td>8.056089</td>
    </tr>
    <tr>
      <th>34906</th>
      <td>AGM</td>
      <td>0.456270</td>
      <td>4.830000</td>
    </tr>
    <tr>
      <th>34849</th>
      <td>MFG</td>
      <td>0.456532</td>
      <td>4.610000</td>
    </tr>
    <tr>
      <th>35217</th>
      <td>MS</td>
      <td>0.456835</td>
      <td>28.510000</td>
    </tr>
    <tr>
      <th>35804</th>
      <td>HIG</td>
      <td>0.457259</td>
      <td>11.870000</td>
    </tr>
    <tr>
      <th>35166</th>
      <td>UHT</td>
      <td>0.457321</td>
      <td>31.520000</td>
    </tr>
    <tr>
      <th>35241</th>
      <td>IEC</td>
      <td>0.457656</td>
      <td>3.840000</td>
    </tr>
    <tr>
      <th>35148</th>
      <td>WINT</td>
      <td>0.458231</td>
      <td>13356.000000</td>
    </tr>
    <tr>
      <th>35069</th>
      <td>HRTX</td>
      <td>0.458231</td>
      <td>18.800000</td>
    </tr>
    <tr>
      <th>35156</th>
      <td>OLED</td>
      <td>0.458231</td>
      <td>9.780000</td>
    </tr>
    <tr>
      <th>35476</th>
      <td>TGH</td>
      <td>0.458347</td>
      <td>11.490000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sv_top_ticker = get_shap_values("2009-06-30T00:00:00.000000000", "WINT")
fig = plt.gcf()
ax = plt.gca()
ax.set_position([0.3, 0.1, 0.65, 0.8])
shap.plots.waterfall(sv_top_ticker[0])
```


    
![png](module5_files/module5_84_0.png)
    



```python
shap.plots.beeswarm(sv)
```


    
![png](module5_files/module5_85_0.png)
    


Indeed we see that the most penalizing feature was close_0. In the plot, the feature close_0 has a SHAP value of approximately -0.06. This value indicates that for the specific instance being analyzed, the close_0 feature contributes to decreasing the model's output by 0.06 units from the base value (average model output over the dataset). If the model's output is a probability, this would mean the presence or value of close_0 for this instance decreases the predicted probability of the positive class by 0.06.

The reverse stock splitting in this case is leaking info to the past; i.e. if the company approved a stock split (bad signal) it means it didn't go under good circumstances and the price is being calculated backwards, so higher prices will be associated with potential future stock splits, hence company in not good economic position and performing bad against sp500.

In this case, we were carrying an investigation to verify a different stock splitting event presented in class, but I wouldn't conclude that fast that there should be something wrong with the model and the importance given to close_0 value. If I had more time, I would like to research a bit and confirm if this might be also something related to the "regression to the mean" phenomenon, which could give us an idea on why lower stocks might have higher probabilities for surpassing sp500!

One possible solution would be to have the adjusted-prices for stocks, so stock splitting would not affect actual prices, but we do not have that feature in the current dataset (maybe online), so we will just drop the suspicious columns.

## 4. Potential improvements

Having said that, let's do some quick adjustments and will re-compute all the metrics to see if the model has improved:
* We will experiment with XGBoost model as it normally has a better performance for tabular data over the rest classification models, although it may be more time-consuming.
* Assuming that stock markets are influenced by economic cycles, we propose that the average economic cycle lasts approximately 6 to 8 years. To incorporate this hypothesis, we aim to introduce a feature that identifies the current economic phase (e.g., inflationary/deflationary, expansion/contraction, interest rates, etc). To simplify this analysis, we'll use the past 4 years as a proxy to define the upcoming quarters' economic conditions. Therefore, we'll train the model on the most recent 16 n_train_quarters instead of the previous 36 quarters.
* With that, now we will train for every quarter (`period_frequency` = 1) in contrast with just using 1 per each 4 periods.


```python
def get_columns_to_remove():
    columns_to_remove = [
        "date",
        "improve_sp500",
        "Ticker",
        "freq",
        "set",
        "close_sp500_365",
        "close_365",
        "stock_change_365",
        "sp500_change_365",
        "stock_change_div_365",
        "stock_change_730",
        "sp500_change_365",
        "stock_change_div_730",
        "diff_ch_sp500",
        "diff_ch_avg_500",
        "execution_date",
        "target",
        "index",
        "quarter",
        "std_730",
        "count",
    ]

    columns_to_remove = columns_to_remove + technical_features

    return columns_to_remove


def clean_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)


def train_model(train_set, test_set, params, n_estimators=300, path_smooth=None):
    columns_to_remove = get_columns_to_remove()

    X_train = train_set.drop(columns=columns_to_remove, errors="ignore")
    X_test = test_set.drop(columns=columns_to_remove, errors="ignore")

    y_train = train_set["target"]
    y_test = test_set["target"]

    def custom_eval_metric(preds, train_data):
        top_dataset = get_top_tickers_per_prob(preds)
        weighted_return = get_weighted_performance_of_stocks(
            top_dataset, "diff_ch_sp500"
        )
        return "custom_eval_metric", weighted_return

    # Initialize the model with custom eval_metric and other parameters
    clf = xgb.XGBClassifier(**params)

    model = clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=custom_eval_metric,  # Use the custom evaluation function
        verbose=False,
    )

    # Compute feature importance
    pi = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0)
    feature_names = X_test.columns
    df_feature_importance = pd.DataFrame(
        {"importance": pi.importances_mean, "feature": feature_names}
    )

    return model, X_train, X_test, df_feature_importance


def run_model_for_execution_date(
    execution_date: str,
    all_results: Dict,
    all_predicted_tickers_list: List[pd.DataFrame],
    all_models: Dict,
    all_feature_importance: pd.DataFrame,
    params: Dict,
    include_nulls_in_test: bool = False,
) -> Tuple[
    Dict,
    List[pd.DataFrame],
    Dict,
    xgb.XGBClassifier,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Runs the model for a specific execution date and computes feature importance.

    Parameters:
    - execution_date (str): The execution date for the model.
    - all_results (Dict): Dictionary to store all results.
    - all_predicted_tickers_list (List[pd.DataFrame]): List to store all predicted tickers.
    - all_models (Dict): Dictionary to store all models.
    - all_feature_importance (pd.DataFrame): DataFrame to store all feature importances.
    - params (Dict): Parameters for the XGBoost model.
    - include_nulls_in_test (bool): Flag to include nulls in the test set.

    Returns:
    - Tuple[Dict, List[pd.DataFrame], Dict, xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated results, predicted tickers list, models, trained model, training features, testing features, feature importance dataframe.
    """
    global train_set
    global test_set

    train_set, test_set = split_train_test_by_period(
        data_set, execution_date, include_nulls_in_test=include_nulls_in_test
    )
    clean_data(train_set)
    clean_data(test_set)
    train_size, _ = train_set.shape
    test_size, _ = test_set.shape

    model = None
    X_train = None
    X_test = None

    if train_size > 0 and test_size > 0:
        model, X_train, X_test, df_feature_importance = train_model(
            train_set, test_set, params
        )
        test_set["prob"] = model.predict_proba(X_test)[:, 1]
        predicted_tickers = test_set.sort_values("prob", ascending=False)
        predicted_tickers["execution_date"] = execution_date
        all_results[execution_date] = model.evals_result_
        all_models[execution_date] = model
        all_predicted_tickers_list.append(predicted_tickers)

        # Append feature importance for this execution date
        df_feature_importance["execution_date"] = execution_date
        all_feature_importance = pd.concat(
            [all_feature_importance, df_feature_importance]
        )

    return (
        all_results,
        all_predicted_tickers_list,
        all_models,
        X_train,
        X_test,
        all_feature_importance,
    )
```


```python
params = {
    "random_state": 1,
    "n_jobs": 10,
    "n_estimators": 20,
    "learning_rate": 0.01,
    "reg_lambda": 0.3,
    "objective": "binary:logistic",
    "missing": np.nan,
}

n_train_quarters = 16
period_frequency = 1

all_results = {}
all_predicted_tickers_list = []
all_models = {}
all_feature_importance = pd.DataFrame()

execution_dates = np.sort(data_set["execution_date"].unique())
# start_date = np.datetime64('2015-06-30')
# execution_dates = [date for date in execution_dates if date >= start_date]


for i, execution_date in enumerate(execution_dates):
    if i % period_frequency == 0:
        print(f"Execution Date {i}: {execution_date}")
        (
            all_results,
            all_predicted_tickers_list,
            all_models,
            X_train,
            X_test,
            all_feature_importance,
        ) = run_model_for_execution_date(
            execution_date,
            all_results,
            all_predicted_tickers_list,
            all_models,
            all_feature_importance,
            params,
        )
```

    Execution Date 0: 2005-06-30T00:00:00.000000000
    Execution Date 1: 2005-09-30T00:00:00.000000000
    Execution Date 2: 2005-12-30T00:00:00.000000000
    Execution Date 3: 2006-03-31T00:00:00.000000000
    Execution Date 4: 2006-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 5: 2006-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 6: 2006-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 7: 2007-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 8: 2007-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 9: 2007-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 10: 2007-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 11: 2008-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 12: 2008-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 13: 2008-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 14: 2008-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 15: 2009-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 16: 2009-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 17: 2009-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 18: 2009-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 19: 2010-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 20: 2010-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 21: 2010-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 22: 2010-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 23: 2011-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 24: 2011-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 25: 2011-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 26: 2011-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 27: 2012-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 28: 2012-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 29: 2012-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 30: 2012-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 31: 2013-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 32: 2013-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 33: 2013-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 34: 2013-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 35: 2014-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 36: 2014-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 37: 2014-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 38: 2014-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 39: 2015-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 40: 2015-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 41: 2015-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 42: 2015-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 43: 2016-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 44: 2016-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 45: 2016-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 46: 2016-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 47: 2017-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 48: 2017-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 49: 2017-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 50: 2017-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 51: 2018-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 52: 2018-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 53: 2018-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 54: 2018-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 55: 2019-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 56: 2019-06-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 57: 2019-09-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 58: 2019-12-30T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 59: 2020-03-31T00:00:00.000000000


    `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.


    Execution Date 60: 2020-06-30T00:00:00.000000000
    Execution Date 61: 2020-09-30T00:00:00.000000000
    Execution Date 62: 2020-12-30T00:00:00.000000000
    Execution Date 63: 2021-03-27T00:00:00.000000000



```python
def parse_results_into_df(set_):
    df = pd.DataFrame()
    for date in all_results:
        df_tmp = pd.DataFrame(all_results[(date)][set_])
        df_tmp["n_trees"] = list(range(len(df_tmp)))
        df_tmp["execution_date"] = date
        df = pd.concat([df, df_tmp])

    df["execution_date"] = df["execution_date"].astype(str)

    return df


test_results = parse_results_into_df("validation_1")
train_results = parse_results_into_df("validation_0")


def compute_learning_rate(set_: str, all_results: dict) -> pd.DataFrame:
    """
    Calculate the normalized learning rates for the model

    Parameters:
    - set_ (str): The dataset type ('training' or 'valid_0').
    - all_results (dict): Dictionary containing training results for each execution date.

    Returns:
    - pd.DataFrame: DataFrame with normalized learning rates, including execution dates and tree categories.
    """

    df = pd.DataFrame()
    for date in all_results:
        df_tmp = pd.DataFrame(all_results[date][set_])
        df_tmp["n_trees"] = range(len(df_tmp))
        df_tmp["execution_date"] = date
        df = pd.concat([df, df_tmp])

    # Calculate the % diff respect to first tree
    df["first_tree_logloss"] = df.groupby(["execution_date"])["logloss"].transform(
        "first"
    )
    df[f"normalized_learning_{set_}"] = (df["logloss"] - df["first_tree_logloss"]) / df[
        "first_tree_logloss"
    ]
    df = df.drop(columns="first_tree_logloss")

    return df


learning_rates_train = compute_learning_rate("validation_0", all_results)
learning_rates_test = compute_learning_rate("validation_1", all_results)

# Filter for n_trees <= 20
learning_rates_train = learning_rates_train[learning_rates_train["n_trees"] <= 20]
learning_rates_test = learning_rates_test[learning_rates_test["n_trees"] <= 20]

# Convert 'n_trees' to a categorical variable for better plotting
learning_rates_train = learning_rates_train[
    ["n_trees", "normalized_learning_validation_0", "execution_date"]
]
learning_rates_test = learning_rates_test[
    ["n_trees", "normalized_learning_validation_1", "execution_date"]
]

# Rename columns for merging
learning_rates_train.rename(
    columns={"normalized_learning_validation_0": "Normalized Learning"}, inplace=True
)
learning_rates_test.rename(
    columns={"normalized_learning_validation_1": "Normalized Learning"}, inplace=True
)

learning_rates_train["Set"] = "Training"
learning_rates_test["Set"] = "Validation"

learning_rates_combined = pd.concat([learning_rates_train, learning_rates_test])


learning_rates_combined["n_trees_cat"] = pd.Categorical(
    learning_rates_combined["n_trees"],
    categories=sorted(learning_rates_combined["n_trees"].unique()),
)

# Plotting
plot = (
    ggplot(
        learning_rates_combined,
        aes(x="n_trees_cat", y="Normalized Learning", color="Set"),
    )
    + geom_boxplot()
    + scale_color_manual(values=["blue", "red"])
)

print(plot)
```


    
![png](module5_files/module5_91_0.png)
    


    


Let's check the feature imporance again:


```python
all_feature_importance = all_feature_importance.sort_values(
    ["execution_date", "importance"], ascending=False
)
all_feature_importance_mean = (
    all_feature_importance.groupby("feature")["importance"].mean().reset_index()
)
all_feature_importance_mean = all_feature_importance_mean.sort_values(
    "importance", ascending=False
)
all_feature_importance_mean_importants = all_feature_importance_mean.head(10)

all_feature_importance_mean_importants["feature"] = pd.Categorical(
    all_feature_importance_mean_importants["feature"],
    categories=all_feature_importance_mean_importants["feature"],
    ordered=True,
)

plot = (
    ggplot(
        all_feature_importance_mean_importants,
        aes(x="feature", y="importance", fill="importance"),
    )
    + geom_col()
    + scale_fill_gradient(low="lightblue", high="darkblue")
    + coord_flip()
)  # This will make the plot horizontal

print(plot)
```

    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy



    
![png](module5_files/module5_93_1.png)
    


    


And if we inspect one particular value for 2016-06-30 (first we need to edit the get_shap_values func, replacing the booster__ attribute with the get_booster() method)


```python
import numpy as np
import shap
import pandas as pd


def get_shap_values(execution_date: str, ticker: str = None):
    """
    Generate SHAP values for a given execution date and optionally for a specific ticker.

    Parameters:
    - execution_date (str): Execution date for which to generate SHAP values.
    - all_models (dict): Dictionary containing trained models with dates as keys.
    - all_predicted_tickers (pd.DataFrame): DataFrame containing predicted tickers with their features.
    - ticker (str, optional): Specific ticker to generate SHAP values for. If None, SHAP values for all tickers are generated.

    Returns:
    - shap.Explanation: SHAP values for the specified execution date and ticker.
    """
    date = np.datetime64(execution_date)
    model = all_models.get(date)
    if model is None:
        raise ValueError(f"No model found for the date {execution_date}")

    # Filter X_test for the specified execution date
    X_test = all_predicted_tickers[all_predicted_tickers["execution_date"] == date]

    feature_names = model.get_booster().feature_names

    # Filter for a specific ticker if provided
    if ticker is not None:
        X_test = X_test[X_test["Ticker"] == ticker]
        X_test = X_test.sort_values("Ticker")
        X_test["Ticker"] = X_test["Ticker"].astype("category")

        explainer = shap.Explainer(model)
        shap_values = explainer(X_test[feature_names])
        shap_values = shap_values[..., 1]

    else:
        explainer = shap.Explainer(model, X_test[feature_names])
        shap_values = explainer(X_test[feature_names])

    return shap_values
```


```python
all_predicted_tickers = []
all_predicted_tickers = pd.concat(all_predicted_tickers_list)
all_predicted_tickers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>date</th>
      <th>AssetTurnover</th>
      <th>CashFlowFromFinancialActivities</th>
      <th>CashFlowFromInvestingActivities</th>
      <th>CashFlowFromOperatingActivities</th>
      <th>CashOnHand</th>
      <th>ChangeInAccountsPayable</th>
      <th>ChangeInAccountsReceivable</th>
      <th>ChangeInAssetsLiabilities</th>
      <th>...</th>
      <th>NetCashFlow_change_1_years</th>
      <th>NetCashFlow_change_2_years</th>
      <th>CurrentRatio_change_1_years</th>
      <th>CurrentRatio_change_2_years</th>
      <th>Market_cap__minus_365</th>
      <th>Market_cap__minus_730</th>
      <th>diff_ch_sp500</th>
      <th>count</th>
      <th>target</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9837</th>
      <td>BRFS</td>
      <td>2006-03-31</td>
      <td>0.0770</td>
      <td>-18.938</td>
      <td>-0.533</td>
      <td>18.025</td>
      <td>8.200</td>
      <td>-2.4880</td>
      <td>-4.031</td>
      <td>2.352</td>
      <td>...</td>
      <td>2.096248e+15</td>
      <td>NaN</td>
      <td>0.189180</td>
      <td>NaN</td>
      <td>742.443</td>
      <td>NaN</td>
      <td>0.808165</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.653183</td>
    </tr>
    <tr>
      <th>9460</th>
      <td>JPM</td>
      <td>2006-03-31</td>
      <td>0.0113</td>
      <td>53821.000</td>
      <td>-34501.000</td>
      <td>-19141.000</td>
      <td>513228.000</td>
      <td>-0.5480</td>
      <td>-125.000</td>
      <td>-9752.000</td>
      <td>...</td>
      <td>-1.071134e+00</td>
      <td>NaN</td>
      <td>0.019588</td>
      <td>NaN</td>
      <td>126085.336</td>
      <td>NaN</td>
      <td>0.002404</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.651234</td>
    </tr>
    <tr>
      <th>9748</th>
      <td>T</td>
      <td>2006-03-31</td>
      <td>0.1091</td>
      <td>-70.000</td>
      <td>-2555.000</td>
      <td>2458.000</td>
      <td>1057.000</td>
      <td>-1.8060</td>
      <td>509.000</td>
      <td>-189.000</td>
      <td>...</td>
      <td>1.472973e+00</td>
      <td>NaN</td>
      <td>0.228051</td>
      <td>NaN</td>
      <td>78731.250</td>
      <td>NaN</td>
      <td>0.353741</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.649785</td>
    </tr>
    <tr>
      <th>8577</th>
      <td>BTI</td>
      <td>2006-03-31</td>
      <td>0.1207</td>
      <td>-354.679</td>
      <td>18.269</td>
      <td>333.527</td>
      <td>65.989</td>
      <td>-2.5758</td>
      <td>2.037</td>
      <td>-8.631</td>
      <td>...</td>
      <td>-4.341307e+00</td>
      <td>NaN</td>
      <td>0.010495</td>
      <td>NaN</td>
      <td>88897.800</td>
      <td>NaN</td>
      <td>0.222591</td>
      <td>2052</td>
      <td>1.0</td>
      <td>0.647899</td>
    </tr>
    <tr>
      <th>8619</th>
      <td>GNW</td>
      <td>2006-03-31</td>
      <td>0.0233</td>
      <td>-76.000</td>
      <td>-649.000</td>
      <td>771.000</td>
      <td>1909.000</td>
      <td>15.0000</td>
      <td>-51.536</td>
      <td>330.000</td>
      <td>...</td>
      <td>1.223108e+00</td>
      <td>NaN</td>
      <td>0.278079</td>
      <td>NaN</td>
      <td>14942.689</td>
      <td>NaN</td>
      <td>-0.186275</td>
      <td>2052</td>
      <td>0.0</td>
      <td>0.647543</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 147 columns</p>
</div>




```python
sv = get_shap_values("2016-06-30T00:00:00.000000000")
fig = plt.gcf()
ax = plt.gca()
ax.set_position([0.3, 0.1, 0.65, 0.8])
shap.plots.bar(sv, max_display=10)
plt.show()
```

    [16:49:53] WARNING: /workspace/src/c_api/c_api.cc:1240: Saving into deprecated binary model format, please consider using `json` or `ubj`. Model format will default to JSON in XGBoost 2.2 if not specified.



    
![png](module5_files/module5_97_1.png)
    



```python
stocks = all_predicted_tickers[all_predicted_tickers["execution_date"] == "2016-06-30"]
stocks.sort_values("prob", ascending=False).head(10)[["Ticker", "prob"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107215</th>
      <td>SNOA</td>
      <td>0.520623</td>
    </tr>
    <tr>
      <th>106133</th>
      <td>FPI</td>
      <td>0.519074</td>
    </tr>
    <tr>
      <th>106966</th>
      <td>EVOK</td>
      <td>0.519074</td>
    </tr>
    <tr>
      <th>107090</th>
      <td>MACK</td>
      <td>0.516523</td>
    </tr>
    <tr>
      <th>107407</th>
      <td>WVE</td>
      <td>0.515716</td>
    </tr>
    <tr>
      <th>106530</th>
      <td>HCA</td>
      <td>0.514517</td>
    </tr>
    <tr>
      <th>107053</th>
      <td>TGH</td>
      <td>0.514380</td>
    </tr>
    <tr>
      <th>107335</th>
      <td>WMC</td>
      <td>0.513756</td>
    </tr>
    <tr>
      <th>106284</th>
      <td>TVTY</td>
      <td>0.513748</td>
    </tr>
    <tr>
      <th>106799</th>
      <td>PTMN</td>
      <td>0.512834</td>
    </tr>
  </tbody>
</table>
</div>




```python
stocks.sort_values("prob", ascending=False).tail(10)[["Ticker", "prob"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105677</th>
      <td>TWO</td>
      <td>0.441638</td>
    </tr>
    <tr>
      <th>107318</th>
      <td>AJX</td>
      <td>0.441477</td>
    </tr>
    <tr>
      <th>105898</th>
      <td>ARCC</td>
      <td>0.441477</td>
    </tr>
    <tr>
      <th>106206</th>
      <td>VOC</td>
      <td>0.441455</td>
    </tr>
    <tr>
      <th>106889</th>
      <td>ET</td>
      <td>0.437134</td>
    </tr>
    <tr>
      <th>104953</th>
      <td>MU</td>
      <td>0.436772</td>
    </tr>
    <tr>
      <th>107016</th>
      <td>BPT</td>
      <td>0.423908</td>
    </tr>
    <tr>
      <th>104957</th>
      <td>SAR</td>
      <td>0.423908</td>
    </tr>
    <tr>
      <th>104749</th>
      <td>NRT</td>
      <td>0.423908</td>
    </tr>
    <tr>
      <th>106185</th>
      <td>CPSS</td>
      <td>0.423908</td>
    </tr>
  </tbody>
</table>
</div>




```python
sv_top_ticker = get_shap_values("2016-06-30T00:00:00.000000000", "NRT")
fig = plt.gcf()
ax = plt.gca()
ax.set_position([0.2, 0.3, 0.6, 0.5])
# Check if sv_top_ticker is empty
if sv_top_ticker.shape[0] > 0:
    shap.plots.waterfall(sv_top_ticker)
else:
    print("No SHAP values available for the specified ticker and date.")

plt.show()
```

    [16:49:55] WARNING: /workspace/src/c_api/c_api.cc:1240: Saving into deprecated binary model format, please consider using `json` or `ubj`. Model format will default to JSON in XGBoost 2.2 if not specified.



    
![png](module5_files/module5_100_1.png)
    


I understand that this horrible plot shows that no feature plays an important role here (x scale is ~0), as described previously.


```python
shap.plots.beeswarm(sv)
```


    
![png](module5_files/module5_102_0.png)
    


Pretty much balanced for all the features. Simplifying a bit, lows EBITDA penalizes the prediction (low earnings -> bad company); also high Price-to-Free Cash Flow impacts positive on the prediction (investors have high expectations for the company -> good company), so at a first glance makes sense to me.


```python
test_results = parse_results_into_df("validation_1")
train_results = parse_results_into_df("validation_0")
```


```python
test_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logloss</th>
      <th>custom_eval_metric</th>
      <th>n_trees</th>
      <th>execution_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.720886</td>
      <td>0.063183</td>
      <td>0</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.720558</td>
      <td>0.080899</td>
      <td>1</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.720176</td>
      <td>0.359252</td>
      <td>2</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.719796</td>
      <td>0.356899</td>
      <td>3</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.719511</td>
      <td>0.477344</td>
      <td>4</td>
      <td>2006-06-30</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logloss</th>
      <th>custom_eval_metric</th>
      <th>n_trees</th>
      <th>execution_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.676475</td>
      <td>2.835162</td>
      <td>0</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.674363</td>
      <td>0.440108</td>
      <td>1</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.672265</td>
      <td>0.266251</td>
      <td>2</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.670189</td>
      <td>0.468292</td>
      <td>3</td>
      <td>2006-06-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.667889</td>
      <td>0.412792</td>
      <td>4</td>
      <td>2006-06-30</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_results_final_tree = test_results.sort_values(
    ["execution_date", "n_trees"]
).drop_duplicates("execution_date", keep="last")
train_results_final_tree = train_results.sort_values(
    ["execution_date", "n_trees"]
).drop_duplicates("execution_date", keep="last")
```


```python
(
    ggplot(test_results_final_tree)
    + geom_point(aes(x="execution_date", y="custom_eval_metric"))
    + theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1))
)
```


    
![png](module5_files/module5_108_0.png)
    





    <Figure Size: (640 x 480)>




```python
(
    ggplot(train_results_final_tree)
    + geom_point(aes(x="execution_date", y="custom_eval_metric"))
    + theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1))
)
```


    
![png](module5_files/module5_109_0.png)
    





    <Figure Size: (640 x 480)>




```python
test_results_final_tree["execution_date"].unique()
```




    array(['2006-06-30', '2006-09-30', '2006-12-30', '2007-03-31',
           '2007-06-30', '2007-09-30', '2007-12-30', '2008-03-31',
           '2008-06-30', '2008-09-30', '2008-12-30', '2009-03-31',
           '2009-06-30', '2009-09-30', '2009-12-30', '2010-03-31',
           '2010-06-30', '2010-09-30', '2010-12-30', '2011-03-31',
           '2011-06-30', '2011-09-30', '2011-12-30', '2012-03-31',
           '2012-06-30', '2012-09-30', '2012-12-30', '2013-03-31',
           '2013-06-30', '2013-09-30', '2013-12-30', '2014-03-31',
           '2014-06-30', '2014-09-30', '2014-12-30', '2015-03-31',
           '2015-06-30', '2015-09-30', '2015-12-30', '2016-03-31',
           '2016-06-30', '2016-09-30', '2016-12-30', '2017-03-31',
           '2017-06-30', '2017-09-30', '2017-12-30', '2018-03-31',
           '2018-06-30', '2018-09-30', '2018-12-30', '2019-03-31',
           '2019-06-30', '2019-09-30', '2019-12-30', '2020-03-31'],
          dtype=object)




```python
test_results_final_tree = merge_against_benchmark(
    test_results_final_tree, all_predicted_tickers
)
test_results_final_tree.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logloss</th>
      <th>custom_eval_metric</th>
      <th>n_trees</th>
      <th>execution_date</th>
      <th>diff_ch_sp500_baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.716178</td>
      <td>0.364069</td>
      <td>19</td>
      <td>2006-06-30</td>
      <td>0.049213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.704118</td>
      <td>-0.029699</td>
      <td>19</td>
      <td>2006-09-30</td>
      <td>0.067796</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.711382</td>
      <td>-0.001069</td>
      <td>19</td>
      <td>2006-12-30</td>
      <td>0.068473</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.701398</td>
      <td>-0.066858</td>
      <td>19</td>
      <td>2007-03-31</td>
      <td>0.048029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.695110</td>
      <td>-0.030262</td>
      <td>19</td>
      <td>2007-06-30</td>
      <td>0.077166</td>
    </tr>
  </tbody>
</table>
</div>




```python
(
    ggplot(
        test_results_final_tree[test_results_final_tree["custom_eval_metric"] < 10],
        aes(x="execution_date"),
    )
    + geom_point(aes(y="custom_eval_metric"), colour="blue")
    + geom_point(aes(y="diff_ch_sp500_baseline"), colour="red")
    + theme(axis_text_x=element_text(angle=90, vjust=0.5, hjust=1))
)
```


    
![png](module5_files/module5_112_0.png)
    





    <Figure Size: (640 x 480)>




```python
print(f"Mean for model: {test_results_final_tree['custom_eval_metric'].mean()}")
print(
    f"Mean for baseline: {test_results_final_tree['diff_ch_sp500_baseline'].mean()}\n"
)

print(f"Median for model: {test_results_final_tree['custom_eval_metric'].median()}")
print(
    f"Median for baseline: {test_results_final_tree['diff_ch_sp500_baseline'].median()}\n"
)
```

    Mean for model: 0.09278869642857143
    Mean for baseline: 0.022159133577893696
    
    Median for model: 0.011363999999999999
    Median for baseline: 0.015525563344158869
    


## Conclusions

**Mean and Median Analysis: Model vs. Baseline**.
The mean value of your model stands at approximately 9.28%, significantly higher than the baseline's mean of about 2.22%. This indicates that on average, your model predicts considerably higher returns compared to the baseline. However, when looking at the median values, a different picture emerges. The model's median is around 1.14%, lower than the baseline's 1.55%. This suggests that while the model excels in generating higher average returns, its performance across various scenarios is less consistent, as the median is more resistant to outliers than the mean.

**High-level and low-level Metrics Approach**. In terms of metrics, we've adopted a dual approach. At the high level, we focused on overarching performance indicators like yearly returns compared to benchmarks like the SP500, assessing the general effectiveness of the model. At the low level, we concentrated on detailed aspects such as variable importance and median performance, which offer insights into the model's behavior in typical scenarios and its resilience to extreme cases. This combination of high-level and low-level analysis provides a comprehensive view, highlighting the model's strong average performance and areas for potential refinement.


1. **Model Performance Improvement**: Enhanced predictive capabilities with reduced validation error, showing improved performance compared to the `diff_ch_sp500_baseline`.
2. **Risk Management and Variable Importance**: Increased robustness and reduced overfitting by addressing data leakage issues, leading to better risk handling by eliminating reliance on misleading variables.
3. **Mean Performance Values**: Significant improvement in final mean values over the initial baseline, with the model outperforming the SP500 on average after refinements.
4. **High-Risk, High-Reward Strategy**: Adoption of a strategy with higher risks but potentially higher rewards, capable of achieving exceptional results in certain periods, indicating a riskier approach.
5. **Potential Areas for Future Improvement**:
    - Exploration of different `n_quarters` variables for temporal trend analysis and optimization of the `top_n` parameter could further balance risk and reward. Additionally, other algorithm for optimising this selection might be worth to explore.
    - Temporal Trend Analysis: Deepening the investigation into different n_quarters variables to better understand and capture market cycles.
    - Optimization of Investment Strategies: Fine-tuning the top_n parameter to improve risk-adjusted returns and balance investment strategies.
    - Variable Selection and Data Leakage: Continual refinement in variable selection to prevent data leakage, thereby enhancing the model's robustness.
    - Handling Stock Splits and Outliers: Implementing strategies to adjust for stock splits and outliers, instead of eliminating such stocks, to maintain data integrity and model effectiveness (e.g. delete only suspicious high price tickers, manage to get price-adjusted stock split, etc).
    - Exploring Alternative Algorithms: Testing various algorithms for optimizing stock selection, focusing on those better suited for financial data complexities.
    - Exploring Advanced Train-Test Splitting Methods: Investigating more sophisticated approaches for dividing data into training and testing sets to enhance the model's ability to generalize to unseen data, potentially incorporating techniques like time-series cross-validation or stratified splits based on market conditions.
    - Continuous Model Evaluation: Regularly updating and back-testing the model against current market trends to ensure its ongoing relevance and accuracy.
6. **Overall Reflection**: Successful transition to a higher level of abstraction in metrics, focusing on yearly returns compared to the SP500, acknowledging the complexity of financial modeling and the potential for ongoing optimization.

-- Now we're more than ready to be hired by Renaissance Tech or Two Sigma :D
