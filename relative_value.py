# %%
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date, datetime, timedelta
import time
import yfinance as yf
from pyetfdb_scraper import etf
from IPython.display import clear_output
import requests
from bs4 import BeautifulSoup
import zipfile
from io import BytesIO

from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm
import scipy
import os
from dotenv import load_dotenv

import nasdaqdatalink as ndl

#TD Ameritrade API for historical equity prices and current quotes
from tda.auth import easy_client

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import selenium.common.exceptions
capa = DesiredCapabilities.CHROME.copy()
capa["pageLoadStrategy"] = "eager"


chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--start-maximized')



load_dotenv('e.env')

# %%
#initialize chromedriver function 
def driver():
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()))

#initialize TDA easy_client
c = easy_client(
    webdriver_func=driver,
    api_key=os.environ['tda_api_key'],
    redirect_uri='https://localhost',
    token_path='token.json'
)


ndl.ApiConfig.api_key = os.environ['ndl_api_key']

# %%
def get_iv_from_straddle(straddle_price: float, underlying_price: float, dte: int or float=30):
    """
    #IV from straddle price - to be used when current IV data isn't avail
    """
    res = (125*straddle_price)/(underlying_price*np.sqrt(dte/360))
    return res

# %%
def get_current_iv(symbol: str, dte: int=30, strike_count: int=2, volume_lookback: int=1, dte_threshold: int=20):
    """
    get the current ATM implied volatility for a symbol and target DTE from yfinance. 

    :param dte: the target DTE of the options. The closest standard (monthly) expiration date to today+DTE is used
    :param strike_count: how many of the closest ATM strikes are considered in the calc.
    :param volume_lookback: number of trading days to look back for traded volume in the strikes being analyzed.
                            if any of the strikes have no volume in the period, NaN is returned.
    """

    td = (datetime.today() - BDay(volume_lookback)).date()

    try:
        tk = yf.Ticker(symbol)
    except KeyError:
        time.sleep(2)
        tk = yf.Ticker(symbol)
    
    try:
        s = tk.info['currentPrice']
    except KeyError:
        s = tk.fast_info['lastPrice']

    thirdfris = pd.date_range(td,td+timedelta(365),freq='WOM-3FRI')

    if len(tk.options) > 0:
        ds = [d for d in tk.options if d in thirdfris]
        e = ds[pd.Series(abs((pd.to_datetime(ds) - datetime.today()).days - dte)).idxmin()]
        if abs((pd.to_datetime(e) - datetime.today()).days - dte) > dte_threshold:
            return np.nan
    else:
        return np.nan


    try: 
        calls = pd.DataFrame(tk.option_chain(e).calls)
        puts = pd.DataFrame(tk.option_chain(e).puts)
    except TypeError:
        time.sleep(2)
        calls = pd.DataFrame(tk.option_chain(e).calls)
        puts = pd.DataFrame(tk.option_chain(e).puts)
        pass


    
    calls = calls.sort_values(by='strike', key = lambda x: abs(x-s)).iloc[:strike_count,:]
    puts = puts.sort_values(by='strike', key = lambda x: abs(x-s)).iloc[:strike_count,:]


    if (calls['lastTradeDate'].dt.date < td).any() or (puts['lastTradeDate'].dt.date < td).any():
        return np.nan
    
    opts = pd.concat([calls, puts])

    if datetime.today().hour >= 10:
        return opts['impliedVolatility'].mean()
    else: 
        straddle_price = opts['lastPrice'].mean()*2
        dte_exact = (datetime.strptime(e,'%Y-%m-%d') - datetime.today()).days
        return get_iv_from_straddle(straddle_price=straddle_price, underlying_price=s, dte=dte_exact)


# %%
def get_hist_iv_data(tickers_avail: list, start_date: datetime or datetime.date=datetime.today()-timedelta(365), end_date: datetime or datetime.date=datetime.today(),
                rows: int=None, write_to_file: bool=False):
    """ 
    pull historical daily IV data from the nasdaq data link series OPT for the provided list of tickers_avail. 
    If 'rows' is specified, pulls the last <rows> data points.
    If not, then pulls data from <start_date> to <end_date>
    """
    iv_dict = {}
    if rows is None:
        for ticker in tqdm(tickers_avail):
            print(ticker)
            iv_dict[ticker] = ndl.get(f'OPT/{ticker}', start_date=start_date, end_date=end_date)[['stockpx','iv30','iv60','iv90']]
            clear_output()
    else:
        for ticker in tqdm(tickers_avail):
            print(ticker)
            iv_dict[ticker] = ndl.get(f'OPT/{ticker}', rows=rows)[['stockpx','iv30','iv60','iv90']]
            clear_output()
        
    stockpx_df = pd.concat([iv_dict[ticker]['stockpx'].rename(ticker) for ticker in iv_dict.keys()], axis=1, join='outer')
    iv30_df = pd.concat([iv_dict[ticker]['iv30'].rename(ticker) for ticker in iv_dict.keys()], axis=1, join='outer')
    iv60_df = pd.concat([iv_dict[ticker]['iv60'].rename(ticker) for ticker in iv_dict.keys()], axis=1, join='outer')
    iv90_df = pd.concat([iv_dict[ticker]['iv90'].rename(ticker) for ticker in iv_dict.keys()], axis=1, join='outer')

    if write_to_file:
        path = f'IV_Data_{datetime.today().date()}'
        if not os.path.exists(path):
            os.mkdir(path)
        stockpx_df.to_csv(path+'/stockpx.csv')
        iv30_df.to_csv(path+'/iv_30.csv')
        iv60_df.to_csv(path+'/iv_60.csv')
        iv90_df.to_csv(path+'/iv_90.csv')
    return stockpx_df, iv30_df, iv60_df, iv90_df

# %%
def get_rvs(stockpx_df: pd.DataFrame, dte: int=30):
    df = np.log(stockpx_df/stockpx_df.shift(1))
    df.fillna(0, inplace=True)
    df = df.rolling(window=dte).std(ddof=0)*np.sqrt(252)*100
    return df.round(2)

# %%
def get_vrps(stockpx_df: pd.DataFrame, iv_df: pd.DataFrame, dte: int=30):
    rv_df = get_rvs(stockpx_df=stockpx_df, dte=dte)
    vrp_df = (iv_df/rv_df).round(2)
    return vrp_df

# %%
def get_correlated_ivs(iv_df, stockpx_df, corr_threshold: float, stockpx_threshold: float=2.0, use_iv_df: bool=False, use_stockpx_corr: bool=False, tickers=None):
    tickers = [x for x in tickers if x in iv_df.columns]
    iv_df = iv_df.loc[:,tickers]

    
    if use_iv_df:
        iv_df = iv_df.iloc[:-1,:]


    if use_stockpx_corr:
        tickers = [x for x in tickers if x in stockpx_df.columns]    
        stockpx_df = stockpx_df.loc[:,tickers]
        corrs = stockpx_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates().rename('r(S)')
    else:
        corrs = iv_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates().rename('r')
        
    highcorrs = corrs[(corrs<1) & (corrs>corr_threshold)]
    highcorrs.index.names = ['pair1', 'pair2']

    highcorrs = pd.DataFrame(highcorrs)

    highcorrs = highcorrs.join(stockpx_df.iloc[-1,:].rename('0'), on = 'pair1').rename(columns={'0': 'stockpx_1'})
    highcorrs = highcorrs.join(stockpx_df.iloc[-1,:].rename('0'), on = 'pair2').rename(columns={'0': 'stockpx_2'})

    highcorrs = highcorrs[(highcorrs['stockpx_1'] > stockpx_threshold) & (highcorrs['stockpx_2'] > stockpx_threshold)]

    IV_ratios = pd.DataFrame({pair: iv_df[pair[0]]/iv_df[pair[1]] for pair in highcorrs.index})
    
    return highcorrs, IV_ratios

# %%
def get_correlated_vrps(vrp_df, stockpx_df, corr_threshold: float, stockpx_threshold: float=2.0, 
                        use_vrp_df: bool=False, use_stockpx_corr: bool=False, tickers=None):
    tickers = [x for x in tickers if x in vrp_df.columns]
    vrp_df = vrp_df.loc[:,tickers]

    
    if use_vrp_df:
        vrp_df = vrp_df.iloc[:-1,:]


    if use_stockpx_corr:
        tickers = [x for x in tickers if x in stockpx_df.columns]    
        stockpx_df = stockpx_df.loc[:,tickers]
        corrs = stockpx_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates().rename('r(S)')
    else:
        corrs = vrp_df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates().rename('r(VRP)')
    highcorrs = corrs[(corrs<1) & (corrs>corr_threshold)]
    highcorrs.index.names = ['pair1', 'pair2']

    highcorrs = pd.DataFrame(highcorrs)
   
    highcorrs = highcorrs.join(stockpx_df.iloc[-1,:].rename('0'), on = 'pair1').rename(columns={'0': 'stockpx_1'})
    highcorrs = highcorrs.join(stockpx_df.iloc[-1,:].rename('0'), on = 'pair2').rename(columns={'0': 'stockpx_2'})

    highcorrs = highcorrs[(highcorrs['stockpx_1'] > stockpx_threshold) & (highcorrs['stockpx_2'] > stockpx_threshold)]

    VRP_ratios = pd.DataFrame({pair: vrp_df[pair[0]]/vrp_df[pair[1]] for pair in highcorrs.index})
    
    return highcorrs, VRP_ratios

# %%
def get_current_iv_ratio_ranks(iv_ratio_df: pd.DataFrame, corr_df:pd.DataFrame, stockpx_df: pd.DataFrame,
                         dte: int=30, strike_count: int=2, volume_lookback: int=1, dte_threshold: int=20, z_window: int=100, use_iv_df: bool=False, iv_df=None):



    tickers = np.unique(np.append(iv_ratio_df.columns.to_frame()[0].values, iv_ratio_df.columns.to_frame()[1].values))
    current_iv_dict = {}
    
    if use_iv_df:
        current_iv_dict = iv_df.iloc[-1,:].T.to_dict()
    else:
        for ticker in tqdm(tickers):
            print(ticker)
            current_iv_dict[ticker] = get_current_iv(ticker, dte=dte, strike_count=strike_count, volume_lookback=volume_lookback, dte_threshold=dte_threshold)
            clear_output()


    ratio_dict = {}
    zscore_dict = {}
    pctl_dict = {}
    beta_dict = {}
    beta_premium_dict = {}
    for pair in iv_ratio_df.columns:
        ratio_dict[pair] = current_iv_dict[pair[0]]/current_iv_dict[pair[1]]
        zscore_dict[pair] = (ratio_dict[pair] - iv_ratio_df[pair][-z_window:].mean())/(iv_ratio_df[pair][-z_window:].std())
        pctl_dict[pair] = scipy.stats.percentileofscore(iv_ratio_df[pair][~np.isnan(iv_ratio_df[pair])], 
                                                                        ratio_dict[pair], 'weak')
        beta_dict[pair] = (abs(stockpx_df[pair[0]].pct_change()/stockpx_df[pair[1]].pct_change())).replace([np.inf,-np.inf], np.nan).mean()
        beta_premium_dict[pair] = abs(current_iv_dict[pair[1]] - current_iv_dict[pair[0]]*beta_dict[pair])


    current_ratio_df = pd.DataFrame.from_dict(ratio_dict, 'index', columns=['iv_ratio'])
    zscore_df = pd.DataFrame.from_dict(zscore_dict, 'index', columns=['zscore'])
    pctl_df = pd.DataFrame.from_dict(pctl_dict, 'index', columns=['pctl'])
    if stockpx_df is not None:
        beta_df = pd.DataFrame.from_dict(beta_dict, 'index', columns=['beta'])
        beta_premium_df = pd.DataFrame.from_dict(beta_premium_dict, 'index', columns=['beta_premium'])

    if stockpx_df is not None:
        res = pd.concat([current_ratio_df, zscore_df, pctl_df, beta_df, beta_premium_df], axis=1, join='inner')
    else:
        res = pd.concat([current_ratio_df, zscore_df, pctl_df], axis=1, join='inner')
        
    res.index = pd.MultiIndex.from_tuples(res.index)
    res.index.names = ['pair1', 'pair2']
    res.reset_index(inplace=True)
    
    current_iv_df = pd.DataFrame.from_dict(current_iv_dict, 'index')

    res = res.join(current_iv_df, on='pair1')
    res = res.join(current_iv_df, on='pair2', rsuffix='2')

    res = res.rename(columns={'0':'iv1', '02':'iv2'})

    res.set_index(['pair1','pair2'], inplace=True)
    corr_df.index.names = ['pair1', 'pair2']
    res = res.join(corr_df)


    res = res.sort_values(by='zscore', ascending=False, key=abs)
    
    return res

# %%
def get_current_vrp_ratio_ranks(vrp_ratio_df: pd.DataFrame, rv_df: pd.DataFrame, corr_df: pd.DataFrame, stockpx_df: pd.DataFrame,
                         dte: int=30, strike_count: int=2, volume_lookback: int=1, dte_threshold: int=20, z_window: int=100, use_vrp_df: bool=False, vrp_df=None):



    tickers = np.unique(np.append(vrp_ratio_df.columns.to_frame()[0].values, vrp_ratio_df.columns.to_frame()[1].values))
    current_vrp_dict = {}
    
    if use_vrp_df:
        current_vrp_dict = vrp_df.iloc[-1,:].T.to_dict()
    else:
        for ticker in tqdm(tickers):
            print(ticker)
            current_vrp_dict[ticker] = get_current_iv(ticker, dte=dte, strike_count=strike_count, 
                                                      volume_lookback=volume_lookback, dte_threshold=dte_threshold)*100/rv_df[ticker][-1]
            clear_output()


    ratio_dict = {}
    zscore_dict = {}
    pctl_dict = {}
    beta_dict = {}
    beta_premium_dict = {}
    for pair in vrp_ratio_df.columns:
        ratio_dict[pair] = current_vrp_dict[pair[0]]/current_vrp_dict[pair[1]]
        zscore_dict[pair] = (ratio_dict[pair] - vrp_ratio_df[pair][-z_window:].mean())/(vrp_ratio_df[pair][-z_window:].std())
        pctl_dict[pair] = scipy.stats.percentileofscore(vrp_ratio_df[pair][~np.isnan(vrp_ratio_df[pair])], 
                                                                        ratio_dict[pair], 'weak')
        beta_dict[pair] = (abs(stockpx_df[pair[0]].pct_change()/stockpx_df[pair[1]].pct_change())).replace([np.inf,-np.inf], np.nan).mean()
        beta_premium_dict[pair] = abs(current_vrp_dict[pair[1]] - current_vrp_dict[pair[0]]*beta_dict[pair])


    current_ratio_df = pd.DataFrame.from_dict(ratio_dict, 'index', columns=['vrp_ratio'])
    zscore_df = pd.DataFrame.from_dict(zscore_dict, 'index', columns=['zscore'])
    pctl_df = pd.DataFrame.from_dict(pctl_dict, 'index', columns=['pctl'])
    if stockpx_df is not None:
        beta_df = pd.DataFrame.from_dict(beta_dict, 'index', columns=['beta'])
        beta_premium_df = pd.DataFrame.from_dict(beta_premium_dict, 'index', columns=['beta_premium'])

    if stockpx_df is not None:
        res = pd.concat([current_ratio_df, zscore_df, pctl_df, beta_df, beta_premium_df], axis=1, join='inner')
    else:
        res = pd.concat([current_ratio_df, zscore_df, pctl_df], axis=1, join='inner')
        
    res.index = pd.MultiIndex.from_tuples(res.index)
    res.index.names = ['pair1', 'pair2']
    res.reset_index(inplace=True)
    
    current_iv_df = pd.DataFrame.from_dict(current_vrp_dict, 'index')

    res = res.join(current_iv_df, on='pair1')
    res = res.join(current_iv_df, on='pair2', rsuffix='2')

    res = res.rename(columns={'0':'vrp1', '02':'vrp2'})

    res.set_index(['pair1','pair2'], inplace=True)
    corr_df.index.names = ['pair1', 'pair2']
    res = res.join(corr_df)


    res = res.sort_values(by='zscore', ascending=False, key=abs)
    
    return res

# %%
def plot_iv_ratios(ranks_df: pd.Series or pd.DataFrame, iv_ratio_df: pd.DataFrame or pd.Series, iv_df: pd.DataFrame or pd.Series, n:int=100,
                   z_window: int=100, interactive: bool=False, write_to_file: bool=True, title: str='IV30', use_df: bool=False):

    ranks_df = ranks_df[:n]
    iv_ratio_df = iv_ratio_df.T
    iv_ratio_df.index.names = ['pair1','pair2']


    if isinstance(ranks_df.index,pd.MultiIndex):
        pairs = (ranks_df.index.to_frame()['pair1'] + '/' + ranks_df.index.to_frame()['pair2'])


    if use_df:
        td = iv_df.index[-1]
        iv_mult=1
    else:
        td = datetime.today().replace(hour=0,minute=0,second=0,microsecond=0)
        iv_mult=100
    
    currivratios = ranks_df['iv_ratio'].rename(td)

    iv_ratio_df = iv_ratio_df.join(currivratios, how='inner').T

    
    currivs = pd.concat([ranks_df.reset_index()[['pair1','iv1']], 
                         ranks_df.reset_index()[['pair2','iv2']]
                         .rename(columns={'pair2':'pair1', 'iv2':'iv1'})]).drop_duplicates().set_index('pair1')
    
    currivs = (currivs.rename(columns={'iv1': td}).T*iv_mult).round(2)
    
    iv_df = pd.concat([iv_df, currivs], axis=0, join='inner')

    
    f = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.01)
    for pair in pairs:
        s1, s2 = pair.split('/')[0], pair.split('/')[1]
        mean = [iv_ratio_df[(s1,s2)][-z_window:].mean()]*len(iv_ratio_df.index)
        sd_upper_2 = mean + 2*iv_ratio_df[(s1,s2)][-z_window:].std()
        sd_lower_2 = mean - 2*iv_ratio_df[(s1,s2)][-z_window:].std()


        f.add_trace(go.Scatter(x=iv_ratio_df.index, y=iv_ratio_df[(s1,s2)], name=pair, visible=False,showlegend=False, line=dict(color='blue')), row=1,col=1)
        f.add_trace(go.Scatter(x=iv_ratio_df.index, y=mean, line=dict(color='black',dash='dash'), name='mean', visible=False, showlegend=False) , row=1,col=1)
        f.add_trace(go.Scatter(x=iv_ratio_df.index, y=sd_lower_2, opacity=0.3, line=dict(color='black',dash='dash'), name='sd_lower_2', visible=False, showlegend=False) , row=1,col=1)
        f.add_trace(go.Scatter(x=iv_ratio_df.index, y=sd_upper_2, opacity=0.3, line=dict(color='black',dash='dash'), name='sd_upper_2', visible=False, showlegend=False) , row=1,col=1)
        f.add_trace(go.Scatter(x=iv_ratio_df.index, y=iv_df[s1], name=s1, visible=False,showlegend=False, line=dict(color='red')), row=2,col=1)
        f.add_trace(go.Scatter(x=iv_ratio_df.index, y=iv_df[s2], name=s2, visible=False,showlegend=False, line=dict(color='orange')), row=2,col=1)
        f.add_scatter(x=[iv_ratio_df.index[-1]], y=[iv_ratio_df[(s1,s2)][-1]], mode='text', text=round(iv_ratio_df[(s1,s2)][-1],2), textposition='top right', 
                      hoverinfo='skip', visible=False, textfont=dict(color='blue'), row=1,col=1)
        f.add_scatter(x=[iv_ratio_df.index[-1]], y=[iv_df[s1][-1]], mode='text', text=iv_df[s1][-1], textposition='top right', 
                      hoverinfo='skip', visible=False, textfont=dict(color='red'), row=2,col=1)
        f.add_scatter(x=[iv_ratio_df.index[-1]], y=[iv_df[s2][-1]], mode='text', text=iv_df[s2][-1], textposition='top right', 
                      hoverinfo='skip', visible=False, textfont=dict(color='orange'), row=2,col=1)


    buttons = []

    ind = range(len(pairs)) 
    for i,pair in enumerate(pairs):
        t_ind = [i*9,i*9+1,i*9+2,i*9+3,i*9+4,i*9+5,i*9+6,i*9+7,i*9+8]
        t_ind_2 = [i*9,i*9+4,i*9+5]
        s1,s2 = pair.split('/')[0], pair.split('/')[1]
        try:
            corr = round(ranks_df.loc[(s1,s2),'r'],2)
        except KeyError:
            corr = round(ranks_df.loc[(s1,s2),'r(S)'],2)
        try:
            beta = round(ranks_df.loc[(s1,s2),'beta'],2)
        except KeyError:
            beta = None
        z = round(ranks_df.loc[(s1,s2), 'zscore'],2)
        pctl = int(round(ranks_df.loc[(s1,s2), 'pctl'], 0))
        buttons.append(
            dict(
                method='update',
                label = pair,
                visible=True,
                args=[
                    {'visible': [(i in t_ind) for i,x in enumerate(f.data)],
                     'showlegend': [(i in t_ind_2) for i,x in enumerate(f.data)]},
                     {'title': {'text': f'{pair} {title}: R = {corr}, Z = {z}, %tile = {pctl}, Beta = {beta}', 'y': 1.1, 'x': 0.8, 'xanchor': 'right', 'yanchor': 'top'}},
                ]
                    
            )
        )

    f.update_layout(updatemenus=[
        dict(type='dropdown',
            direction='right',
            y=1.1,
            xanchor='left',
            yanchor='top',
            showactive=False,
            buttons=buttons)], hovermode='x unified', width=1800, height=900,margin=dict(l=5,r=5,t=10,b=5))


    if interactive:
        f.show()
    
    if write_to_file:
        if not os.path.exists('IV_Plots'):
            os.mkdir('IV_Plots')
        f.write_html(f'IV_Plots/{title}_Ratios_{datetime.today().replace(microsecond=0)}.html')


# %%
def plot_vrp_ratios(ranks_df: pd.Series or pd.DataFrame, vrp_ratio_df: pd.DataFrame or pd.Series, vrp_df: pd.DataFrame or pd.Series, n:int=100,
                   z_window: int=100, interactive: bool=False, write_to_file: bool=True, title: str='VRP30', use_df: bool=False):

    ranks_df = ranks_df[:n]
    vrp_ratio_df = vrp_ratio_df.T
    vrp_ratio_df.index.names = ['pair1','pair2']


    if isinstance(ranks_df.index,pd.MultiIndex):
        pairs = (ranks_df.index.to_frame()['pair1'] + '/' + ranks_df.index.to_frame()['pair2'])


    if use_df:
        td = vrp_df.index[-1]
    else:
        td = datetime.today().replace(hour=0,minute=0,second=0,microsecond=0)
    
    currvrpratios = ranks_df['vrp_ratio'].rename(td)

    vrp_ratio_df = vrp_ratio_df.join(currvrpratios, how='inner').T

    
    currvrps = pd.concat([ranks_df.reset_index()[['pair1','vrp1']], 
                         ranks_df.reset_index()[['pair2','vrp2']]
                         .rename(columns={'pair2':'pair1', 'vrp2':'vrp1'})]).drop_duplicates().set_index('pair1')
    
    currvrps = (currvrps.rename(columns={'vrp1': td}).T).round(2)
    
    vrp_df = pd.concat([vrp_df, currvrps], axis=0, join='inner')

    
    f = make_subplots(rows=2,cols=1, shared_xaxes=True, vertical_spacing=0.01)
    for pair in pairs:
        s1, s2 = pair.split('/')[0], pair.split('/')[1]
        mean = [vrp_ratio_df[(s1,s2)][-z_window:].mean()]*len(vrp_ratio_df.index)
        sd_upper_2 = mean + 2*vrp_ratio_df[(s1,s2)][-z_window:].std()
        sd_lower_2 = mean - 2*vrp_ratio_df[(s1,s2)][-z_window:].std()


        f.add_trace(go.Scatter(x=vrp_ratio_df.index, y=vrp_ratio_df[(s1,s2)], name=pair, visible=False,showlegend=False, line=dict(color='blue')), row=1,col=1)
        f.add_trace(go.Scatter(x=vrp_ratio_df.index, y=mean, line=dict(color='black',dash='dash'), name='mean', visible=False, showlegend=False) , row=1,col=1)
        f.add_trace(go.Scatter(x=vrp_ratio_df.index, y=sd_lower_2, opacity=0.3, line=dict(color='black',dash='dash'), name='sd_lower_2', visible=False, showlegend=False) , row=1,col=1)
        f.add_trace(go.Scatter(x=vrp_ratio_df.index, y=sd_upper_2, opacity=0.3, line=dict(color='black',dash='dash'), name='sd_upper_2', visible=False, showlegend=False) , row=1,col=1)
        f.add_trace(go.Scatter(x=vrp_ratio_df.index, y=vrp_df[s1], name=s1, visible=False,showlegend=False, line=dict(color='red')), row=2,col=1)
        f.add_trace(go.Scatter(x=vrp_ratio_df.index, y=vrp_df[s2], name=s2, visible=False,showlegend=False, line=dict(color='orange')), row=2,col=1)
        f.add_scatter(x=[vrp_ratio_df.index[-1]], y=[vrp_ratio_df[(s1,s2)][-1]], mode='text', text=round(vrp_ratio_df[(s1,s2)][-1],2), textposition='top right', 
                      hoverinfo='skip', visible=False, textfont=dict(color='blue'), row=1,col=1)
        f.add_scatter(x=[vrp_ratio_df.index[-1]], y=[vrp_df[s1][-1]], mode='text', text=vrp_df[s1][-1], textposition='top right', 
                      hoverinfo='skip', visible=False, textfont=dict(color='red'), row=2,col=1)
        f.add_scatter(x=[vrp_ratio_df.index[-1]], y=[vrp_df[s2][-1]], mode='text', text=vrp_df[s2][-1], textposition='top right', 
                      hoverinfo='skip', visible=False, textfont=dict(color='orange'), row=2,col=1)


    buttons = []

    ind = range(len(pairs)) 
    for i,pair in enumerate(pairs):
        t_ind = [i*9,i*9+1,i*9+2,i*9+3,i*9+4,i*9+5,i*9+6,i*9+7,i*9+8]
        t_ind_2 = [i*9,i*9+4,i*9+5]
        s1,s2 = pair.split('/')[0], pair.split('/')[1]
        try:
            corr = round(ranks_df.loc[(s1,s2),'r(VRP)'],2)
        except KeyError:
            corr = round(ranks_df.loc[(s1,s2),'r(S)'],2)
        try:
            beta = round(ranks_df.loc[(s1,s2),'beta'],2)
        except KeyError:
            beta = None
        z = round(ranks_df.loc[(s1,s2), 'zscore'],2)
        pctl = int(round(ranks_df.loc[(s1,s2), 'pctl'], 0))
        buttons.append(
            dict(
                method='update',
                label = pair,
                visible=True,
                args=[
                    {'visible': [(i in t_ind) for i,x in enumerate(f.data)],
                     'showlegend': [(i in t_ind_2) for i,x in enumerate(f.data)]},
                     {'title': {'text': f'{pair} {title}: R = {corr}, Z = {z}, %tile = {pctl}, Beta = {beta}', 'y': 1.1, 'x': 0.8, 'xanchor': 'right', 'yanchor': 'top'}},
                ]
                    
            )
        )

    f.update_layout(updatemenus=[
        dict(type='dropdown',
            direction='right',
            y=1.1,
            xanchor='left',
            yanchor='top',
            showactive=False,
            buttons=buttons)], hovermode='x unified', width=1800, height=900,margin=dict(l=5,r=5,t=10,b=5))


    if interactive:
        f.show()
    
    if write_to_file:
        if not os.path.exists('IV_Plots'):
            os.mkdir('IV_Plots')
        f.write_html(f'IV_Plots/{title}_Ratios_{datetime.today().replace(microsecond=0)}.html')


# %%

def scrape_yahoo_screener(url: str):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options, desired_capabilities=capa)

    driver.get(url + '?offset=0&count=100')
    results = int(driver.find_element(By.CSS_SELECTOR, 'span[class="Mstart(15px) Fw(500) Fz(s)"]').text.split(' ')[-2])
    print(f'RESULTS: {results}')

    offset=0
    dfs = []
    while offset < results: 
        print(f'PAGE {int((offset+100)/100)} of {results//100 + 1}')
        driver.get(f'{url}?count=100&offset={offset}')
        el=driver.find_element(By.CSS_SELECTOR, 'div[class="Ovx(a) Ovx(h)--print Ovy(h) W(100%) "]')
        dfs.append(pd.read_html(el.get_attribute('innerHTML'))[0])

        offset+=100
        clear_output()
    driver.close()

    df = pd.concat(dfs)
    return df


# %%
def get_earnings_next_x_days(days: int=7):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options, desired_capabilities=capa)
    dayrange = pd.date_range(datetime.today(), datetime.today()+timedelta(days))
    start = datetime.strftime(dayrange[0],'%Y-%m-%d')
    end = datetime.strftime(dayrange[-1],'%Y-%m-%d')

    dfs_outer = []
    for day in tqdm(dayrange):
        day = datetime.strftime(day,'%Y-%m-%d')
        driver.get(f'https://finance.yahoo.com/calendar/earnings?from={start}&to={end}&day={day}')
        try:
            results = int(driver.find_element(By.CSS_SELECTOR, 'span[class="Mstart(15px) Fw(500) Fz(s)"]').text.split(' ')[-2])
        except selenium.common.exceptions.NoSuchElementException:
            continue

        offset=0
        dfs_inner = []
        while offset < results: 
            driver.get(f'https://finance.yahoo.com/calendar/earnings?from={start}&to={end}&day={day}&offset={offset}')
            try:
                el=driver.find_element(By.CSS_SELECTOR, 'div[class="Ovx(a) Ovx(h)--print Ovy(h) W(100%) "]')
                dfs_inner.append(pd.read_html(el.get_attribute('innerHTML'))[0])
            except selenium.common.exceptions.NoSuchElementException:
                continue
            
            offset+=100

        df_inner = pd.concat(dfs_inner)
        df_inner['earnings_date'] = day
        dfs_outer.append(df_inner)

    df = pd.concat(dfs_outer).set_index('Symbol')
    return df

# %%
def get_earnings_last_x_days(days: int=7):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options, desired_capabilities=capa)
    dayrange = pd.date_range(datetime.today()-timedelta(days), datetime.today())
    start = datetime.strftime(dayrange[0],'%Y-%m-%d')
    end = datetime.strftime(dayrange[-1],'%Y-%m-%d')

    dfs_outer = []
    for day in tqdm(dayrange):
        day = datetime.strftime(day,'%Y-%m-%d')
        driver.get(f'https://finance.yahoo.com/calendar/earnings?from={start}&to={end}&day={day}')
        try:
            results = int(driver.find_element(By.CSS_SELECTOR, 'span[class="Mstart(15px) Fw(500) Fz(s)"]').text.split(' ')[-2])
        except selenium.common.exceptions.NoSuchElementException:
            continue

        offset=0
        dfs_inner = []
        while offset < results: 
            driver.get(f'https://finance.yahoo.com/calendar/earnings?from={start}&to={end}&day={day}&offset={offset}')
            try:
                el=driver.find_element(By.CSS_SELECTOR, 'div[class="Ovx(a) Ovx(h)--print Ovy(h) W(100%) "]')
                dfs_inner.append(pd.read_html(el.get_attribute('innerHTML'))[0])
            except selenium.common.exceptions.NoSuchElementException:
                continue
            
            offset+=100

        df_inner = pd.concat(dfs_inner)
        df_inner['earnings_date'] = day
        dfs_outer.append(df_inner)

    df = pd.concat(dfs_outer).set_index('Symbol')
    return df

# %%
def get_available_tickers(yahoo_screen_url: str, etf_screen_url: str, earnings_days_forward: int=7, earnings_days_back: int=7, etf_limit: int=50):
    apikey = os.environ['ndl_api_key']
    r = requests.get(f'https://data.nasdaq.com/api/v3/databases/OPT/metadata?api_key={apikey}', stream=True)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall(path='')

    tickers_avail = pd.read_csv('OPT_metadata.csv', index_col=0)
    tickers_avail = tickers_avail[pd.to_datetime(tickers_avail['refreshed_at']) >= datetime.today()-BDay(2)]

    yf_screen = scrape_yahoo_screener(yahoo_screen_url)['Symbol'].to_list()

    if earnings_days_forward > 0:
        earnings_next_x = get_earnings_next_x_days(earnings_days_forward).index.to_list()
    else:
        earnings_next_x = []

    if earnings_days_back > 0:
        earnings_last_x = get_earnings_last_x_days(earnings_days_back).index.to_list()
    else:
        earnings_last_x = []

    ivticks = [x for x in yf_screen if x in tickers_avail.index and x not in earnings_next_x]  
    vrpticks = [x for x in yf_screen if x in tickers_avail.index and x not in earnings_last_x]

    if etf_limit > 0:
        etfs = scrape_yahoo_screener(etf_screen_url)['Symbol'].to_list()[:etf_limit]
        ivticks = ivticks + etfs
        vrpticks = vrpticks + etfs
        return list(ivticks), list(vrpticks)
    else:
        return list(ivticks), list(vrpticks)

# %%
def read_hist_iv_data_from_csv(path):
    dfdict = {}
    for filename in os.listdir(path):
        dfdict[filename.replace('.csv', '')] = pd.read_csv(os.path.join(path, filename), index_col=0)
        dfdict[filename.replace('.csv', '')].index = pd.to_datetime(dfdict[filename.replace('.csv', '')].index)

    return dfdict

# %%

def update_iv_csvs(basepath: str, rows=None, start_date = datetime.today()-timedelta(1), end_date = datetime.today(), keep: str='first'):


    dfdict = {}
    if rows is not None:
        for file in ['stockpx', 'iv_30', 'iv_60', 'iv_90']: 
            dfdict[file] = pd.read_csv(os.path.join(basepath, f'{file}.csv'), index_col=0)
            dfdict[file].index = pd.to_datetime(dfdict[file].index)
    else:
        for file in ['stockpx', 'iv_30', 'iv_60', 'iv_90']: 
            dfdict[file] = pd.read_csv(os.path.join(basepath, f'{file}.csv'), index_col=0)
            dfdict[file].index = pd.to_datetime(dfdict[file].index)
            dfdict[file] = dfdict[file].loc[:start_date - timedelta(1), :]

    if rows is not None:
        stockpx_append, iv30_append, iv60_append, iv90_append = get_hist_iv_data(list(dfdict['iv_30'].columns), rows=rows, write_to_file=False)
    else:
        stockpx_append, iv30_append, iv60_append, iv90_append = get_hist_iv_data(list(dfdict['iv_30'].columns), start_date=start_date, end_date=end_date, write_to_file=False)

    stockpx_df = pd.concat([dfdict['stockpx'], stockpx_append], axis=0)
    iv30_df = pd.concat([dfdict['iv_30'], iv30_append], axis=0)
    iv60_df = pd.concat([dfdict['iv_60'], iv60_append], axis=0)
    iv90_df = pd.concat([dfdict['iv_90'], iv90_append], axis=0)

    stockpx_df = stockpx_df.dropna(axis=0, how='all')
    iv30_df = iv30_df.dropna(axis=0, how='all')
    iv60_df = iv60_df.dropna(axis=0, how='all')
    iv90_df = iv90_df.dropna(axis=0, how='all')

    stockpx_df = stockpx_df[~stockpx_df.index.duplicated(keep=keep)].sort_index()
    iv30_df = iv30_df[~iv30_df.index.duplicated(keep=keep)].sort_index()
    iv60_df = iv60_df[~iv60_df.index.duplicated(keep=keep)].sort_index()
    iv90_df = iv90_df[~iv90_df.index.duplicated(keep=keep)].sort_index()


    stockpx_df.to_csv(f'{basepath}/stockpx.csv')
    iv30_df.to_csv(f'{basepath}/iv_30.csv')
    iv60_df.to_csv(f'{basepath}/iv_60.csv')
    iv90_df.to_csv(f'{basepath}/iv_90.csv')

    dfdict = {'stockpx': stockpx_df, 'iv_30': iv30_df, 'iv_60': iv60_df, 'iv_90': iv90_df}

    return dfdict



# %%
def update_and_plot_ratios(yahoo_screen_url: str, etf_screen_url: str, basepath: str, plot_vrp: bool=False, earnings_days_forward: int=7, earnings_days_back: int=7, rows: int=None, 
                            start_date=datetime.today()-timedelta(1), end_date=datetime.today(), 
                            corr_thresholds: tuple=(0.9,0.91,0.92,0.8,0.84,0.86), stockpx_threshold=2.0,  use_df=False, use_stockpx_corr=False, 
                            strike_count=2, volume_lookback: int=1, dte_threshold: int=20, z_window=100, plots_n=100, etf_limit: int=50, plot_titles=('IV30', 'IV60', 'IV90', 'VRP30', 'VRP60', 'VRP90')):

    #earnings file
    #https://www.barchart.com/stocks/earnings-within-7-days?viewName=main&orderBy=nextEarningsDate&orderDir=asc

    if plot_vrp:
        ivticks, vrpticks = get_available_tickers(yahoo_screen_url=yahoo_screen_url, etf_screen_url=etf_screen_url, earnings_days_forward=earnings_days_forward, 
                                                earnings_days_back=earnings_days_back, etf_limit=etf_limit)
    else:
        ivticks = get_available_tickers(yahoo_screen_url=yahoo_screen_url, etf_screen_url=etf_screen_url, earnings_days_forward=earnings_days_forward, 
                                              earnings_days_back=0, etf_limit=etf_limit)[0]

    if rows is not None:
        dfdict = update_iv_csvs(basepath, rows=rows)
    else:
        dfdict = update_iv_csvs(basepath, start_date=start_date, end_date=end_date)
    

    iv30_df = dfdict['iv_30']
    iv60_df = dfdict['iv_60']
    iv90_df = dfdict['iv_90']
    stockpx_df = dfdict['stockpx']

    rv_30 = get_rvs(stockpx_df,30)
    rv_60 = get_rvs(stockpx_df,60)
    rv_90 = get_rvs(stockpx_df,90)

    if plot_vrp:
        vrp_30 = get_vrps(stockpx_df, iv30_df, 30)
        vrp_60 = get_vrps(stockpx_df, iv60_df, 60)
        vrp_90 = get_vrps(stockpx_df, iv90_df, 90)


    highcorrs_30, IV_ratios_30 = get_correlated_ivs(iv30_df, stockpx_df, corr_thresholds[0], stockpx_threshold, 
                                                    use_iv_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers=ivticks)
    highcorrs_60, IV_ratios_60 = get_correlated_ivs(iv60_df, stockpx_df, corr_thresholds[1], stockpx_threshold, 
                                                    use_iv_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers=ivticks)
    highcorrs_90, IV_ratios_90 = get_correlated_ivs(iv90_df, stockpx_df, corr_thresholds[2], stockpx_threshold, 
                                                    use_iv_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers=ivticks)
    
    if plot_vrp:
        highvrpcorrs_30, VRP_ratios_30 = get_correlated_vrps(vrp_30, stockpx_df, corr_thresholds[3], stockpx_threshold,  
                                                            use_vrp_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers = vrpticks)
        highvrpcorrs_60, VRP_ratios_60 = get_correlated_vrps(vrp_60, stockpx_df, corr_thresholds[4], stockpx_threshold,  
                                                            use_vrp_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers = vrpticks)
        highvrpcorrs_90, VRP_ratios_90 = get_correlated_vrps(vrp_90, stockpx_df, corr_thresholds[5], stockpx_threshold,  
                                                            use_vrp_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers = vrpticks)
    
    ivranks_30 = get_current_iv_ratio_ranks(IV_ratios_30, highcorrs_30, stockpx_df, dte=30, strike_count=strike_count, 
                                            volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_iv_df=use_df, iv_df=iv30_df)
    ivranks_60 = get_current_iv_ratio_ranks(IV_ratios_60, highcorrs_60, stockpx_df, dte=60, strike_count=strike_count, 
                                            volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_iv_df=use_df, iv_df=iv60_df)
    ivranks_90 = get_current_iv_ratio_ranks(IV_ratios_90, highcorrs_90, stockpx_df, dte=90, strike_count=strike_count, 
                                            volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_iv_df=use_df, iv_df=iv90_df)

    if plot_vrp:
        vrpranks_30 = get_current_vrp_ratio_ranks(VRP_ratios_30, rv_30, highvrpcorrs_30, stockpx_df, dte=30, strike_count=strike_count, 
                                                  volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_vrp_df=use_df, vrp_df=vrp_30)
        vrpranks_60 = get_current_vrp_ratio_ranks(VRP_ratios_60, rv_60, highvrpcorrs_60, stockpx_df, dte=60, strike_count=strike_count, 
                                                  volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_vrp_df=use_df, vrp_df=vrp_60)
        vrpranks_90 = get_current_vrp_ratio_ranks(VRP_ratios_90, rv_90, highvrpcorrs_90, stockpx_df, dte=90, strike_count=strike_count, 
                                                  volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_vrp_df=use_df, vrp_df=vrp_90)
        
    plot_iv_ratios(ivranks_30, IV_ratios_30, iv30_df, n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[0], use_df=use_df)
    plot_iv_ratios(ivranks_60, IV_ratios_60, iv60_df, n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[1], use_df=use_df)
    plot_iv_ratios(ivranks_90, IV_ratios_90, iv90_df, n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[2], use_df=use_df)

    if plot_vrp:
        plot_vrp_ratios(vrpranks_30, VRP_ratios_30, vrp_30,  n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[3], use_df=use_df)
        plot_vrp_ratios(vrpranks_60, VRP_ratios_60, vrp_60,  n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[4], use_df=use_df)
        plot_vrp_ratios(vrpranks_90, VRP_ratios_90, vrp_90,  n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[5], use_df=use_df)

# %%
def plot_current_ratios(yahoo_screen_url: str, etf_screen_url: str, basepath: str, plot_vrp: bool=False, earnings_days_forward: int=7, earnings_days_back: int=7,
                        corr_thresholds: tuple=(0.9,0.91,0.92,0.8,0.84,0.86), stockpx_threshold=2.0, use_df=False, use_stockpx_corr=False, 
                        strike_count=2, volume_lookback: int=1, dte_threshold: int=20, z_window=100, plots_n=100, etf_limit: int=50, 
                        plot_titles=('IV30', 'IV60', 'IV90', 'VRP30', 'VRP60', 'VRP90')):

    #earnings file
    #https://www.barchart.com/stocks/earnings-within-7-days?viewName=main&orderBy=nextEarningsDate&orderDir=asc

    if plot_vrp:
        ivticks, vrpticks = get_available_tickers(yahoo_screen_url=yahoo_screen_url, etf_screen_url=etf_screen_url, earnings_days_forward=earnings_days_forward, 
                                              earnings_days_back=earnings_days_back, etf_limit=etf_limit)
    else:
        ivticks = get_available_tickers(yahoo_screen_url=yahoo_screen_url, etf_screen_url=etf_screen_url, earnings_days_forward=earnings_days_forward, 
                                              earnings_days_back=0, etf_limit=etf_limit)[0]


    dfdict = read_hist_iv_data_from_csv(basepath)
    

    iv30_df = dfdict['iv_30']
    iv60_df = dfdict['iv_60']
    iv90_df = dfdict['iv_90']
    stockpx_df = dfdict['stockpx']

    rv_30 = get_rvs(stockpx_df,30)
    rv_60 = get_rvs(stockpx_df,60)
    rv_90 = get_rvs(stockpx_df,90)

    if plot_vrp:
        vrp_30 = get_vrps(stockpx_df, iv30_df, 30)
        vrp_60 = get_vrps(stockpx_df, iv60_df, 60)
        vrp_90 = get_vrps(stockpx_df, iv90_df, 90)

    highcorrs_30, IV_ratios_30 = get_correlated_ivs(iv30_df, stockpx_df, corr_thresholds[0], stockpx_threshold, 
                                                    use_iv_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers=ivticks)
    highcorrs_60, IV_ratios_60 = get_correlated_ivs(iv60_df, stockpx_df, corr_thresholds[1], stockpx_threshold, 
                                                    use_iv_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers=ivticks)
    highcorrs_90, IV_ratios_90 = get_correlated_ivs(iv90_df, stockpx_df, corr_thresholds[2], stockpx_threshold, 
                                                    use_iv_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers=ivticks)
    
    if plot_vrp:
        highvrpcorrs_30, VRP_ratios_30 = get_correlated_vrps(vrp_30, stockpx_df, corr_thresholds[3], stockpx_threshold,  
                                                            use_vrp_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers = vrpticks)
        highvrpcorrs_60, VRP_ratios_60 = get_correlated_vrps(vrp_60, stockpx_df, corr_thresholds[4], stockpx_threshold,  
                                                            use_vrp_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers = vrpticks)
        highvrpcorrs_90, VRP_ratios_90 = get_correlated_vrps(vrp_90, stockpx_df, corr_thresholds[5], stockpx_threshold,  
                                                            use_vrp_df=use_df, use_stockpx_corr=use_stockpx_corr, tickers = vrpticks)
        
    
    ivranks_30 = get_current_iv_ratio_ranks(IV_ratios_30, highcorrs_30, stockpx_df, dte=30, strike_count=strike_count, 
                                            volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_iv_df=use_df, iv_df=iv30_df)
    ivranks_60 = get_current_iv_ratio_ranks(IV_ratios_60, highcorrs_60, stockpx_df, dte=60, strike_count=strike_count, 
                                            volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_iv_df=use_df, iv_df=iv60_df)
    ivranks_90 = get_current_iv_ratio_ranks(IV_ratios_90, highcorrs_90, stockpx_df, dte=90, strike_count=strike_count, 
                                            volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_iv_df=use_df, iv_df=iv90_df)
    
    if plot_vrp:
        vrpranks_30 = get_current_vrp_ratio_ranks(VRP_ratios_30, rv_30, highvrpcorrs_30, stockpx_df, dte=30, strike_count=strike_count, 
                                                  volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_vrp_df=use_df, vrp_df=vrp_30)
        vrpranks_60 = get_current_vrp_ratio_ranks(VRP_ratios_60, rv_60, highvrpcorrs_60, stockpx_df, dte=60, strike_count=strike_count, 
                                                  volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_vrp_df=use_df, vrp_df=vrp_60)
        vrpranks_90 = get_current_vrp_ratio_ranks(VRP_ratios_90, rv_90, highvrpcorrs_90, stockpx_df, dte=90, strike_count=strike_count, 
                                                  volume_lookback=volume_lookback, dte_threshold=dte_threshold, z_window=z_window, use_vrp_df=use_df, vrp_df=vrp_90)

    plot_iv_ratios(ivranks_30, IV_ratios_30, iv30_df, n=plots_n, z_window=z_window, interactive=False, write_to_file=True ,title=plot_titles[0], use_df=use_df)
    plot_iv_ratios(ivranks_60, IV_ratios_60, iv60_df, n=plots_n, z_window=z_window, interactive=False, write_to_file=True ,title=plot_titles[1], use_df=use_df)
    plot_iv_ratios(ivranks_90, IV_ratios_90, iv90_df, n=plots_n, z_window=z_window, interactive=False, write_to_file=True ,title=plot_titles[2], use_df=use_df)
    
    if plot_vrp:
        plot_vrp_ratios(vrpranks_30, VRP_ratios_30, vrp_30,  n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[3], use_df=use_df)
        plot_vrp_ratios(vrpranks_60, VRP_ratios_60, vrp_60,  n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[4], use_df=use_df)
        plot_vrp_ratios(vrpranks_90, VRP_ratios_90, vrp_90,  n=plots_n, z_window=z_window, interactive=False, write_to_file=True, title=plot_titles[5], use_df=use_df)


