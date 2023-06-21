# %%
import relative_value as rv
import requests
from urllib.parse import unquote
import pandas as pd
import numpy as np
import yfinance as yf
from pyetfdb_scraper import etf
from tqdm import tqdm
import itertools
from datetime import datetime, timedelta
from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore", module = 'pyetfdb_scraper')
warnings.filterwarnings("ignore", module = 'pandas')

# %%
def scrape_barchart(url: str, api_method: str, params: dict):
    geturl=url
    apiurl=f'https://www.barchart.com/proxies/core-api/v1/{api_method}'
    
    getheaders={
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'upgrade-insecure-requests': '1',
        "referer":url,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'
        }
    
    s=requests.Session()
    r=s.get(geturl, headers=getheaders)
    headers={
        'accept': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36",
        'x-xsrf-token': unquote(unquote(s.cookies.get_dict()['XSRF-TOKEN']))
    }
    
    r=s.get(apiurl,params=params,headers=headers)
    return pd.DataFrame(r.json()['data']).set_index('symbol')

# %%
def get_etf_holdings(yahoo_screen: str, n: int=200, volume_threshold: int=1_000_000):
    """ returns dict of dfs and list of all holdings"""

    etf_list = rv.scrape_yahoo_screener(yahoo_screen)['Symbol'].to_list()[:n]
    
    holdings_dict = {}
    all_holdings = []

    for ticker in tqdm(etf_list):
        print(ticker)
        bc_params = {
                "composite":ticker,
                 "fields":"symbol,symbolName,percent",
                 "orderBy":"percent","orderDir":"desc"
                 }
        init = etf.ETF(ticker)

        if (init.info['dbtheme']['data']['Asset Class']['text'] != 'Equity') or int(init.info['historical_trade_data']['data']['3 Month Avg. Volume'].replace(',','')) < volume_threshold:
            clear_output()
            continue
        else:
            try:
                holdings_dict[ticker] = scrape_barchart(f'https://www.barchart.com/stocks/quotes/{ticker}/constituents', 'EtfConstituents', bc_params)
                all_holdings.extend(holdings_dict[ticker].index.to_list())
            except KeyError:
                continue

        clear_output()

    all_holdings = [x for x in all_holdings if len(x) > 0]
    
    return holdings_dict, list(set(all_holdings))

# %%
def get_all_holdings_ivs(all_holdings: list, dte: int=30, strike_count: int=1, use_hist_data: bool=False, iv_df=None):
    holdings_iv_dict = {}
    if use_hist_data:
        all_holdings = [x for x in all_holdings if x in iv_df.columns]
        holdings_iv_dict = iv_df.loc[:,all_holdings].iloc[-1,:].to_dict()
        return holdings_iv_dict
 
    else:
        for holding in tqdm(all_holdings):
            print('GETTING IV: '+holding)
            try:
                holdings_iv_dict[holding] = rv.get_current_iv(holding, dte=dte, strike_count=strike_count)
            except KeyError:
                holdings_iv_dict[holding] = np.nan
            clear_output()
        return holdings_iv_dict

# %%
def get_all_etf_ivs(holdings_dict: dict, dte: int=30, strike_count: int=1, use_hist_data: bool=False, iv_df=None):
    etf_iv_dict = {}
    
    if use_hist_data:
        etfs = [x for x in list(holdings_dict.keys()) if x in iv_df.columns]
        etf_iv_dict = iv_df.loc[:,etfs].iloc[-1,:].to_dict()
    else:
        etfs = list(holdings_dict.keys())
        for ticker in tqdm(etfs):
            print('GETTING IV: '+ticker)
            try:
                etf_iv_dict[ticker] = rv.get_current_iv(ticker, dte=dte, strike_count=strike_count)
            except KeyError:
                continue
            clear_output()
    return etf_iv_dict

# %%
def get_all_holdings_corrs(stockpx_df: pd.DataFrame, corr_window: int=365):
    corrs = stockpx_df.iloc[-corr_window:, :].pct_change().corr()
    return corrs

# %%
def get_implied_corrs(holdings_dict: dict, holdings_iv_dict: dict, etf_iv_dict: dict, corr_df: pd.DataFrame, share_threshold: float):
    dispersion_dict = {}

    for ticker in tqdm(list(holdings_dict.keys())):
        if ticker not in list(etf_iv_dict.keys()):
            continue
        print(f'WORKING: {ticker}')
        df = holdings_dict[ticker][(holdings_dict[ticker].index.isin(holdings_iv_dict.keys())) & (holdings_dict[ticker].index != ticker)]
        if df.shape[0] == 0:
            continue

        for h in df.index:
            df.loc[h, 'current_iv'] = holdings_iv_dict[h]/100
            
        
        df = df[df['current_iv'] > 0]
        df['Share_num'] = pd.to_numeric(df['percent'].str.replace('%',''))/100
        total_share_represented = df['Share_num'].sum()
        if total_share_represented <= share_threshold:
            continue
        
        etf_iv = etf_iv_dict[ticker]/100
        

        term1 = ((df['Share_num']**2)*(df['current_iv']**2)).sum()
        term2 = 2*(sum([(df.loc[t1, 'Share_num']/total_share_represented)*(df.loc[t2, 'Share_num']/total_share_represented)*df.loc[t1, 'current_iv']*df.loc[t2, 'current_iv'] 
                                        for t1,t2 in itertools.combinations(df.index,2)]))
        term2wcorr = 2*(sum([(df.loc[t1, 'Share_num']/total_share_represented)*(df.loc[t2, 'Share_num']/total_share_represented)*df.loc[t1, 'current_iv']*df.loc[t2, 'current_iv']*corr_df.loc[t1,t2]
                                        for t1,t2 in itertools.combinations(df.index,2)]))

        etf_implied_corr = (etf_iv**2 - term1) / term2
        etf_expected_iv = (term1 + term2wcorr)**0.5
        etf_dispersion = ((df['Share_num']/total_share_represented)*(df['current_iv']**2)).sum() - etf_iv**2
        
        dispersion_dict[ticker] = [total_share_represented*100, etf_implied_corr*100, etf_dispersion, 
                                   etf_expected_iv*100, etf_iv*100, (etf_iv - etf_expected_iv)*100, list(df.index)]
        clear_output()

    dispersion_df = pd.DataFrame.from_dict(dispersion_dict, 'index', 
                                           columns = ['total_share_represented' ,'implied_corr', 'dispersion', 'expected_iv', 'current_iv', 'premium', 'holdings'])

    return dispersion_df



# %%
def get_ivs_and_implied_corrs(yahoo_screen: str, basepath: str, n: int=200, use_hist_data: bool=True, corr_window: int=365, volume_threshold: int=1_000_000, share_threshold: float=0.5, strike_count: int=1):
    
    dfdict = rv.read_hist_iv_data_from_csv(basepath)
    stockpx_df = dfdict['stockpx']
    iv30_df = dfdict['iv_30']
    iv60_df = dfdict['iv_60']
    iv90_df = dfdict['iv_90']
    corr_df = get_all_holdings_corrs(stockpx_df=stockpx_df,corr_window=corr_window)

    holdings_dict, all_holdings = get_etf_holdings(yahoo_screen, n, volume_threshold=volume_threshold)

    holdings_iv_dict_30 = get_all_holdings_ivs(all_holdings, dte=30, strike_count=strike_count, use_hist_data=use_hist_data, iv_df=iv30_df)
    holdings_iv_dict_60 = get_all_holdings_ivs(all_holdings, dte=60, strike_count=strike_count, use_hist_data=use_hist_data, iv_df=iv60_df)
    holdings_iv_dict_90 = get_all_holdings_ivs(all_holdings, dte=90, strike_count=strike_count, use_hist_data=use_hist_data, iv_df=iv90_df)

    etf_iv_dict_30 = get_all_etf_ivs(holdings_iv_dict_30, dte=30, strike_count=strike_count, use_hist_data=use_hist_data, iv_df=iv30_df)
    etf_iv_dict_60 = get_all_etf_ivs(holdings_iv_dict_60, dte=60, strike_count=strike_count, use_hist_data=use_hist_data, iv_df=iv60_df)
    etf_iv_dict_90 = get_all_etf_ivs(holdings_iv_dict_90, dte=60, strike_count=strike_count, use_hist_data=use_hist_data, iv_df=iv90_df)
    
    implied_corrs_30 = get_implied_corrs(holdings_dict, holdings_iv_dict_30, etf_iv_dict_30, corr_df, share_threshold=share_threshold)
    implied_corrs_60 = get_implied_corrs(holdings_dict, holdings_iv_dict_60, etf_iv_dict_60, corr_df, share_threshold=share_threshold)
    implied_corrs_90 = get_implied_corrs(holdings_dict, holdings_iv_dict_90, etf_iv_dict_90, corr_df, share_threshold=share_threshold)

    implied_corrs_30.to_csv(f'ETF_Implied_Corrs/implied_corrs_30_{datetime.today().replace(microsecond=0)}.csv')
    implied_corrs_60.to_csv(f'ETF_Implied_Corrs/implied_corrs_60_{datetime.today().replace(microsecond=0)}.csv')
    implied_corrs_90.to_csv(f'ETF_Implied_Corrs/implied_corrs_90_{datetime.today().replace(microsecond=0)}.csv')

    return implied_corrs_30, implied_corrs_60, implied_corrs_90




