# %%
import os
from dotenv import load_dotenv
import asyncio
import json
import yfinance as yf
import pandas as pd
from datetime import date, datetime, timedelta
import requests
from requests import JSONDecodeError
import zipfile
from io import BytesIO
from pandas.tseries.offsets import BDay
from tqdm import tqdm
from IPython.display import clear_output
import time
from bs4 import BeautifulSoup
import relative_value as rv



load_dotenv('e.env')

# %%
def get_pre_releases(yahoo_screen_url: str, next_x_days: int=7, write_to_file: bool=False):
    dfs = []
    
    tickers = rv.get_available_tickers(yahoo_screen_url=yahoo_screen_url, etf_screen_url=None, earnings_days_forward=0, earnings_days_back=0, etf_limit=0)[0]
    earnings_next_x = rv.get_earnings_next_x_days(next_x_days)[['earnings_date']]

    for ticker in tqdm(tickers):
        if ticker not in earnings_next_x.index:
            continue
        

        time.sleep(0.5)
        t = yf.Ticker(ticker)

        try:
            if len(t.get_news())==0:
                clear_output()
                continue
        except JSONDecodeError:
            clear_output()
            print(f'ERROR: {ticker}')
            continue

        try:
            news = pd.DataFrame(t.get_news())[['title', 'publisher','providerPublishTime','link']]
        except JSONDecodeError:
            clear_output()
            print(f'ERROR: {ticker}')
            continue
        news['providerPublishTime'] = (1_000_000_000*news['providerPublishTime']).apply(pd.Timestamp)
        news['ticker'] = ticker
        news = news[news['title'].str.lower().str.contains('preliminary')]
        try:
            if news.size == 0:
                clear_output()
                continue
        except JSONDecodeError:
            clear_output()
            print(f'ERROR: {ticker}')
            continue
        dfs.append(news)

        clear_output()

    if len(dfs) == 0:
        print('No Pre-Releases Found')
        return None
    
    
    pre_releases = pd.concat(dfs).set_index('ticker')

    pre_releases = pre_releases.join(earnings_next_x, how='left').rename(columns={'providerPublishTime': 'published'}).sort_values(by='earnings_date')
    pre_releases = pre_releases.drop_duplicates()
    if write_to_file:
        pre_releases.to_csv(f'pre_releases_{datetime.today().date()}.csv')

    return pre_releases

# %%
##########################
#MAIN
##########################

#pre_releases = get_pre_releases(volume_file='option_volume_gt1k.csv', next_x_days=2)


