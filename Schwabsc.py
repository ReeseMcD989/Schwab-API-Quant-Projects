import tda
import tda.orders.equities as ord_eq
from tda.streaming import StreamClient
from tda.orders.common import Duration, Session
from apis import Gcpsc
from utils import drivers
from utils.time_handler import get_date_time_from_unix
import sys
import pandas as pd
import numpy as np
import utils.time_handler as th
import httpx
import math
import time
import json
from pathlib import Path
import urllib.parse
import requests
from requests.exceptions import HTTPError
from datetime import datetime,timezone,timedelta
sys.path.append('../')
from Quote import Quote
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

r = requests.get('https://www.google.com')

class Tdasc:
    """ A TD Ameritrade API object.
        Given a TD Ameritrade account
            1) Authenticate the given account
            2) Make a client available """
    def __init__(self,username,cred_path = None):
        self._username = username
        self._tda_account_id = -1
        self._cred_path = cred_path
        self._gcp_api = Gcpsc.Gcpsc()
        self._client = self.get_client(username)
        self.api_sleep = 0.001
        self.last_apicall = datetime.now()
        self.quote_test = -1
        self.valid_stream_quotes_metrics = [
                               'bid_price','ask_price','bid_size','ask_size'
                               ,'last_price','last_size'
                               ,'volatility', 'net_change','total_volume'
                                ,'quote_time_in_long','trade_time_in_long'
                                ]
        self._ticker = None
        self.quote = Quote(self._ticker)

    # def read_token(self):
    #     token = self._gcp_api.access_secret_version(
    #                               secret_id=f"tda_api_token_{self._username}",
    #                               version_id="latest")
    #     return token

    # def write_token(self,token, refresh_token=None):
    #     global IN_MEMORY_TOKEN
    #     IN_MEMORY_TOKEN = token

    def get_client(self,username):
        """ Get a TDA client
        """
        creds = self._gcp_api.access_secret_version(
                                  secret_id=f"tda_api_{username}",
                                  version_id="latest")
        client = tda.auth.client_from_access_functions(
            creds["API_KEY"],
            token_read_func=self.read_token,
            token_write_func=self.write_token)
        self._tda_account_id = creds["ACCOUNT_ID"]
        return client

    # def get_easy_credentials(self,credential_path = None):
    #     """Read TDA credentials file.
    #        This won't be in production as we use secrets manager.
    #        Useful for testing and getting refresh tokens for new accounts
    #     :param credential_path: Fully qualified path to credentials file(.json)
    #     :type credential_path: a string
    #     :raises Credentials_Not_Found: Credentials could not be found given this cred_id
    #     :return: A set of TDA specific parameters for authentication to the TDA API
    #     :rtype: Python Dict
    #     """
    #     if not bool(credential_path):
    #         # look for creds within app
    #         path = Path(f"./creds/tda_creds_{self._username}.json")
    #     else:
    #         path = Path(f"{credential_path}/tda_creds_{self._username}.json")

    #     with open(path) as json_file:
    #         creds = json.load(json_file)

    #     return creds

    # def get_easy_client(self,username):
    #     """Create an 'easy' TDA client
    #         An easy client
    #             1) gets credentials from local file
    #             2) reads and stores TDA tokens from local file
    #        Requires an account_id and tokens
    #     :raises None
    #     :return: A client object from the TDA API
    #     :rtype: A client object from the TDA API
    #     """
    #     #Get credentials for this TDA object instance.
    #     creds = self.get_easy_credentials(username)
    #     # The app saves the token in a .json file upon creation of the client, name that location
    #     TOKEN_PATH = "./creds/tda-token-{}.json".format(creds['CRED_ID'])
    #     # Currently using 'easy_client'
    #     client = tda.auth.easy_client(
    #         creds["API_KEY"],
    #         creds["REDIRECT_URI"],
    #         TOKEN_PATH,
    #         drivers.make_webdriver)
    #     return client, creds
    
    # #create a function to fill na with the appropriate data type
    # def fill_na(self,df):
    #     empty_date_time = datetime.fromtimestamp(0)
    #     df['quote_timestamp'] = df['quote_timestamp'].apply(lambda x: empty_date_time if pd.isnull(x) else x)
    #     df['trade_timestamp'] = df['trade_timestamp'].apply(lambda x: empty_date_time if pd.isnull(x) else x)
    #     df['quote_minute'] = df['quote_minute'].apply(lambda x: empty_date_time if pd.isnull(x) else x)
    #     df['trade_minute'] = df['trade_minute'].apply(lambda x: empty_date_time if pd.isnull(x) else x)
    #     return df
    
    # def process_streaming_message(self,message):
    #     # this is for standard streaming quotes format regardless of source
    #     quotes_dict = dict.fromkeys(self.quote.quote_columns)
    #     # We'll turn the dict into a dataframe before returning
    #     df_quote = self.quote.df_quotes.copy()
    #     df_quotes = self.quote.df_quotes.copy()

    #     try:
    #         # if the server time is sync'd to the atomic clock, we can measure latency
    #         message_time = datetime.fromtimestamp(int(message['timestamp'])/1000)
    #         machine_time = datetime.now(timezone.utc).astimezone().replace(tzinfo=None)
    #         for item in message['content']:
    #             # convert keys to lower case, as the standard quotes format is lower case
    #             item = {k.lower(): v for k, v in item.items()}
    #             # check for the existence of a ticker
    #             if not 'key' in item.keys():
    #                 #print(f"process_tda_message: no ticker in item.keys()")
    #                 continue
    #             ticker = item['key']
    #             # check for an empty ticker
    #             if not ticker:
    #                 continue
                
    #             if not 'last_price' in item.keys():
    #                 continue

    #             if not 'quote_time_in_long' in item.keys() and not 'trade_time_in_long' in item.keys():
    #                 continue

    #             # if we don't have required metrics, skip, otherwise fill the dict with data
    #             if float(item['last_price']) > 0:

    #                 quotes_dict['message_time'] = message_time
    #                 quotes_dict['machine_time'] = machine_time
    #                 quotes_dict['ticker'] = ticker
                    
    #                 # The item level data is sparse, so we need to check for the existence of a metric
    #                 for metric in item.keys():
    #                     # only add metrics we care about and ensure they are not null
    #                     if metric in self.valid_stream_quotes_metrics and item[metric] != '':
    #                         if metric == 'quote_time_in_long':
    #                             quotes_dict['quote_timestamp'] = datetime.fromtimestamp(int(item[metric])/1000)
    #                             quotes_dict['quote_minute'] = quotes_dict['quote_timestamp'].replace(second=0, microsecond=0)

    #                         elif metric == 'trade_time_in_long':
    #                             quotes_dict['trade_timestamp'] = datetime.fromtimestamp(int(item[metric])/1000)
    #                             quotes_dict['trade_minute'] = quotes_dict['trade_timestamp'].replace(second=0, microsecond=0)
    #                         else:
    #                             quotes_dict[metric] = item[metric]
                    
    #                 if not quotes_dict['quote_timestamp'] and quotes_dict['trade_timestamp']:
    #                     quotes_dict['quote_timestamp'] = quotes_dict['trade_timestamp']
    #                     quotes_dict['quote_minute'] = quotes_dict['trade_minute']

    #                 if quotes_dict['quote_timestamp'] and quotes_dict['last_price']:
    #                     df_quote = pd.DataFrame(quotes_dict, index=[0])
    #                     df_quotes = pd.concat([df_quotes, df_quote], ignore_index=True)
    #                 # clear quote level dict and df
    #                 df_quote = self.quote.df_quotes.copy()
    #                 quotes_dict = dict.fromkeys(self.quote.quote_columns)
            
    #         df_quotes.reset_index(inplace=True)
    #         df_quotes = self.fill_na(df_quotes)

    #     except Exception as e:
    #         print(f"process_streaming_message error: {e}")
    #         raise
    #     return df_quotes
    

    def get_quote(self, ticker):
        try:
            response = self._client.get_quote(ticker)
        except Exception as e:
            print(f"Tdasc.get_quote: {e}")
            pass
        assert response.status_code == httpx.codes.OK, response.raise_for_status()
        return response.json()
    
    def get_quotes(self, ticker_list):
        encoded_tickers = urllib.parse.quote(','.join(ticker_list))
        headers = {'Accept': 'application/json'}
        url = f"https://api.tdameritrade.com/v1/marketdata/quotes?apikey=6FG09A71NVWGAARUPMWNBCVO6VRVYOTS&symbol={encoded_tickers}"
        response = requests.get(url, headers=headers)
        #print(response.status_code)
        assert response.status_code == httpx.codes.OK, response.raise_for_status()
        time.sleep(self.api_sleep)
        return response.json()

    def get_candles_1min_ticker(self, ticker, start_datetime=None, end_datetime=None,need_extended_hours_data=None):
        """ Given a stock symbol, grab price history for the symbol. Intent is to write to Bigquery. Column order matters.

        :param ticker: a short string of alpha characters, typical 1-5 characters long
        :type ticker: a string of alpha characters, note special characters are not supported by TDA API
        ...
        :raises some kind of errors: we should check for http response codes and date conversions
        ...
        :return: a Pandas data frame of price history with following schema, columns ordered.
        ...   ticker, open, high, low, close, date_time(formatted as date), datetime_unix(10 or 13 digits?)
        :rtype: Pandas Dataframe
        """
        # # unpack kwargs
        # start_datetime = kwargs.['start_datetime']
        # end_datetime = kwargs.values()['end_datetime']
        # need_extended_hours_data = kwargs.values()['need_extended_hours_data']

        # get ticker data from api, returns (http_response, json object)
        response = self._client.get_price_history_every_minute(
            ticker,
            start_datetime = start_datetime,
            end_datetime= end_datetime,
            need_extended_hours_data = need_extended_hours_data
            )
        #print(response)
        #assert response.status_code == httpx.codes.OK, response.raise_for_status()
        # create a dataframe from json result --->
        #     with headers:  candles(dict of OHLCV), symbol(aka ticker), empty(boolean, data may not exist)
        df = pd.read_json(response)
        prices_dict = response.json()
        #print(prices_dict)
        df_candles = pd.DataFrame()
        # create new data frame with just the candles
        if response.status_code == httpx.codes.OK and not prices_dict['empty']:
            try:
                df_candles = df.from_dict(prices_dict['candles'])
                # add ticker and empty to candles
                df_candles['ticker'] = prices_dict['symbol']
                df_candles['ticker'] = df_candles['ticker'].values.astype(str)
                df_candles.rename({'datetime': 'datetime_unix'}, axis=1, inplace=True)
                df_candles['date_time'] = df_candles['datetime_unix'].apply(lambda x: th.get_date_time_from_unix(int(x/1000),'minute'))
                df_candles['date_time'] = pd.to_datetime(df_candles['date_time'])
                df_candles['volume'] = df_candles['volume'].astype(float)
                df_candles['datetime_unix'] = df_candles['datetime_unix'].astype(np.int64)
            except:
                pass
            return df_candles[['ticker','date_time','open','high','low','close','volume','datetime_unix']]
        else:
            return df_candles

    def get_current_quotes(self,loops):
        col_list = ['symbol', 'description', 'lastPrice', 'lastSize', 'bidPrice', 'bidSize', 'askPrice', 'askSize',
                    'quoteTimeInLong']
        df_return = pd.DataFrame(columns=col_list)
        df_return['date_time'] = None
        tickers_df = self.get_tickers()
        tickers_list = tickers_df['ticker'].tolist()
        ticker_cnt = len(tickers_list)
        chunk_size = 400
        chunks = math.ceil(ticker_cnt / chunk_size)
        #print(f"tickers to get quotes for: {ticker_cnt}")

        for i in range(0, loops):
            time.sleep(2)
            for i in range(0, chunks):
                chunk_of_tickers = tickers_df.iloc[i * chunk_size:i * chunk_size + chunk_size]['ticker'].tolist()
                r = self._client.get_quotes(chunk_of_tickers)
                assert r.status_code == httpx.codes.OK, r.raise_for_status()
                data = r.json()
                df = pd.DataFrame
                df = df.from_dict(data).T
                df = df[col_list]
                df['date_time'] = df['quoteTimeInLong'].apply(
                    lambda x: get_date_time_from_unix(int(x / 1000), 'second'))
                df_return = pd.concat([df_return, df])
                df_return.reset_index()
        time.sleep(self.api_sleep)
        return df_return

    @staticmethod
    def last_quote_in_df(df):
        df = df.sort_values(by=['symbol', 'quoteTimeInLong'], ascending=True)
        dfgroup = df.groupby(['symbol'], as_index=False).last()
        dfgroup.reset_index()
        return dfgroup

    @staticmethod
    def write_to_bq(df,table_name):
        df.to_gbq(destination_table=f"tda.{table_name}", if_exists='append')
    
    def get_balances(self):
        balances = self.get_account_info()
        time.sleep(self.api_sleep)
        return balances['securitiesAccount']['currentBalances']

    def get_positions(self):
        try:
            response = self._client.get_account(self._tda_account_id,fields=[tda.client.Client.Account.Fields.POSITIONS])
            assert response.status_code == httpx.codes.OK, response.raise_for_status()
        except Exception as e:
            print(f"Tdasc.get_positions: {e}")
            pass
        positions = response.json()
        time.sleep(self.api_sleep)
        return positions

    def get_transactions(self):
        response = self._client.get_transactions(self._tda_account_id, transaction_type=tda.client.Client.Transactions.TransactionType.TRADE)
        assert response.status_code == httpx.codes.OK, response.raise_for_status()
        transactions = response.json()
        time.sleep(self.api_sleep)
        return transactions
    
    def delay(self):
        api_call = sys._getframe(1).f_code.co_name
        now = datetime.now()        
        seconds_since_last_call = (now - self.last_apicall).total_seconds()
        if seconds_since_last_call < 0.5:
            print(f"{now}:  {api_call}: seconds_since_last_call: {seconds_since_last_call}: sleeping for {self.api_sleep} seconds")
            time.sleep(self.api_sleep)
            self.last_apicall = now
        else:
            print(f"{now}:  {api_call}: seconds_since_last_call: {seconds_since_last_call}: No delay needed")
            self.last_apicall = now

    def get_account_info(self):
        self.delay()
        account_dict={}
        fields_request = [tda.client.Client.Account.Fields.POSITIONS]
        try:
            response = self._client.get_account(self._tda_account_id,fields=fields_request)
            assert response.status_code == httpx.codes.OK, response.raise_for_status()
            account_dict = response.json()

        except Exception as e:
            print(f"{datetime.now()}:  {sys._getframe(0).f_code.co_name}: {e}")

        return account_dict
          

    def buy_market(self,ticker,quantity):
        """ Given a stock symbol and number of shares, place an order at TDA.
                :param ticker: Specify the equity to trade, typically 1-5 characters long
                :type ticker: a string of alpha characters, note special characters are not supported by TDA API
                ...
                :raises invalid http response codes
                ...
                :return: response,order_spec:  HTTP response from TDA api call, the order specification object specific to tda-api
                ...
                :rtype: http response object, python dict
                """
        retries = 1
        response = None
        order_spec = None
        successful_order = False
        self.delay()
        # for n in range(retries):
            #print(order_spec)
        try:

            order_spec = ord_eq.equity_buy_market(ticker, quantity).build()
            response = self._client.place_order(self._tda_account_id, order_spec=order_spec)
            print(f"{datetime.now()}: {sys._getframe(0).f_code.co_name}: {ticker}:{quantity}: response: ",response)
            response.raise_for_status()
            successful_order = True

        except (HTTPError,Exception) as e:
            print(f"{datetime.now()}: {sys._getframe(0).f_code.co_name}: {ticker}:{quantity}: {e}")
            print("sleeping for 0.1 seconds")
            time.sleep(0.1)
        return successful_order
    
    def sell_market(self,ticker,quantity):
        """ Given a stock symbol and number of shares, place an order at TDA.
            :param ticker: Specify the equity to trade, typically 1-5 characters long
            :type ticker: a string of alpha characters, note special characters are not supported by TDA API
            ...
            :raises invalid http response codes
            ...
            :return: response,order_spec:  HTTP response from TDA api call, the order specification object specific to tda-api
            ...
            :rtype: http response object, python dict
            """
        retries = 1
        response = None
        order_spec = None
        successful_order = False
        self.delay()

        # for n in range(retries):
            #print(order_spec)
        try:
            order_spec = ord_eq.equity_sell_market(ticker, quantity).build()
            response = self._client.place_order(self._tda_account_id, order_spec=order_spec)
            print(f"{datetime.now()}: {sys._getframe(1).f_code.co_name}: {ticker}:{quantity}: response: ",response)
            response.raise_for_status()
            successful_order = True

        except (HTTPError,Exception) as e:
            print(f"{datetime.now()}: {sys._getframe(1).f_code.co_name}: {ticker}:{quantity}: {e}")
            print("sleeping for 0.1 seconds")
            time.sleep(0.1)

        return successful_order
    
    def buy_limit(self,ticker,quantity,price):
        """ Given a stock symbol and number of shares, place an order at TDA.
                :param ticker: Specify the equity to trade, typically 1-5 characters long
                :type ticker: a string of alpha characters, note special characters are not supported by TDA API
                ...
                :raises invalid http response codes
                ...
                :return: response,order_spec:  HTTP response from TDA api call, the order specification object specific to tda-api
                ...
                :rtype: http response object, python dict
                """
        self.delay()
        order_spec = ord_eq.equity_buy_limit(ticker, quantity,price)\
            .set_duration(Duration.GOOD_TILL_CANCEL)\
            .set_session(Session.SEAMLESS)\
            .build()
        try:
            response = self._client.place_order(self._tda_account_id, order_spec=order_spec)
            assert response.status_code == httpx.codes.OK, response.raise_for_status()

        except (HTTPError,Exception) as e:
            print(f"{datetime.now()}: {sys._getframe(1).f_code.co_name}: {ticker}:{quantity}: {e}")
            time.sleep(2)
        return order_spec
    
    def sell_limit(self,ticker,quantity,price):
        """ Given a stock symbol and number of shares, place an order at TDA.
                :param ticker: Specify the equity to trade, typically 1-5 characters long
                :type ticker: a string of alpha characters, note special characters are not supported by TDA API
                ...
                :raises invalid http response codes
                ...
                :return: response,order_spec:  HTTP response from TDA api call, the order specification object specific to tda-api
                ...
                :rtype: http response object, python dict
                """
        self.delay()
        order_spec = ord_eq.equity_sell_limit(ticker, quantity,price)\
            .set_duration(Duration.GOOD_TILL_CANCEL)\
            .set_session(Session.SEAMLESS)\
            .build()
        try:
            response = self._client.place_order(self._tda_account_id, order_spec=order_spec)
            assert response.status_code == httpx.codes.OK, response.raise_for_status()
        except Exception as e:
            print(f"Tdasc.sell_limit: {e}")
            pass
        return response,order_spec
    def sell_short_market(self,ticker,quantity):
        """ Given a stock symbol and number of shares, place an order at TDA.
                :param ticker: Specify the equity to trade, typically 1-5 characters long
                :type ticker: a string of alpha characters, note special characters are not supported by TDA API
                ...
                :raises invalid http response codes
                ...
                :return: response,order_spec:  HTTP response from TDA api call, the order specification object specific to tda-api
                ...
                :rtype: http response object, python dict
                """
        self.delay()
        order_spec = ord_eq.equity_sell_short_market(ticker, quantity)\
            .set_session(Session.SEAMLESS)\
            .build()
        try:
            response = self._client.place_order(self._tda_account_id, order_spec=order_spec)
            assert response.status_code == httpx.codes.OK, response.raise_for_status()
        except Exception as e:
            print(f"Tdasc.sell_short_market: {e}")
            pass
        time.sleep(self.api_sleep)
        return response,order_spec
    def sell_short_limit(self,ticker,quantity,price):
        """ Given a stock symbol and number of shares, place an order at TDA.
                :param ticker: Specify the equity to trade, typically 1-5 characters long
                :type ticker: a string of alpha characters, note special characters are not supported by TDA API
                ...
                :raises invalid http response codes
                ...
                :return: response,order_spec:  HTTP response from TDA api call, the order specification object specific to tda-api
                ...
                :rtype: http response object, python dict
                """
        self.delay()
        order_spec = ord_eq.equity_sell_short_limit(ticker, quantity,price)\
            .set_duration(Duration.GOOD_TILL_CANCEL)\
            .set_session(Session.SEAMLESS)\
            .build()
        try:
            response = self._client.place_order(self._tda_account_id, order_spec=order_spec)
            assert response.status_code == httpx.codes.OK, response.raise_for_status()
        except Exception as e:
            print(f"Tdasc.sell_short_limit: {e}")
            pass
        time.sleep(self.api_sleep)
        return response,order_spec

    def get_orders(self):
        """ Given a stock symbol and number of shares, place an order at TDA.
                :param ticker: Specify the equity to trade, typically 1-5 characters long
                :type ticker: a string of alpha characters, note special characters are not supported by TDA API
                ...
                :raises invalid http response codes
                ...
                :return: response,order_spec:  HTTP response from TDA api call, the order specification object specific to tda-api
                ...
                :rtype: http response object, python dict
                """
        self.delay()
        try:
            response = self._client.get_orders_by_path(self._tda_account_id)
            assert response.status_code == httpx.codes.OK, response.raise_for_status()
        except Exception as e:
            print(f"Tdasc.sell_short_market: {e}")
            pass
        time.sleep(self.api_sleep)
        return response.json()
