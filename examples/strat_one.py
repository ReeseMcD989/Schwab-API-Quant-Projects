import schwabdev
from datetime import datetime, timedelta
from dotenv import load_dotenv
from time import sleep
import os

def main():
    # place your app key and app secret in the .env file
    load_dotenv()  # load environment variables from .env file

    client = schwabdev.Client(os.getenv('app_key'), os.getenv('app_secret'), os.getenv('callback_url'))
    # streamer = client.stream

    # # a "terminal emulator" to play with the API
    # print("\nTerminal emulator - enter python code to execute.")
    # while True:
    #     try:
    #         entered = input(">")
    #         exec(entered)
    #         print("[Succeeded]")
    #     except Exception as error:
    #         print("[ERROR]")
    #         print(error)

    print("\n\nAccounts and Trading - Accounts.")

    # get account number and hashes for linked accounts
    print("|\n|client.account_linked().json()", end="\n|")
    linked_accounts = client.account_linked().json()
    print(linked_accounts)

    # # this will get the first linked account
    account_hash = linked_accounts[0].get('hashValue')
    print(account_hash)
    # sleep(3)

    # get positions for linked accounts
    # print("|\n|client.account_details_all().json()", end="\n|")
    # print(client.account_details_all().json())
    # sleep(3)    

    order = {"orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": "SELL",
                    "quantity": 1,
                    "instrument": {
                        "symbol": "ENVX",
                        "assetType": "EQUITY"
                    }
                }
                ]
            }  

    # # client.order_place(account_hash, order)

    # resp = client.order_place(account_hash, order)
    # print("|\n|client.order_place(account_hash, order).json()", end="\n|")
    # print(f"Response code: {resp}")

    # # get the order ID - if order is immediately filled then the id might not be returned
    # order_id = resp.headers.get('location', '/').split('/')[-1] 
    # print(f"Order id: {order_id}")
    # sleep(3)

    # # get specific order details
    # print("|\n|client.order_details(account_hash, order_id).json()", end="\n|")
    # print(client.order_details(account_hash, order_id).json())
    # sleep(3)

if __name__ == '__main__':
    print("Welcome to the unofficial Schwab interface!\nGithub: https://github.com/tylerebowers/Schwab-API-Python")
    main()  # call the user code above
