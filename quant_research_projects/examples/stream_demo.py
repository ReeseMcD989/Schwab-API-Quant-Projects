import schwabdev
from dotenv import load_dotenv
import os
import time
import pandas as pd

# # Create an empty DataFrame to store streaming data
# columns = ["timestamp", "key", "1", "2", "3", "4", "5"]
# df = pd.DataFrame(columns=columns)

# def my_handler(message):
#     global df
#     print("test_handler:" + message)  # Keep the print for debugging
    
#     # Extract data from the message (this needs to be adapted to your message format)
#     try:
#         # Convert message string to dictionary
#         data = eval(message)  # Using eval here for simplicity, though json.loads() is recommended
        
#         if "data" in data:
#             for item in data["data"]:
#                 if "content" in item:
#                     # Each content item is a dictionary containing the values you want
#                     for content in item["content"]:
#                         # Extract timestamp and relevant fields from content
#                         timestamp = item["timestamp"]
#                         key = content.get("key", None)
#                         value_1 = content.get("1", None)
#                         value_2 = content.get("2", None)
#                         value_3 = content.get("3", None)
#                         value_4 = content.get("4", None)
#                         value_5 = content.get("5", None)

#                         # Append the new data to the DataFrame
#                         new_row = {
#                             "timestamp": timestamp,
#                             "key": key,
#                             "1": value_1,
#                             "2": value_2,
#                             "3": value_3,
#                             "4": value_4,
#                             "5": value_5
#                         }
#                         df = df.append(new_row, ignore_index=True)

#                         # Write DataFrame to CSV
#                         df.to_csv("streaming_data.csv", mode='w', header=True, index=False)

#     except Exception as e:
#         print(f"Error handling message: {e}")

def main():
    # place your app key and app secret in the .env file
    load_dotenv()  # load environment variables from .env file

    client = schwabdev.Client(os.getenv('app_key'), os.getenv('app_secret'), os.getenv('callback_url'))

    # define a variable for the steamer:
    streamer = client.stream

    """
    # example of using your own response handler, prints to main terminal.
    # the first parameter is used by the stream, additional parameters are passed to the handler
    def my_handler(message):
        print("test_handler:" + message)
    streamer.start(my_handler)
    """

    # start steamer with default response handler (print):
    streamer.start()

    """
    You can stream up to 500 keys.
    By default all shortcut requests (below) will be "ADD" commands meaning the list of symbols will be added/appended 
    to current subscriptions for a particular service, however if you want to overwrite subscription (in a particular 
    service) you can use the "SUBS" command. Unsubscribing uses the "UNSUBS" command. To change the list of fields use
    the "VIEW" command.
    """

    # these three do the same thing
    # streamer.send(streamer.basic_request("LEVELONE_EQUITIES", "ADD", parameters={"keys": "AMD,INTC", "fields": "0,1,2,3,4,5,6,7,8"}))
    # streamer.send(streamer.level_one_equities("AMD,INTC", "0,1,2,3,4,5,6,7,8", command="ADD"))
    # streamer.send(streamer.level_one_equities("AMD,INTC", "0,1,2,3,4,5,6,7,8"))


    # streamer.send(streamer.level_one_options("GOOGL 240712C00200000", "0,1,2,3,4,5,6,7,8")) # key must be from option chains api call.

    streamer.send(streamer.level_one_futures("/ES", "0,1,2,3,4,5,6"))

    # streamer.send(streamer.level_one_futures_options("./OZCZ23C565", "0,1,2,3,4,5"))

    # streamer.send(streamer.level_one_forex("EUR/USD", "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.nyse_book(["F", "NIO"], "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.nasdaq_book("AMD", "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.options_book("GOOGL 240712C00200000", "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.chart_equity("AMD", "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.chart_futures("/ES", "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.screener_equity("NASDAQ_VOLUME_30", "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.screener_options("OPTION_CALL_TRADES_30", "0,1,2,3,4,5,6,7,8"))

    # streamer.send(streamer.account_activity("Account Activity", "0,1,2,3"))


    # stop the stream after 60 seconds (since this is a demo)

    time.sleep(600)
    streamer.stop()
    # if you don't want to clear the subscriptions, set clear_subscriptions=False
    # streamer.stop(clear_subscriptions=False)


if __name__ == '__main__':
    print("Welcome to the unofficial Schwab interface!\nGithub: https://github.com/tylerebowers/Schwab-API-Python")
    main()  # call the user code above
