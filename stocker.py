# Quandl for financial analysis, pandas and numpy for data manipulation
# fbprophet for additive models, #pytrends for Google trend data
import quandl
import pandas as pd
import numpy as np
import fbprophet
import datetime
#tkinter for GUI, messagebox for showing error messages
import tkinter as tk
from tkinter import messagebox
# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
import matplotlib
import sqlite3


#Connectivity to sql database to save user login data
connection=sqlite3.connect("login_master")
crsr=connection.cursor()
try:
    #create the table
    crsr.execute('create table login (user varchar(30) primary key,password varchar(30),money int(30))')
except:
    #table has already been created, can pass
    pass

# Class for analyzing and (attempting) to predict future prices
# Contains a number of visualizations and analysis methods
class Stocker():
    
    # Initialization requires a ticker symbol
    def __init__(self, ticker, exchange='WIKI'):
        
        # Enforce capitalization
        ticker = ticker.upper()
        
        # Symbol is used for labeling plots
        self.symbol = ticker
        
        # Use Personal Api Key
        quandl.ApiConfig.api_key = 'NEykTiDA4K6T1twrP8xA'

        # Retrieval the financial data
        try:
            stock = quandl.get('%s/%s' % (exchange, ticker))
            
            date_lst = []
            today = datetime.datetime.now()
            for i in range(len(stock)-1,-1,-1):
                prev = today - datetime.timedelta(days=i)
                date_lst.append(prev)
    
        except Exception as e:
            messagebox.showerror('Error Retrieving Data',"Could not recieve stock data. Please check your internet connection or the symbol entered")
            return None
##            print('Error Retrieving Data.')
##            print(e)
##            return
        
        # Set the index to a column called Date
        stock = stock.reset_index(level=0)
        stock['Date'] = np.array(date_lst)
        
        # Columns required for prophet
        stock['ds'] = stock['Date']

        if ('Adj. Close' not in stock.columns):
            stock['Adj. Close'] = stock['Close']
            stock['Adj. Open'] = stock['Open']
        
        stock['y'] = stock['Adj. Close']
        stock['Daily Change'] = stock['Adj. Close'] - stock['Adj. Open']
        
        # Data assigned as class attribute
        self.stock = stock.copy()
        
        # Minimum and maximum date in range
        self.min_date = min(stock['Date'])
        self.max_date = max(stock['Date'])
        
        # Find max and min prices and dates on which they occurred
        self.max_price = np.max(self.stock['y'])
        self.min_price = np.min(self.stock['y'])
        
        self.min_price_date = self.stock[self.stock['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.stock[self.stock['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]
        
        # The starting price (starting with the opening price)
        self.starting_price = float(self.stock.loc[0, 'Adj. Open'])
        
        # The most recent price
        self.most_recent_price = float(self.stock.loc[self.stock.index[-1], 'y'])

        # Whether or not to round dates
        self.round_dates = True
        
        # Number of years of data to train on
        self.training_years = 3

        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05 
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None
        
        print('{} Stocker Initialized. Data covers {} to {}.'.format(self.symbol,
                                                                     self.min_date,
                                                                     self.max_date))
    
    """
    Make sure start and end dates are in the range and can be
    converted to pandas datetimes. Returns dates in the correct format
    """
    def handle_dates(self, start_date, end_date):
        
        
        # Default start and end date are the beginning and end of data
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        
        try:
            # Convert to pandas datetime for indexing dataframe
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        
        except Exception as e:
            print('Enter valid pandas date format.')
            print(e)
            return
        
        valid_start = False
        valid_end = False
        
        # User will continue to enter dates until valid dates are met
        while (not valid_start) & (not valid_end):
            valid_end = True
            valid_start = True
            
            if end_date < start_date:
                print('End Date must be later than start date.')
                start_date = pd.to_datetime(input('Enter a new start date: '))
                end_date= pd.to_datetime(input('Enter a new end date: '))
                valid_end = False
                valid_start = False
            
            else: 
                if end_date > self.max_date:
                    print('End Date exceeds data range')
                    end_date= pd.to_datetime(input('Enter a new end date: '))
                    valid_end = False

                if start_date < self.min_date:
                    print('Start Date is before date range')
                    start_date = pd.to_datetime(input('Enter a new start date: '))
                    valid_start = False
                
        
        return start_date, end_date
        
    """
    Return the dataframe trimmed to the specified range.
    """
    def make_df(self, start_date, end_date, df=None):
        
        # Default is to use the object stock data
        if not df:
            df = self.stock.copy()
        
        
        start_date, end_date = self.handle_dates(start_date, end_date)
        
        # keep track of whether the start and end dates are in the data
        start_in = True
        end_in = True

        # If user wants to round dates (default behavior)
        if self.round_dates:
            # Record if start and end date are in df
            if (start_date not in list(df['Date'])):
                start_in = False
            if (end_date not in list(df['Date'])):
                end_in = False

            # If both are not in dataframe, round both
            if (not end_in) & (not start_in):
                trim_df = df[(df['Date'] >= start_date) & 
                             (df['Date'] <= end_date)]
            
            else:
                # If both are in dataframe, round neither
                if (end_in) & (start_in):
                    trim_df = df[(df['Date'] >= start_date) & 
                                 (df['Date'] <= end_date)]
                else:
                    # If only start is missing, round start
                    if (not start_in):
                        trim_df = df[(df['Date'] > start_date) & 
                                     (df['Date'] <= end_date)]
                    # If only end is imssing round end
                    elif (not end_in):
                        trim_df = df[(df['Date'] >= start_date) & 
                                     (df['Date'] < end_date)]

        
        else:
            valid_start = False
            valid_end = False
            while (not valid_start) & (not valid_end):
                start_date, end_date = self.handle_dates(start_date, end_date)
                
                # No round dates, if either data not in, print message and return
                if (start_date in list(df['Date'])):
                    valid_start = True
                if (end_date in list(df['Date'])):
                    valid_end = True
                    
                # Check to make sure dates are in the data
                if (start_date not in list(df['Date'])):
                    print('Start Date not in data (either out of range or not a trading day.)')
                    start_date = pd.to_datetime(input(prompt='Enter a new start date: '))
                    
                elif (end_date not in list(df['Date'])):
                    print('End Date not in data (either out of range or not a trading day.)')
                    end_date = pd.to_datetime(input(prompt='Enter a new end date: ') )

            # Dates are not rounded
            trim_df = df[(df['Date'] >= start_date) & 
                         (df['Date'] <= end_date.date)]

        
            
        return trim_df


    # Basic Historical Plots and Basic Statistics
    def plot_stock(self, start_date=None, end_date=None, stats=['Adj. Close'], plot_type='basic'):
        
        self.reset_plot()
        
        if start_date is None:
            start_date = self.min_date
        if end_date is None:
            end_date = self.max_date
        
        stock_plot = self.make_df(start_date, end_date)

        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        
        for i, stat in enumerate(stats):
            
            stat_min = min(stock_plot[stat])
            stat_max = max(stock_plot[stat])

            stat_avg = np.mean(stock_plot[stat])
            
            date_stat_min = stock_plot[stock_plot[stat] == stat_min]['Date']
            date_stat_min = date_stat_min[date_stat_min.index[0]]
            date_stat_max = stock_plot[stock_plot[stat] == stat_max]['Date']
            date_stat_max = date_stat_max[date_stat_max.index[0]]
            
            print('Maximum {} = {:.2f} on {}.'.format(stat, stat_max, date_stat_max))
            print('Minimum {} = {:.2f} on {}.'.format(stat, stat_min, date_stat_min))
            print('Current {} = {:.2f} on {}.\n'.format(stat, self.stock.loc[self.stock.index[-1], stat], self.max_date))
            
            # Percentage y-axis
            if plot_type == 'pct':
                # Simple Plot 
                plt.style.use('fivethirtyeight');
                if stat == 'Daily Change':
                    plt.plot(stock_plot['Date'], 100 * stock_plot[stat],
                         color = colors[i], linewidth = 2.4, alpha = 0.9,
                         label = stat)
                else:
                    plt.plot(stock_plot['Date'], 100 * (stock_plot[stat] -  stat_avg) / stat_avg,
                         color = colors[i], linewidth = 2.4, alpha = 0.9,
                         label = stat)

                plt.xlabel('Date'); plt.ylabel('Change Relative to Average (%)'); plt.title('%s Stock History' % self.symbol); 
                plt.legend(prop={'size':10})
                plt.grid(color = 'k', alpha = 0.4); 

            # Stat y-axis
            elif plot_type == 'basic':
                plt.style.use('fivethirtyeight');
                plt.plot(stock_plot['Date'], stock_plot[stat], color = colors[i], linewidth = 3, label = stat, alpha = 0.8)
                plt.xlabel('Date'); plt.ylabel('US $'); plt.title('%s Stock History' % self.symbol); 
                plt.legend(prop={'size':10})
                plt.grid(color = 'k', alpha = 0.4); 
      
        plt.show();
        
    # Reset the plotting parameters to clear style formatting
    def reset_plot(self):
        
        # Restore default parameters
        matplotlib.rcdefaults()
        
        # Adjust a few parameters to liking
        matplotlib.rcParams['figure.figsize'] = (8, 5)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'
    
    # Method to linearly interpolate prices on the weekends
    def resample(self, dataframe):
        # Change the index and resample at daily level
        dataframe = dataframe.set_index('ds')
        dataframe = dataframe.resample('D')
        
        # Reset the index and interpolate nan values
        dataframe = dataframe.reset_index(level=0)
        dataframe = dataframe.interpolate()
        return dataframe
    
    # Remove weekends from a dataframe
    def remove_weekends(self, dataframe):
        
        # Reset index to use ix
        dataframe = dataframe.reset_index(drop=True)
        
        weekends = []
        
        # Find all of the weekends
        for i, date in enumerate(dataframe['ds']):
            if (date.weekday()) == 5 | (date.weekday() == 6):
                weekends.append(i)
            
        # Drop the weekends
        dataframe = dataframe.drop(weekends, axis=0)
        
        return dataframe
    
    
    # Calculate and plot profit from buying and holding shares for specified date range
    def buy_and_hold(self, start_date=None, end_date=None, nshares=1):
        self.reset_plot()
        
        start_date, end_date = self.handle_dates(start_date, end_date)
            
        # Find starting and ending price of stock
        start_price = float(self.stock[self.stock['Date'] == start_date]['Adj. Open'])
        end_price = float(self.stock[self.stock['Date'] == end_date]['Adj. Close'])
        
        # Make a profit dataframe and calculate profit column
        profits = self.make_df(start_date, end_date)
        profits['hold_profit'] = nshares * (profits['Adj. Close'] - start_price)
        
        # Total profit
        total_hold_profit = nshares * (end_price - start_price)
        
        print('{} Total buy and hold profit from {} to {} for {} shares = ${:.2f}'.format
              (self.symbol, start_date, end_date, nshares, total_hold_profit))
        
        # Plot the total profits 
        plt.style.use('dark_background')
        
        # Location for number of profit
        text_location = (end_date - pd.DateOffset(months = 1))
        
        # Plot the profits over time
        plt.plot(profits['Date'], profits['hold_profit'], 'b', linewidth = 3)
        plt.ylabel('Profit ($)'); plt.xlabel('Date'); plt.title('Buy and Hold Profits for {} {} to {}'.format(
                                                                self.symbol, start_date, end_date))
        
        # Display final value on graph
        plt.text(x = text_location, 
             y =  total_hold_profit + (total_hold_profit / 40),
             s = '$%d' % total_hold_profit,
            color = 'g' if total_hold_profit > 0 else 'r',
            size = 14)
        
        plt.grid(alpha=0.2)
        plt.show();
        
    # Create a prophet model without training
    def create_model(self):

        # Make the model
        model = fbprophet.Prophet(daily_seasonality=self.daily_seasonality,  
                                  weekly_seasonality=self.weekly_seasonality, 
                                  yearly_seasonality=self.yearly_seasonality,
                                  changepoint_prior_scale=self.changepoint_prior_scale,
                                  changepoints=self.changepoints)
        
        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        
        return model
   
    # Basic prophet model for specified number of days  
    def create_prophet_model(self, days=0, resample=False):
        
        self.reset_plot()
        
        model = self.create_model()
        
        # Fit on the stock history for self.training_years number of years
        stock_history = self.stock[self.stock['Date'] > (self.max_date - pd.DateOffset(years = self.training_years))]
        
        if resample:
            stock_history = self.resample(stock_history)
        
        model.fit(stock_history)
        
        # Make and predict for next year with future dataframe
        future = model.make_future_dataframe(periods = days, freq='D')
        future = model.predict(future)
        
        if days > 0:
            # Print the predicted price
            print('Predicted Price on {} = ${:.2f}'.format(
                future.loc[future.index[-1], 'ds'], future.loc[future.index[-1], 'yhat']))

            title = '%s Historical and Predicted Stock Price'  % self.symbol
        else:
            title = '%s Historical and Modeled Stock Price' % self.symbol
        
        # Set up the plot
        fig, ax = plt.subplots(1, 1)

        # Plot the actual values
        ax.plot(stock_history['ds'], stock_history['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
        
        # Plot the predicted values
        ax.plot(future['ds'], future['yhat'], 'forestgreen',linewidth = 2.4, label = 'Modeled');

        # Plot the uncertainty interval as ribbon
        ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.3, 
                       facecolor = 'g', edgecolor = 'k', linewidth = 1.4, label = 'Confidence Interval')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10}); plt.xlabel('Date'); plt.ylabel('Price $');
        plt.grid(linewidth=0.6, alpha = 0.6)
        plt.title(title);
        plt.show()
        
        return model, future
      
    # Evaluate prediction model for one year
    def get_cost(self,nshares = 1):
        #start_date = datetime.datetime.now().date() - pd.DateOffset(years=1)
        print(self.stock.tail(1)['Close'])
        for i in self.stock.tail(1)['Close']:
            cost_1 = int(i)
        return cost_1*nshares
        
    def evaluate_prediction(self, start_date=None, end_date=None, nshares = None):
        
        # Default start date is one year before end of data
        # Default end date is end date of data
        if start_date is None:
            start_date = self.max_date - pd.DateOffset(years=1)
        if end_date is None:
            end_date = self.max_date
            
        start_date, end_date = self.handle_dates(start_date, end_date)
        
        # Training data starts self.training_years years before start date and goes up to start date
        train = self.stock[(self.stock['Date'] < start_date) & 
                           (self.stock['Date'] > (start_date - pd.DateOffset(years=self.training_years)))]
        
        # Testing data is specified in the range
        test = self.stock[(self.stock['Date'] >= start_date) & (self.stock['Date'] <= end_date)]
        
        # Create and train the model
        model = self.create_model()
        model.fit(train)
        
        # Make a future dataframe and predictions
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)
        
        # Merge predictions with the known values
        test = pd.merge(test, future, on = 'ds', how = 'inner')

        train = pd.merge(train, future, on = 'ds', how = 'inner')
        
        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()

        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff'][1:]) == np.sign(test['real_diff'][1:])) * 1
        
        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])

        # Calculate mean absolute error
        test_errors = abs(test['y'] - test['yhat'])
        test_mean_error = np.mean(test_errors)

        train_errors = abs(train['y'] - train['yhat'])
        train_mean_error = np.mean(train_errors)

        # Calculate percentage of time actual value within prediction range
        test['in_range'] = False

        for i in test.index:
            if (test.loc[i, 'y'] < test.loc[i, 'yhat_upper']) & (test.loc[i, 'y'] > test.loc[i, 'yhat_lower']):
                test.loc[i, 'in_range'] = True

        in_range_accuracy = 100 * np.mean(test['in_range'])

        if not nshares:

            # Date range of predictions
            print('\nPrediction Range: {} to {}.'.format(start_date,
                end_date))

            # Final prediction vs actual value
            print('\nPredicted price on {} = ${:.2f}.'.format(max(future['ds']), future.loc[future.index[-1], 'yhat']))
            print('Actual price on    {} = ${:.2f}.\n'.format(max(test['ds']), test.loc[test.index[-1], 'y']))

            print('Average Absolute Error on Training Data = ${:.2f}.'.format(train_mean_error))
            print('Average Absolute Error on Testing  Data = ${:.2f}.\n'.format(test_mean_error))

            # Direction accuracy
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            print('The actual value was within the {:d}% confidence interval {:.2f}% of the time.'.format(int(100 * model.interval_width), in_range_accuracy))


             # Reset the plot
            self.reset_plot()
            
            # Set up the plot
            fig, ax = plt.subplots(1, 1)

            # Plot the actual values
            ax.plot(train['ds'], train['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            ax.plot(test['ds'], test['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            
            # Plot the predicted values
            ax.plot(future['ds'], future['yhat'], 'navy', linewidth = 2.4, label = 'Predicted');

            # Plot the uncertainty interval as ribbon
            ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.6, 
                           facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Confidence Interval')

            # Put a vertical line at the start of predictions
            plt.vlines(x=min(test['ds']), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                       linestyles='dashed', label = 'Prediction Start')

            # Plot formatting
            plt.legend(loc = 2, prop={'size': 8}); plt.xlabel('Date'); plt.ylabel('Price $');
            plt.grid(linewidth=0.6, alpha = 0.6)
                       
            plt.title('{} Model Evaluation from {} to {}.'.format(self.symbol,
                start_date, end_date));
            plt.show();

        
        # If a number of shares is specified, play the game
        elif nshares:
            
            # Only playing the stocks when we predict the stock will increase
            test_pred_increase = test[test['pred_diff'] > 0]
            
            test_pred_increase.reset_index(inplace=True)
            prediction_profit = []
            
            # Iterate through all the predictions and calculate profit from playing
            for i, correct in enumerate(test_pred_increase['correct']):
                
                # If we predicted up and the price goes up, we gain the difference
                if correct == 1:
                    prediction_profit.append(nshares * test_pred_increase.loc[i, 'real_diff'])
                # If we predicted up and the price goes down, we lose the difference
                else:
                    prediction_profit.append(nshares * test_pred_increase.loc[i, 'real_diff'])
            
            test_pred_increase['pred_profit'] = prediction_profit
            
            # Put the profit into the test dataframe
            test = pd.merge(test, test_pred_increase[['ds', 'pred_profit']], on = 'ds', how = 'left')
            test.loc[0, 'pred_profit'] = 0
        
            # Profit for either method at all dates
            test['pred_profit'] = test['pred_profit'].cumsum().ffill()
            test['hold_profit'] = nshares * (test['y'] - float(test.loc[0, 'y']))
            
            # Display information
            print('You played the stock market in {} from {} to {} with {} shares.\n'.format(
                self.symbol, start_date, end_date, nshares))
            
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            # Display some friendly information about the perils of playing the stock market
            print('The total profit using the Prophet model = ${:.2f}.'.format(np.sum(prediction_profit)))
            print('The Buy and Hold strategy profit =         ${:.2f}.'.format(float(test.loc[test.index[-1], 'hold_profit'])))
            print('\nThanks for playing the stock market!\n')
            
           
            
            # Plot the predicted and actual profits over time
            self.reset_plot()
            
            # Final profit and final smart used for locating text
            final_profit = test.loc[test.index[-1], 'pred_profit']
            final_smart = test.loc[test.index[-1], 'hold_profit']

            # text location
            last_date = test.loc[test.index[-1], 'ds']
            text_location = (last_date - pd.DateOffset(months = 1))

            plt.style.use('dark_background')

            # Plot smart profits
            plt.plot(test['ds'], test['hold_profit'], 'b',
                     linewidth = 1.8, label = 'Buy and Hold Strategy') 

            # Plot prediction profits
            plt.plot(test['ds'], test['pred_profit'], 
                     color = 'g' if final_profit > 0 else 'r',
                     linewidth = 1.8, label = 'Prediction Strategy')

            # Display final values on graph
            plt.text(x = text_location, 
                     y =  final_profit + (final_profit / 40),
                     s = '$%d' % final_profit,
                    color = 'g' if final_profit > 0 else 'r',
                    size = 18)
            
            plt.text(x = text_location, 
                     y =  final_smart + (final_smart / 40),
                     s = '$%d' % final_smart,
                    color = 'g' if final_smart > 0 else 'r',
                    size = 18);

            # Plot formatting
            plt.ylabel('Profit  (US $)'); plt.xlabel('Date'); 
            plt.title('Predicted versus Buy and Hold Profits');
            plt.legend(loc = 2, prop={'size': 10});
            plt.grid(alpha=0.2); 
            plt.show(block=False)
            #returning profit from Prophet, profit from Buy-n-Hold
            return(np.sum(prediction_profit),float(test.loc[test.index[-1], 'hold_profit']))
    # Predict the future price for a given range of days
    def predict_future(self, days=30):
        
        # Use past self.training_years years for training
        train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=self.training_years))]
        
        model = self.create_model()
        
        model.fit(train)
        
        # Future dataframe with specified number of days to predict
        future = model.make_future_dataframe(periods=days, freq='D')
        future = model.predict(future)
        
        # Only concerned with future dates
        future = future[future['ds'] >= max(self.stock['Date'])]
        
        # Remove the weekends
        future = self.remove_weekends(future)
        
        # Calculate whether increase or not
        future['diff'] = future['yhat'].diff()
    
        future = future.dropna()

        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff'] > 0) * 1
        
        # Rename the columns for presentation
        future = future.rename(columns={'ds': 'Date', 'yhat': 'estimate', 'diff': 'change', 
                                        'yhat_upper': 'upper', 'yhat_lower': 'lower'})
        
        future_increase = future[future['direction'] == 1]
        future_decrease = future[future['direction'] == 0]
        
        # Print out the dates
        print('\nPredicted Increase: \n')
        print(future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        print('\nPredicted Decrease: \n')
        print(future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])
        
        self.reset_plot()
        
        # Set up plot
        plt.style.use('fivethirtyeight')
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 12
        
        # Plot the predictions and indicate if increase or decrease
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot the estimates
        ax.plot(future_increase['Date'], future_increase['estimate'], 'g^', ms = 12, label = 'Pred. Increase')
        ax.plot(future_decrease['Date'], future_decrease['estimate'], 'rv', ms = 12, label = 'Pred. Decrease')

        # Plot errorbars
        ax.errorbar(future['Date'].dt.to_pydatetime(), future['estimate'], 
                    yerr = future['upper'] - future['lower'], 
                    capthick=1.4, color = 'k',linewidth = 2,
                   ecolor='darkblue', capsize = 4, elinewidth = 1, label = 'Pred with Range')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10});
        plt.xticks(rotation = '45')
        plt.ylabel('Predicted Stock Price (US $)');
        plt.xlabel('Date'); plt.title('Predictions for %s' % self.symbol);
        plt.show()
#Function to change the money stored in the database for the user
def add_money(user,money):
    connection.execute("""update login set money=money + ? where user= ?""",(money,user,))
    connection.commit()

#Function to get the current amount stored in the database for a perticular user
def get_user_balance(username):
    a=crsr.execute("select money from login where user=?",(username,)).fetchall()
    money = a[0][0]
    return money

#Function called when a user is created or logged in
def manage_user(user,password="",login=0,change=0,new_password=""):
    if (user,) not in connection.execute("select user from login").fetchall() and login==0:
        #Trying to log in with a username that doesn't exist in the database
        messagebox.showerror('Error!','This user doesn\'t exist! Please create a user, then try to log in')
        return False
    else:
        if change==1:
            #changing password of the user
            connection.execute("""update login set password= ? where user= ?""",(new_password,user,))
            connection.commit()
        else:
            #Create a new user, or login
            if login==0:
                #Login with username and password
                a=crsr.execute("select password from login where user=?",(user,)).fetchall()
                if password==a[0][0]:
                    #logged in sucsessfully
                    return True
                else:
                    #Wrong password entered
                    messagebox.showerror('Error!','Wrong password was entered!')
                    return False
            elif login==1:
                #Create a new user with a password and $25000
                try:
                    #Create user
                    connection.execute("insert into login values(?,?,?)",(user,password,25000))
                    connection.commit()
                except:
                    #Current username already exists in the user database
                    messagebox.showerror('Error!','Username is taken, please choose another one')

#Function to clear the screen without creating a new one
def clear_window(window):
    for ele in window.winfo_children():
        ele.destroy()

#Display Help screen of the "Play the Market" Game
def help_screen(username):
    global root
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Rules of the Game', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    #Rules of the game
    rule_str = ''
    rule_str += 'Thanks for playing this game. The rules for this game is simple.\n To play the game, an account is required to be created.\n'
    rule_str += 'An amount of money ($25000) is automatically added to your account\n when you create the account. This money is required\n'
    rule_str += 'to buy stocks of the company. The game "plays" the market for a\n period of one year (ranging from last year to this date). The profit or loss incurred\n'
    rule_str += 'by these transactions are added to your account. \n\nYou can choose strategies in which the game can play the market for you. Strategy One is called the\n'
    rule_str += 'Buy and Hold Strategy. In this strategy, you buy n number of shares\n and hold them for a period of one year, then sell them at the end of the year.\n'
    rule_str += 'The profits from this transaction is added to your account.\n\n'
    rule_str += 'The second strategy is the Prophet Predictive Strategy.\n In this strategy, a predictive model, at every interval, predicts either a rise in the stock\n'
    rule_str += 'price, or a drop in the stock price. If a rise is predicted,\n stocks are bought. If the model predicts a decrease, it doesn\'t play the market\n'
    rule_str += 'If the model predicted an increase and the stock does increase, you \nrecieve the change in price of the stock over that day times the number of shares.\n'
    rule_str += 'If the model predicted an increase and the stock decreases, you lose the \nchange in price of the stock over that day times the number of shares.\n'
    rule_str += 'The net profits or losses are added to your account'
    Text_Label = tk.Label(root, text = rule_str, font = 'Ariel 15 bold')#, pady = 20)
    Text_Label.pack()
    #Function for back button
    def back_button():
        game(username)
    Back_Button = tk.Button(root, text = 'Back', font = 'Ariel 25 bold', command = back_button, pady = 15, width = 29)
    Back_Button.pack()

#Display the game screen, with the appropriate user details
def game(username):
    global root
    clear_window(root)
    money = get_user_balance(username)
    #Strategy ONE
    def play_the_market_buynhold():
        symbol = Symbol_Entry.get()
        stock = Stocker(symbol)
        if stock == None:
            return
        try:
            stocks = int(Stock_Entry.get())
        except:
            messagebox.showerror('Error!','Enter number of stocks as an integer')
            return
        cost = stock.get_cost(stocks)
        if money > cost:
            a,b = stock.evaluate_prediction(nshares=stocks)
            a,b = int(a),int(b)
            profit = b
            add_money(username,profit)
            game(username)
            messagebox.showinfo('Done','You played the market with the Buy-n-Hold strategy. You earned a net profit/loss of $'+str(b)+' amount. If you played the Prophet Predictive Strategy, you would have gained $'+str(a))
        else:
            messagebox.showerror('Error!','Stocks Cost too much, you cannot afford it.')
    #Strategy TWO
    def play_the_market_prophet():
        symbol = Symbol_Entry.get()
        stock = Stocker(symbol)
        if stock == None:
            return
        try:
            stocks = int(Stock_Entry.get())
        except:
            messagebox.showerror('Error!','Enter number of stocks as an integer')
            return
        cost = stock.get_cost(stocks)
        if money > cost:
            a,b = stock.evaluate_prediction(nshares=stocks)
            a,b = int(a),int(b)
            profit = a
            add_money(username,profit)
            game(username)
            messagebox.showinfo('Done','You played the market with the Prophet Predictive strategy. You earned a net profit/loss of $'+str(a)+' amount. If you played the Buy-n-Hold Strategy, you would have gained $'+str(b))
        else:
            messagebox.showerror('Error!','Stocks Cost too much, you cannot afford it.')

    #Function for back button
    def back_button():
        game_login()
    #Function for the button to call the function that displays help page
    def help_button():
        help_screen(username)
    #GUI definition
    Title_Label = tk.Label(root, text = 'Play the Market', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Text_Label = tk.Label(root, text = 'Welcome, '+username, font = 'Ariel 20 bold')#, pady = 20)
    Text_Label.pack()
    Money_Label = tk.Label(root, text = 'Balance = $'+str(money), font = 'Ariel 20 bold')#, pady = 20)
    Money_Label.pack()
    Frame1 = tk.Frame(root, pady = 20)
    Frame1.pack()
    Symbol_label = tk.Label(Frame1, pady = 15, padx = 10, text = 'Enter Company\'s symbol',font='Ariel 25 bold')
    Symbol_label.grid(row = 0,column = 0)
    Symbol_Entry = tk.Entry(Frame1, font = 'Ariel 25 bold')
    Symbol_Entry.grid(row = 0, column = 1)
    Stock_label = tk.Label(Frame1, text = 'Enter the number of stocks \nyou want to buy',font='Ariel 25 bold')
    Stock_label.grid(row = 1,column = 0)
    Stock_Entry = tk.Entry(Frame1, font = 'Ariel 25 bold')
    Stock_Entry.grid(row = 1, column = 1)
    Help_Button = tk.Button(root, text = 'Rules of the game', font = 'Ariel 25 bold', command = help_button, pady = 15, width = 29)
    Help_Button.pack()
    Play_Button = tk.Button(root, text = 'Play the Buy-n-Hold strategy', font = 'Ariel 25 bold', command = play_the_market_buynhold, pady = 15, width = 29)
    Play_Button.pack()
    Play2_Button = tk.Button(root, text = 'Play the Prophet Predictive strategy', font = 'Ariel 25 bold', command = play_the_market_prophet, pady = 15, width = 29)
    Play2_Button.pack()
    Logout_Button = tk.Button(root, text = 'Logout', font = 'Ariel 25 bold', command = back_button, pady = 15, width = 29)
    Logout_Button.pack()

#Display the Game login Screen 
def game_login():
    global root
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Login', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Text_Label = tk.Label(root, text = 'Please create an account or login to play the market', font = 'Ariel 20 bold', pady = 20)
    Text_Label.pack()
    #function for when the login button was clicked
    def login_button():
        if User_Entry.get() == '' or Pass_Entry.get() == '':
            messagebox.showerror('Error!','Please enter a username and password')
        else:
            returned = manage_user(User_Entry.get(),Pass_Entry.get(),login=0)
            if returned == True:
                game(User_Entry.get())
    #function for when the create button was clicked
    def create_button():
        if User_Entry.get() == '' or Pass_Entry.get() == '':
            messagebox.showerror('Error!','Please enter a username and password')
        else:
            manage_user(User_Entry.get(),Pass_Entry.get(),login=1)
    #Back button function
    def back_button():
        clear_window(root)
        mainmenu()
    #GUI definition
    Frame1 = tk.Frame(root, pady = 20)
    Frame1.pack()
    User_label = tk.Label(Frame1, pady = 15, padx = 10, text = 'Enter Username',font='Ariel 25 bold')
    User_label.grid(row = 0,column = 0)
    User_Entry = tk.Entry(Frame1, font = 'Ariel 25 bold')
    User_Entry.grid(row = 0, column = 1)
    Pass_label = tk.Label(Frame1, text = 'Enter Password',font='Ariel 25 bold')
    Pass_label.grid(row = 1,column = 0)
    Pass_Entry = tk.Entry(Frame1, font = 'Ariel 25 bold')
    Pass_Entry.grid(row = 1, column = 1)
    Login_Button = tk.Button(root, text = 'Login', font = 'Ariel 25 bold', command = login_button, pady = 15, width = 20)
    Login_Button.pack()
    Create_Button = tk.Button(root, text = 'Create Account', font = 'Ariel 25 bold', command = create_button, pady = 15, width = 20)
    Create_Button.pack()
    Back_Button = tk.Button(root, text = 'Back', font = 'Ariel 25 bold', command = back_button, pady = 15, width = 20)
    Back_Button.pack()
    
def functions_menu(stock_object,days_to_predict):
    global root
    clear_window(root)
    def show_stock():
        stock_object.plot_stock()
    def show_model_comparison():
        model, model_data = stock_object.create_prophet_model(days=days_to_predict)
    def evaluate_model_accuracy():
        stock_object.evaluate_prediction()
    def predict_future():
        stock_object.predict_future(days=days_to_predict)
    def back():
        start_menu()
    #GUI definition
    Title_Label = tk.Label(root, text = 'Stock Graphs and data', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Wait_Label = tk.Label(root, text = 'Please Wait....', font = 'Ariel 40 bold', pady = 20)
    ShowStock_Button = tk.Button(root, text = 'Show Stock History', font = 'Ariel 25 bold', command = show_stock, pady = 20, width = 25)
    ShowStock_Button.pack()
    ShowModel_Button = tk.Button(root, text = 'Show Model Comparison', font = 'Ariel 25 bold', command = show_model_comparison, pady = 20, width = 25)
    ShowModel_Button.pack()
    Eval_Button = tk.Button(root, text = 'Evaluate Model Accuracy', font = 'Ariel 25 bold', command = evaluate_model_accuracy, pady = 20, width = 25)
    Eval_Button.pack()
    Predict_Button = tk.Button(root, text = 'Predict Next '+str(days_to_predict)+' days', font = 'Ariel 25 bold', command = predict_future, pady = 20, width = 25)
    Predict_Button.pack()
    Back_Button = tk.Button(root, text = 'Back', font = 'Ariel 25 bold', command = back, pady = 20, width = 25)
    Back_Button.pack()
    
def start_menu():
    global root
    clear_window(root)
    #Functions for every button
    def back_button():
        clear_window(root)
        mainmenu()
    def go_button():
        name = Company_Entry.get()
        try:
            days = int(Days_Entry.get())
        except:
            messagebox.showerror('Error!','Number of days has to be a number')
            return
        if name == '':
            messagebox.showerror('Error!','Please enter a company symbol! Example: "AMZN"')
        else:
            obj = Stocker(name)
            if obj != None:
                functions_menu(obj,days)
    Title_Label = tk.Label(root, text = 'Stock Predictor', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Frame = tk.Frame(root, pady=20)
    Frame.pack()
    Company_Label = tk.Label(Frame, text='Enter the Company\'s Symbol:', font = 'Ariel 25 bold')
    Company_Label.grid(row=0,column=0)
    Company_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Company_Entry.grid(row=0, column=1, padx=10, pady = 15)
    Days_Label = tk.Label(Frame, text='Enter the Number of days to predict:', font = 'Ariel 25 bold')
    Days_Label.grid(row=1,column=0, padx = 10, pady = 15)
    Days_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Days_Entry.grid(row=1,column=1)
    Predict_Button = tk.Button(root, text = 'Go!', font = 'Ariel 25 bold', command = go_button, pady = 10, width = 25)
    Predict_Button.pack()
    Back_Button = tk.Button(root, text = 'Back', font = 'Ariel 25 bold', command = back_button, pady = 10, width = 25)
    Back_Button.pack()

def mainmenu():
    #function to start and initialize the program
    global root
    root.attributes("-fullscreen",True)
    #display the main greeting menu
    def predict_button():
        start_menu()
    def play_stock_button():
        game_login()
    def exit_button():
        exit(0)
    Title_Label = tk.Label(root, text = 'Stocker', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Catchphrase_Label = tk.Label(root, text = 'Stalking your Stocks!', font='Ariel 15', pady = 20)
    Catchphrase_Label.pack()
    Start_Button = tk.Button(root, text = 'Predict Stocks', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = predict_button)
    Start_Button.pack()
    Update_Button = tk.Button(root, text = 'Play The Stock Market',font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = play_stock_button)
    Update_Button.pack()
    Exit_Button = tk.Button(root, text = 'Exit', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = exit_button)
    Exit_Button.pack()
    
root = tk.Tk()
mainmenu()
