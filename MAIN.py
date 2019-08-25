#Main project

#importing required packages
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime,math
import tkinter as tk

root = tk.Tk()

def clear_window(window):
    for ele in window.winfo_children():
        ele.destroy()

def predict(company_name, days_to_predict):
    style.use('ggplot')
    df= pd.read_csv("GOOG.csv", header=0, index_col='Date', parse_dates=True)
    #df = quandl.get("WIKI/GOOGL")
    df = df[['Open',  'High',  'Low',  'Adj Close', 'Volume']]
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
    df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
    forecast_col = 'Adj Close'
    df.fillna(value=-99999, inplace=True)
    #forecast_out = int(math.ceil(0.1 * len(df)))
    forecast_out = days_to_predict
    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)

    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = svm.SVR(kernel='linear')
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)


    forecast_set = clf.predict(X_lately)
    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

    check_difference = df['Adj Close'][-days_to_predict-1] - df['Forecast'][len(df[0:-days_to_predict])]
    df['Forecast'] = df['Forecast']+check_difference
        
    df['Adj Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

##    style.use('ggplot')
##    df= pd.read_csv("GOOG.csv", header=0, index_col='Date', parse_dates=True)
##    df = df[['Open',  'High',  'Low',  'Adj Close', 'Volume']]
##    df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
##    df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0
##    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
##    forecast_col = 'Adj Close'
##    df.fillna(value=-99999, inplace=True)
##    forecast_out = days_to_predict
##    df['label'] = df[forecast_col].shift(-forecast_out)
##    X = np.array(df.drop(['label'], 1))
##    #print(X)
##    X = preprocessing.scale(X)
##    print(X)
##    X_lately = X[-forecast_out:]
##    X = X[:-forecast_out]
##
##    df.dropna(inplace=True)
##
##    y = np.array(df['label'])
##
##    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
##    clf = LinearRegression(n_jobs=-1)
##    clf.fit(X_train, y_train)
##    confidence = clf.score(X_test, y_test)
##
##    forecast_set = clf.predict(X_lately)
##    df['Forecast'] = np.nan
##
##    last_date = df.iloc[-1].name
##    last_unix = last_date.timestamp()
##    one_day = 86400
##    next_unix = last_unix + one_day
##
##    for i in forecast_set:
##        next_date = datetime.datetime.fromtimestamp(next_unix)
##        next_unix += 86400
##        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
##
##    check_difference = df['Adj Close'][-days_to_predict-1] - df['Forecast'][len(df[0:-days_to_predict])]
##    df['Forecast'] = df['Forecast']+check_difference
##    
##    df['Adj Close'].plot()
##    df['Forecast'].plot()
##    plt.legend(loc=4)
##    plt.xlabel('Date')
##    plt.ylabel('Price')
##    plt.show()
    
def start_menu():
    global root
    clear_window(root)
    def predict_button():
        predict(Company_Entry.get(), int(Days_Entry.get()))
    def back_button():
        clear_window(root)
        mainmenu()
    Title_Label = tk.Label(root, text = 'Predictor', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Frame = tk.Frame(root, pady=20)
    Frame.pack()
    Company_Label = tk.Label(Frame, text='Company Name:', font = 'Ariel 25 bold')
    Company_Label.grid(row=0,column=0)
    Company_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Company_Entry.grid(row=0, column=1, padx=10, pady = 15)
    Days_Label = tk.Label(Frame, text='Number of days to predict:', font = 'Ariel 25 bold')
    Days_Label.grid(row=1,column=0, padx = 10, pady = 15)
    Days_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Days_Entry.grid(row=1,column=1)
    Predict_Button = tk.Button(root, text = 'Predict!', font = 'Ariel 25 bold', command = predict_button, pady = 10, width = 25)
    Predict_Button.pack()
    Back_Button = tk.Button(root, text = 'Back', font = 'Ariel 25 bold', command = back_button, pady = 10, width = 25)
    Back_Button.pack()

def mainmenu():
    #function to start and initialize the program
    global root
    root.attributes("-fullscreen",True)
    def start_button():
        start_menu()
    def update_button():
        pass
    def exit_button():
        exit(0)
    Title_Label = tk.Label(root, text = 'Stocker', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Catchphrase_Label = tk.Label(root, text = 'Stalking your Stocks!', font='Ariel 15', pady = 20)
    Catchphrase_Label.pack()
    Start_Button = tk.Button(root, text = 'Start', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = start_button)
    Start_Button.pack()
    Update_Button = tk.Button(root, text = 'Update',font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = update_button)
    Update_Button.pack()
    Exit_Button = tk.Button(root, text = 'Exit', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = exit_button)
    Exit_Button.pack()
mainmenu()
