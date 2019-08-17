import pandas as pd
import quandl, math
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

#We get the historial data from quandl
df = quandl.get("WIKI/GOOGL")
#Now we tweak the dataframe to include only those columns we are interested in
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
#We calculate the percent diffrerence between the highest and lowest scores & also the percent of change
# between the closing score and the opening score for the day
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
#We again tweak the dataframe to include the columns we are interested in
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#Just record the last date present in the dataframe. We'll use this while plotting the predicted prices
last_date = df.iloc[-1].name
df['Adj. Close'].plot()
'''
We need to now create a situation which allows us to train our machine. In this case, we want the machine to be able to tell what the future price is while looking at the current price.
The best way to do that is by telling the machine to look at the price at a particular date and then check the price 2 days later (just an example). This would help the machine learn the pattern of change between the current price and future price
'''
#As explained, our close values are future values. For e.g. value of the 27th is a future value for the price on date 20th
forecast_col = 'Adj. Volume'
#Make sure we handle missing data with the next line. This line would be have executed earlier too.
df.fillna(value=-99999, inplace=True)
#Now we intend to tell our machine to predict 0.5% more of the histrical data that we have.
#Basically, if we have 2000 values, we want the machine to be able to give us additional 20 values, in our case, prices.
forecast_out = int(math.ceil(0.01 * len(df)))
#label is the apparent future price that we just spoke about in line 21-24. What the shift function does is "moves"
#or "shifts" the prices upward in the dataframe to a past date, hence achieving a "future" date on a particular row
df['label'] = df[forecast_col].shift(-forecast_out)
#Now, after we move or shift the prices upward, there will be many rows with no prices at the end. We need to remove those rows completely.
df.dropna(inplace=True)

y = np.array(df['label'])
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)

df.dropna(inplace=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]
forecast_set = clf.predict(X_lately)
print(df['Adj. Close'])
print(forecast_set, confidence, forecast_out)

style.use('ggplot')
df['Forecast'] = np.nan

last_unix = last_date.timestamp()
one_day = 86400 #Since one day = 86400 seconds
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
