import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

df = pd.read_csv('GOOG.csv')
print(df)
last_date = df.iloc[len(df)-1].name
print(last_date)

df = df[['Open',  'High',  'Low',  'Adj Close', 'Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0
df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0
df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
forecast_col = 'Adj Close'

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

print(forecast_set,confidence, forecast_out)
print(last_date)
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

print(df)
df['Adj Close'].plot()
df['Forecast'].plot()
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
