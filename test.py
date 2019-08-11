import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as pl

dates = []
prices = []

def get_data(filename):
    count = 0
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            count +=1
            dates.append(count)
            prices.append(float(row[1]))

def  predict_prices(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel='poly',C=1e3, degree = 2)
    #svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    #svr_poly.fit(dates, prices)
    #svr_rbf.fit(dates, prices)
    
    pl.scatter(dates, prices, color='black', label='Data')
    #pl.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    pl.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    #pl.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    pl.xlabel('Date')
    pl.ylabel('Price')
    pl.title('Support Vector Regression')
    pl.legend(loc='upper left')
    pl.show()

    return svr_lin.predict(x)[0]#, svr_poly.predict(x)[0], svr_rbf.predict(x)[0]

get_data('GOOG.csv')

predicted_price = predict_prices(dates, prices, [[0],[0]])
print(predicted_price)

