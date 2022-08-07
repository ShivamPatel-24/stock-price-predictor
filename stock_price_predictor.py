# This program predicts stock prices by using machine learning models

import quandl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import datetime
from dateutil.relativedelta import relativedelta
# Data viz
import plotly.graph_objs as go

# Get the stock data of the desired company
df = quandl.get("WIKI/FB")

# Get the Adjusted Close Price
df = df[['Adj. Close']]

# A varibale for predicting 'n days out into the future
forecast_out = 1000

# Create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

# Create the independent data set (x)
# Convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'], 1))

# Remove the last 'n' (forecast_out) rows
X = X[:-forecast_out]
# print(X)

# Create the dependent data set (Y)
# Convert the dataframe to a numpy array (all of the values including the NaN)
Y = np.array(df['Prediction'])

# Get all of the y values except the last 'n' rows
Y = Y[:-forecast_out]
# print(Y)

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
# print("svm confidence: ", svm_confidence)

# Create and train the Linear Regression Model
lr = LinearRegression()

# Train the model
lr.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
# print("lr confidence: ", lr_confidence)

# Set x_forecast equal to the last n rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
print("x_forecast: ", x_forecast)

# Print support vector regressor model predictions for the next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
# print(svm_prediction)

# Print linear regression model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
# print(lr_prediction)


# Compute the month/year range of prediction
start_date = df[-1:].index[0]
end_date = datetime.datetime.today() + relativedelta(days=+forecast_out)
date_range = pd.date_range(
    start_date, end_date, freq='MS').strftime("%Y-%b").tolist()

# Figure declaration
graph = go.Figure()

# Adding Trace of Support Vector Machine and Linear Regression
graph.add_trace(go.Scatter(x=date_range, y=svm_prediction,
                name="Support Vector Machine"))
graph.add_trace(go.Scatter(
    x=date_range, y=lr_prediction, name="Linear Regression"))

# Adding updates
graph.update_xaxes(
    rangeslider_visible=True, title='Slider above'
)

# Plotting
graph.show()
