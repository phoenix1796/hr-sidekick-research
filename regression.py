# # Multiple Linear Regression

# # Importing the libraries
# import numpy as np
# # import matplotlib.pyplot as plt
# import pandas as pd

# # Importing the dataset
# d1 = pd.read_csv('2007_P.csv')
# d2 = pd.read_csv('2007_W.csv')
# d3 = pd.merge(d2, d1, on="Date")
# # x = d3.iloc[:,1:-5].values
# x = d3.iloc[:,2].values
# x = x.reshape(-1, 1)
# y = d3.iloc[:,10].values

# # Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

# # Feature Scaling
# # from sklearn.preprocessing import StandardScaler
# # sc_X = StandardScaler()
# # X_train = sc_X.fit_transform(X_train)
# # X_test = sc_X.transform(X_test)

# #to convert y into array
# # y_train = y_train.reshape(-1,1) 
# # y_test = y_test.reshape(-1,1) 

# # sc_y = StandardScaler()
# # y_train = sc_y.fit_transform(y_train)
# # y_test = sc_y.fit_transform(y_test) 

# # Fitting Multiple Linear Regression to the Training set


# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# # print(X_train)
# regressor.fit(x_train, y_train)

# # Predicting the Test set results
# y_pred = regressor.predict(x_test)

# """#Building the optimal model using Backward Elimination
# import statsmodels.formula.api as sm
# X = np.append(arr = np.ones((249 , 1)).astype(int), values = X, axis = 1)
# X_opt = X[:, [0,1,2,3,4,5,6]]
# #ordinary least squares
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()

# X_opt = X[:, [0,1,2,3,5,6]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()

# X_opt = X[:, [1,2,3,5,6]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()

# X_opt = X[:, [1,2,5,6]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()"""


import pandas as pd
from process import getDataset

hr  = getDataset('HR_comma_sep.csv')

hr = pd.DataFrame(hr['data'],columns = hr['headers'])

x = hr.drop(columns = ['left','sales','salary']).convert_objects(convert_numeric=True).values
y = pd.to_numeric(hr['left'].values)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

csvAry = []

for i in range(0,len(y_test)):
    csvAry.append([y_test[i],y_pred[i]])

frame = pd.DataFrame(csvAry,columns = ['testing','pred'])

frame.to_csv('data.csv',index=False)