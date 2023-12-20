import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from Model_Development_Plots import *

df = pd.read_csv('laptops.csv',header=0)

################################ Single Linear Regression ################################
lm = LinearRegression()
X = df[['CPU_frequency']]
Y = df['Price']
lm.fit(X,Y)
Yhat = lm.predict(X)
print(Yhat[0:5])

#Draw the Distribution Plot using function from  Model_Development_Plots
draw_distplot(df['Price'], Yhat)

#Print the findings
print('Intercept is: ', lm.intercept_)
print('Slope is: ', lm.coef_)
print('R-square for Linear Regression is: ',lm.score(X, Y))
print('mean square error of price and predicted value is: ',mean_squared_error(df['Price'], Yhat))


################################ Multiple Linear Regression ################################
lm1 = LinearRegression()
Z = df[['CPU_frequency','RAM_GB','Storage_GB_SSD','CPU_core','OS','GPU','Category']]
lm1.fit(Z,Y)
Yhat = lm1.predict(Z)
draw_distplot(Z, Yhat)
print('Intercept is: ', lm1.intercept_)
print('Slope is: ', lm1.coef_)
print('R-square for MLR is: ',lm1.score(Z, Y))
print('mean square error of price and predicted value is: ',mean_squared_error(df['Price'], Yhat))

################################ Polynomial Regression ################################
X = X.to_numpy().flatten()
f1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(X, Y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(X, Y, 5)
p5 = np.poly1d(f5)

draw_Polly(p1, X, Y, 'CPU_frequency')

r_squared_1 = r2_score(Y, p1(X))
print('The R-square value for 1st degree polynomial is: ', r_squared_1)
print('The MSE value for 1st degree polynomial is: ', mean_squared_error(Y,p1(X)))
r_squared_3 = r2_score(Y, p3(X))
print('The R-square value for 3rd degree polynomial is: ', r_squared_3)
print('The MSE value for 3rd degree polynomial is: ', mean_squared_error(Y,p3(X)))
r_squared_5 = r2_score(Y, p5(X))
print('The R-square value for 5th degree polynomial is: ', r_squared_5)
print('The MSE value for 5th degree polynomial is: ', mean_squared_error(Y,p5(X)))


################################ Pipeline ################################
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print('MSE for multi-variable polynomial pipeline is: ', mean_squared_error(Y, ypipe))
print('R^2 for multi-variable polynomial pipeline is: ', r2_score(Y, ypipe))