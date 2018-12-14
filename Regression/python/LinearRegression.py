
#import the libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

#load data
advertising = pd.read_csv("advertising.csv")
advertising.head()

#print shape
advertising.shape

#print the dataset information
advertising.info()

#describe the dataset
advertising.describe()

#Visualizing the data
import matplotlib.pyplot as plt
import seaborn as sns

#pair plot
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', size=4, aspect=1, kind='scatter')
plt.show()

#correlation heatmap
sns.heatmap(advertising.corr(), cmap='YlGnBu', annot=True)
plt.show()

#Performing Simple Linear Regression
X = advertising['TV']
y = advertising['Sales']

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

#check the train data
X_train.head()
y_train.head()

#Build a Linear Model
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()
lr.params

#print the linear regression summary
print(lr.summary())

#visualize the fitted line
plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()

#Residual analysis
y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)

fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()

#check the residual plots
plt.scatter(X_train,res)

#Prediction on Test Set
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)
y_pred.head()

#Check the fitness metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)
r_squared

plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()
plt.show()
