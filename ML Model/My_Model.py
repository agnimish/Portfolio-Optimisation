#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'ML Model'))
	print(os.getcwd())
except:
	pass

#%%
# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Model Evaluation Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
np.warnings.filterwarnings('ignore')

# Importing Models
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb

#%% [markdown]
# # BUILDING LINEAR REGRESSION MODEL
# ### TRAINING AND TESTING

#%%
# Loading Dataset

df=pd.read_csv('G1.csv')
df_test=pd.read_csv('G2.csv')


#%%
df.head()


#%%
df_test.head()

#%% [markdown]
# ## Correlation Heatmap

#%%
# # Correlation Heatmap

plt.subplots(figsize=(20,13 ))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False)

#%% [markdown]
# ### Finding Correlation of features with Target Variable (TRR)

#%%
# Correlation of features with target feature (Return)
correlation_matrix['TRR'].sort_values(ascending = False)

#%% [markdown]
# ### We found that F21 is the only feature having Correlation > 0.9 with TRR
# ***We Choose top 4 highest correlated features***

#%%
# # Retaining numerical features only (1st Attempt)
# numerical_df = df.drop(['TimeStamp', 'Identifier','F7', 'F13', 'AssetGroup', 'F2', 'F3', 'F4', 'F12', 'F15', 'F22', 'F23', 'F26', 'F27', 'F29', 'F30'], axis=1)
# numerical_df_test = df_test.drop(['TimeStamp', 'Identifier','F7', 'F13', 'AssetGroup', 'F2', 'F3', 'F4', 'F12', 'F15', 'F22', 'F23', 'F26', 'F27', 'F29', 'F30'], axis=1)

# Retaining Top 4 features
numerical_df = df[['F21','S','F11','F14', 'TRR']]
numerical_df_test = df_test[['F21','S','F11','F14', 'TRR']]


#%%
numerical_df.describe()

#%% [markdown]
# ### Pair Plots between features
# ***We can clearly see that, Plot b/w F21 vs TRR depicts high positive Correlation***

#%%
# PairPlot between F21 and TRR (highly correlated)
# NOTE: Running this snippet takes around 2-3 Minutes.

sns.set(style="ticks")
sns.pairplot(numerical_df)
plt.savefig('pairplots_coloured')

#%% [markdown]
# ### Removing outliers using Z-Score Technique

#%%
numerical_df = numerical_df[(np.abs(stats.zscore(numerical_df)) < 3).all(axis=1)]
numerical_df_test = numerical_df_test.dropna()
numerical_df_test = numerical_df_test[(np.abs(stats.zscore(numerical_df_test)) < 3).all(axis=1)]


#%%
numerical_df.shape


#%%
numerical_df_test.shape


#%%
# Preparing input and target arrays for the model

# Separating out the target
y = numerical_df.loc[:,['TRR']].values
y_new= numerical_df_test.loc[:,['TRR']].values

# Separating out the Input
x = numerical_df.drop(['TRR'], axis=1).values
x_new=numerical_df_test.drop(['TRR'],axis=1).values

#%% [markdown]
# ### FEATURE ENGINEERING 
#%% [markdown]
# ***Scaling Features***

#%%

x = StandardScaler().fit_transform(x)
x_new = StandardScaler().fit_transform(x_new)


#%%
# 1. Applying PCA (not required)

# pca = PCA(n_components=10)
# principalComponents = pca.fit_transform(x)
# principalComponents_new = pca.fit_transform(x_new)

# principalDf = pd.DataFrame(data = principalComponents)
# principalDf_new = pd.DataFrame(data = principalComponents_new) 

# TRR = pd.DataFrame(data=y)
# X = np.array(principalDf)
# X_new = np.array(principalDf_new)


#%%
# Converting into array for model
X = np.array(x)
X_new = np.array(x_new)

#%% [markdown]
# ### Applying Linear Regression Model

#%%

model = LinearRegression().fit(X, y)

print("Multi Linear Regression using PCA :")

yPred = model.predict(X_new)

print("mean squared error is : " + str(mean_squared_error(y_new, yPred)))
print("R^2 value is : " + str(r2_score(y_new, yPred)))

plt.scatter(y_new, yPred, color='red')
plt.xlabel('y_new')
plt.ylabel('yPred')

#%% [markdown]
# ### Gradient Boosting using XG-Boost 

#%%

print("Gradient Boosting :")

xgdmat = xgb.DMatrix(X,y)
our_params = {'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:squarederror','max_depth':3,'min_child_weight':1}
final_gb = xgb.train(our_params,xgdmat)
tesdmat = xgb.DMatrix(X_new)
y_pred = final_gb.predict(tesdmat)

print("Mean sqaured error:" + str(mean_squared_error(y_new,y_pred)))
print("R^2 score :" + str(r2_score(y_new,y_pred)))

plt.scatter(y_new, yPred, color='green')
plt.xlabel('y_new')
plt.ylabel('yPred')

#%% [markdown]
# ### We can Clearly see that Linear Regression Model Works far better than XG-Boost.
#%% [markdown]
# ## PEDICTING RETURNS USING TRAINED MODEL
# 

#%%
# PREDIT RETURNS

months = ['2016-01-31','2016-02-29','2016-03-31','2016-04-30','2016-05-31','2016-06-30','2016-07-31','2016-08-31',
            '2016-09-30','2016-10-31','2016-11-30','2016-12-31','2017-01-31','2017-02-28','2017-03-31','2017-04-30','2017-05-31','2017-06-30','2017-07-31','2017-08-31',
            '2017-09-30','2017-10-31','2017-11-30','2017-12-31','2018-01-31','2018-02-28','2018-03-31','2018-04-30','2018-05-31','2018-06-30','2018-07-31','2018-08-31',
            '2018-09-30','2018-10-31','2018-11-30']
stocks = []

for i in range(0, len(months)):
    stocksDF = df_test[df_test['TimeStamp']==months[i]][['F21','S','F11','F14']]
    stocksDF = stocksDF.dropna()
    stocksDF = stocksDF[(np.abs(stats.zscore(stocksDF)) < 3).all(axis=1)]

    x = stocksDF.values
    x = StandardScaler().fit_transform(x)
    X = np.array(x)

    returns = model.predict(X)
    returns = returns.transpose()
    stocks.append(returns[0])


#%%
stocks = pd.DataFrame(data=stocks).iloc[:, 0:400]
stocks['Months'] = months
stocks.set_index('Months', inplace=True)


#%%
stocks.to_csv('predicted_stocks.csv')


#%%
stocks

#%% [markdown]
# ## FINDING OPTIMAL WEIGHTS
# ### Using Predicted Stock Returns

#%%
df = pd.read_csv('predicted_stocks.csv')
# Building Portfolio using 4 stocks
df = df.iloc[:, 0:5]
df.head()

#%% [markdown]
# ***Statistical Properties of stocks***

#%%

mean_return = df.mean(axis=0)  # Stocks Return
std_dev = df.std(axis=0)  # Stocks Risk (Standard Deviation)
cov_matrix = np.matrix(df.cov())  # Stocks Covariance Matrix
corr_matrix = df.corr()  # Stocks Correlation Matrix

#%% [markdown]
# ### Building Portfolios

#%%
import random  # For generating weights

ports = 5000  # Count of Total Portfolios
(m, n) = df.shape

portfolios = []  # Set of Portfolios
all_weights = []  # Set of Portfolio Weights

for i in range(1, ports):
#    Generating Weights
    w = [np.sqrt(random.random()*random.random())*(random.random()*50) 
         for i in range(1,n)]
    s = sum(w)
    weight = [ i/s for i in w ]
    all_weights.append(weight)
#     Portfolio Properties
    portfolio_return = np.dot(weight, mean_return)
    variance = np.matmul(np.matmul(weight, cov_matrix), np.transpose(weight))
    portfolio_std_dev = np.sqrt(variance[0,0])
    sharpe_ratio = portfolio_return / portfolio_std_dev  # Assuming Rf=0
#     Add Portfolio to the list
    portfolios.append((portfolio_return, portfolio_std_dev, sharpe_ratio))
    
portfolios = pd.DataFrame(portfolios, 
                          columns=['Return', 'Std. Dev.', 'Sharpe Ratio'])

#%% [markdown]
# ### Plotting Portfolios and finding Optimal Weights

#%%
# Plot Portfolios
plt.figure(figsize=(12,8))
plt.scatter(portfolios.iloc[:,1], portfolios.iloc[:,0],
            c=portfolios.iloc[:,2], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risk')
plt.ylabel('Return')

# Find Point with Maximum sharpe Ratio
idx = np.argmax(portfolios.iloc[:,2])
plt.scatter(portfolios.iloc[idx, 1], portfolios.iloc[idx, 0], c='red', s=50) # Plotting the point (red dot)

plt.show()

#%% [markdown]
# ### Optimal Portfolio

#%%
# Return Weights with Max Sharpe Ratio
opt_weight = all_weights[idx]
opt_weight

#%% [markdown]
# ## _Thank You!_

