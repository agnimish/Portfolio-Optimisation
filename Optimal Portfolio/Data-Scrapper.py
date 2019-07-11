# # Program to get returns data

import pandas as pd
import numpy as np
import pandas_datareader as dr
from datetime import datetime
pd.options.display.float_format = '{:,.2f}'.format

#add symbols of firms here in next line. Check symbols on any website
companies=[]
#sheet=pd.read_excel('tickers.xlsx',header=4,usecols='A',names=['ticker'],nrows=50)
#for i in range(1,51):
#companies.append(sheet.ticker[i-1])
companies = ['goog','mo','dal','fb', 'vedl','amzn','aapl','t','aa','axp','DB','AEM','APD','AMBA','NVS','ANF','LULU'] 
#print(companies,len(companies))
#companies.remove('ABX')
#companies.remove('AET')
#companies.remove('BMH.AX')
df = dr.data.get_data_yahoo(companies,start=datetime(2016,6,1),
                            end=datetime(2018,5,1),interval='m')
data = df[['Adj Close']].iloc[1:20]
log_data = np.log(data)

# ### Formatting and drop NaN values

df = log_data['Adj Close'].diff()
ndata = 100*df
ndata = ndata.dropna()

# ### This will save your data as a .csv file

ndata.to_csv('out.csv')
#print(sheet)
print(ndata)
