import pandas as pd
import matplotlib.mlab as malb
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from pandas.tseries.offsets import DateOffset

df = pd.read_csv('monthly-milk-production-pounds-p.csv' )
df['Month']= pd.to_datetime(df['Month'])
df.set_index('Month',inplace=True)
print(df.head())
print(df.describe().transpose())
df.plot()
#plt.show()


time_series = df['milk']
time_series.rolling(12).mean().plot(label = 'Rolling mean')
time_series.rolling(12).std().plot(label = 'STD')
time_series.plot()
plt.legend()
#plt.show()

decomp = seasonal_decompose(time_series,freq=12)
decomp.plot()
plt.show()

result = adfuller(df['milk'])
print(result)

df['FirstDiff'] = df['milk'] - df['milk'].shift(1)
result = adfuller(df['FirstDiff'].dropna())
print(result)
df['SeasonalFirstDiff'] = df['FirstDiff'] - df['FirstDiff'].shift(12)
result = adfuller(df['SeasonalFirstDiff'].dropna())
print(result)

model = sm.tsa.statespace.SARIMAX(time_series,order=(0,1,0),seasonal_order=(1,1,1,12))
results = model.fit()
plt.show(results.resid.plot())
#df['forecast']=results.predict()
#print(df)
#plt.plot(df[['milk','forecast']])
#plt.show()
future_dates = [df.index[-1]+ DateOffset(months=x) for x in range(1,24)]
future_df=pd.DataFrame(index=future_dates,columns=df.columns)
final_df = pd.concat([df,future_df])


final_df['forecast']=results.predict(start=168,end=192)
final_df['milk'].plot()
final_df['forecast'].plot()
#time_series.plot()
#finalts = final_df['forecast']
#finalts.plot()
plt.show()
print(final_df)

