#!/usr/bin/env python
# coding: utf-8

# # Install libraries

# In[ ]:


# !pip uninstall pandas -y
# !pip install numpy==1.19.2
# !pip install pandas==1.1.5
# !pip install scikit-learn


# # Imports

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import random
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[ ]:


# Verifying the version
# !pip freeze | grep pandas


# # Data Import

# In[59]:


df_oct = pd.read_csv("../data/2019-Oct.csv")
df_nov = pd.read_csv("../data/2019-Nov.csv")
df_dec = pd.read_csv("../data/2019-Dec.csv")


# In[60]:


df_feb = pd.read_csv("../data/2020-Feb.csv")
df_jan = pd.read_csv("../data/2020-Jan.csv")


# In[ ]:


# Data Exploration and Feature Engineering


# In[ ]:


# df_oct.brand.isnull().sum()


# In[ ]:


# Training Data


# In[ ]:


df = pd.concat([df_oct, df_nov, df_dec])


# In[5]:


# length of training data
len(df)


# In[61]:


# Preprocessing


# In[7]:


df = df[df['price'] != 0.0]


# In[62]:


# feature engineering


# In[66]:


# Convert the 'event_time' column to datetime
df['event_time'] = pd.to_datetime(df['event_time'])


# In[ ]:


# Extract date from 'event_time'
df['date'] = df['event_time'].dt.strftime('%Y-%m-%d')


# In[ ]:


# splitting the event type column
df = pd.get_dummies(df, columns=['event_type'], prefix='', prefix_sep='')


# In[ ]:


# creating product popularity feature
df['product_popularity'] = df['product_id'].map(df['product_id'].value_counts())


# In[ ]:


# creating brand popularity feature
df['brand_popularity'] = df['brand'].map(df['brand'].value_counts())


# In[ ]:


# creating user session activity basis product id
df['session_activity'] = df['user_session'].map(df.groupby('user_session')['product_id'].count())


# In[ ]:


# Creating the Weekpart - to identify the whether the day was a weekday or a weekend
df['Week_Part'] = np.where(df['event_time'].dt.weekday < 5, 'Weekday', 'Weekend')


# In[ ]:


df['product_brand_mix'] = df.groupby(['product_id', 'brand']).ngroup()


# In[ ]:


# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])


# In[ ]:


# Extracting the year, week and the year week
df['year'] = df['date'].dt.isocalendar().year
df['week'] = df['date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) +  df['week'].astype(str)


# In[ ]:


# aggregating the data
df = df.groupby(["year_week", 'product_id', 'category_id', 'price']).agg(
    product_popularity=('product_popularity', 'first'),
    brand_popularity=('brand_popularity', 'first'),
    session_activity=('session_activity', 'sum'),
    Week_Part=('Week_Part', lambda x:x.value_counts().index[0]),  # Most frequent week part
    product_brand_mix=('product_brand_mix', 'first'),
    cart=('cart', 'sum'),
    purchase=('purchase', 'sum'),
    remove_from_cart=('remove_from_cart', 'sum'),
    view=('view', 'sum')
).reset_index()


# In[67]:


# exploring the year week
df['year_week'].value_counts()


# In[70]:


# Sort the DataFrame by year_week
df.sort_values('year_week', inplace=True)


# In[ ]:


# rolling sum - to aggregate data across
def rolling_sum(df, column):
    result = df.groupby(['product_id', 'category_id'])[column].rolling(window=4, min_periods=1).sum()
    result.index = result.index.droplevel(['product_id', 'category_id'])
    return result


# In[ ]:


# Apply the function to the 'view', 'purchase', 'cart' and 'remove_from_cart' columns
for column in ['view', 'purchase', 'remove_from_cart', 'cart']:
    df[f'{column}_last_3_weeks'] = rolling_sum(df, column)


# In[77]:


df


# In[90]:


df['Week_Part'] = df['Week_Part'].map({'Weekday': 0, 'Weekend': 1})


# In[91]:


df = df.fillna(0)


# In[95]:


df['year_week'] = df['year_week'].replace('20201', '202001')
df['year_week'] = df['year_week'].replace('20202', '202002')
df['year_week'] = df['year_week'].replace('20203', '202003')
df['year_week'] = df['year_week'].replace('20204', '202004')
df['year_week'] = df['year_week'].replace('20205', '202005')
df['year_week'] = df['year_week'].replace('20206', '202006')
df['year_week'] = df['year_week'].replace('20207', '202007')
df['year_week'] = df['year_week'].replace('20208', '202008')
df['year_week'] = df['year_week'].replace('20209', '202009')


# In[96]:


data = df.loc[df['purchase'] != 0.0]


# In[97]:


data['purchase'] = data['purchase'].astype(int)


# In[98]:


data


# In[ ]:


# Building Model


# In[99]:


# Split|ting the data into input features (X) and target columns (Y)
X = data[['product_id', 'category_id', 'price', 'product_popularity','brand_popularity', 'session_activity', 'Week_Part','product_brand_mix', 'view_last_3_weeks', 'purchase_last_3_weeks','remove_from_cart_last_3_weeks', 'cart_last_3_weeks', 'year_week']].values
# Y = df[['view', 'remove_from_cart', 'cart', 'purchase', 'count']].values
Y = data[['purchase']].values


# In[ ]:





# In[100]:


# history - checking the history of columns


# In[101]:


data


# In[103]:


# Scaling values for model
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y)


# In[ ]:


# Split the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(X_train, y_train)


# In[157]:


y_pred = model.predict(X_test)


# In[ ]:


y_test_unscaled = y_scaler.inverse_transform(y_test)
y_pred_unscaled = y_scaler.inverse_transform(y_pred)


# In[158]:


# Assuming that y_pred_unscaled is your numpy array
# Clip negative values to 0
y_pred_unscaled = np.maximum(0, y_pred_unscaled)

# Get the decimal part
decimals = y_pred_unscaled % 1

# Apply the conditional ceiling or flooring
y_pred_unscaled = np.where(decimals > 0.75, np.ceil(y_pred_unscaled), np.floor(y_pred_unscaled))


# In[159]:



# Now compute the metrics using inverted values:

MAE = np.mean(np.abs(y_test_unscaled - y_pred_unscaled))
MSE = np.mean((y_test_unscaled - y_pred_unscaled)**2)

# For MAPE, add a small constant to the denominator to avoid division by zero
epsilon = 1e-10 
MAPE = np.mean(np.abs((y_test_unscaled - y_pred_unscaled) / (y_test_unscaled+epsilon))) * 100

# MAAPE
MAAPE = np.mean(np.arctan(np.abs((y_test_unscaled - y_pred_unscaled) / (y_test_unscaled+epsilon)))) 

# R-square
SSR = np.sum((y_pred_unscaled - y_test_unscaled)**2)
SST = np.sum((y_test_unscaled - np.mean(y_test_unscaled))**2)
r2 = 1 - (SSR/SST)

from sklearn.metrics import r2_score
r2score = r2_score(y_test_unscaled, y_pred_unscaled)

# Now you can print or return those metrics
print('MAE:', MAE)
print('MSE:', MSE)
print('MAPE:', MAPE)
print('MAAPE:', MAAPE)
print('R-squared:', r2)
print('R-squared Score:', r2score)


# # Now, let's evaluate the model on the test set
# test_loss = model.evaluate(X_test, y_test, verbose=0)
# print(f"Loss on test set: {test_loss}")


# In[ ]:


# # training 
# MAE: 1.3472884172791535
# MSE: 9.399897001375576
# MAPE: 55.45943970098375
# MAAPE: 0.3975697544125528
# R-squared: 0.8680761823499007
# R-squared Score: 0.8680761823499007
    
# # Final - actual
# MAE: 2.4547002247349976
# MSE: 24.045456959502943
# MAPE: 88.3920433264438
# MAAPE: 0.5978816381935829
# R-squared: 0.6000997243103858
# R-squared Score: 0.6000997243103858
    
# # final after post process
# MAE: 2.2513284132841327
# MSE: 23.280826000567696
# MAPE: 67.9961681750089
# MAAPE: 0.545997490962087
# R-squared: 0.6128163107239442
# R-squared Score: 0.6128163107239442


# In[58]:


import pickle

# save the model to disk
filename = 'LR_finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))

