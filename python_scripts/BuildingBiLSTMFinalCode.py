#!/usr/bin/env python
# coding: utf-8

# # Install libraries

# In[9]:


# # uninstall pandas
# !pip uninstall pandas -y

# # install specific version of pandas, numpy, tensorflow and other dependencies
# !pip install numpy==1.19.2
# !pip install pandas==1.1.5
# !pip install scikit-learn
# !pip install tensorflow==2.6.2
# !pip install pyre2
# !pip install gensim==4.1.2
# !pip install protobuf==3.20.*


# In[ ]:


# Imports


# In[ ]:


from __future__ import print_function

import pandas as pd
import numpy as np
import pickle
import random
import sys
from gensim.models import Word2Vec
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization, Add, Subtract, Concatenate, SpatialDropout1D
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, LSTM, Embedding, Bidirectional, Flatten
from tensorflow.keras.layers import Lambda

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import regularizers


from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras import metrics
from keras.callbacks import EarlyStopping


# In[10]:


# # Verifying the version
# !pip freeze | grep pandas


# # Data Import

# In[5]:


df_oct = pd.read_csv("../data/2019-Oct.csv")
df_nov = pd.read_csv("../data/2019-Nov.csv")
df_dec = pd.read_csv("../data/2019-Dec.csv")


# In[194]:


df_feb = pd.read_csv("../data/2020-Feb.csv")
df_jan = pd.read_csv("../data/2020-Jan.csv")


# # Data Exploration and Feature Engineering

# In[ ]:


df_oct.brand.isnull().sum()


# In[ ]:


df = pd.concat([df_oct, df_nov, df_dec])


# In[ ]:


# length of training data
print(len(df))


# # Training Data

# In[236]:


df = pd.concat([df_oct, df_nov, df_dec])


# # Preprocessing

# In[237]:


df = df[df['price'] != 0.0]


# # feature engineering

# In[245]:


# Converting the event_time column to datetime format
df['event_time'] = pd.to_datetime(df['event_time'])


# In[ ]:


# Extracting date from event_time
df['date'] = df['event_time'].dt.strftime('%Y-%m-%d')


# In[246]:


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


# In[247]:


# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])


# In[248]:


# Extracting the year, week and the year week
df['year'] = df['date'].dt.isocalendar().year
df['week'] = df['date'].dt.isocalendar().week
df['year_week'] = df['year'].astype(str) +  df['week'].astype(str)


# In[249]:


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


# In[ ]:


# exploring the year week
df['year_week'].value_counts()


# In[258]:


# Sort the DataFrame by year_week
df.sort_values('year_week', inplace=True)


# In[260]:


# rolling sum - to aggregate data acroos
def rolling_sum(df, column):
    result = df.groupby(['product_id', 'category_id'])[column].rolling(window=4, min_periods=1).sum()
    result.index = result.index.droplevel(['product_id', 'category_id'])
    return result


# In[ ]:


# Apply the function to the 'view', 'purchase', 'cart' and 'remove_from_cart' columns
for column in ['view', 'purchase', 'remove_from_cart', 'cart']:
    df[f'{column}_last_3_weeks'] = rolling_sum(df, column)


# In[ ]:


df


# In[263]:


df['Week_Part'] = df['Week_Part'].map({'Weekday': 0, 'Weekend': 1})


# In[276]:


df = df.fillna(0)


# In[295]:


df['year_week'] = df['year_week'].replace('20201', '202001')
df['year_week'] = df['year_week'].replace('20202', '202002')
df['year_week'] = df['year_week'].replace('20203', '202003')
df['year_week'] = df['year_week'].replace('20204', '202004')
df['year_week'] = df['year_week'].replace('20205', '202005')
df['year_week'] = df['year_week'].replace('20206', '202006')
df['year_week'] = df['year_week'].replace('20207', '202007')
df['year_week'] = df['year_week'].replace('20208', '202008')
df['year_week'] = df['year_week'].replace('20209', '202009')


# In[351]:


data = df.loc[df['purchase'] != 0.0]


# In[398]:


data['purchase'] = data['purchase'].astype(int)


# In[399]:


data


# # Building Model

# In[532]:


# Splitting the data into input features (X) and target columns (Y)
X = data[['product_id', 'category_id', 'price', 'product_popularity','brand_popularity', 'session_activity', 'Week_Part','product_brand_mix', 'view_last_3_weeks', 'purchase_last_3_weeks','remove_from_cart_last_3_weeks', 'cart_last_3_weeks', 'year_week']].values
Y = data[['purchase']].values


# In[533]:


# history - checking the history of columns


# In[534]:


data


# In[ ]:


# Scaling values for model
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y)


# In[538]:


# Split the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# In[539]:


# Reshape the input data to have 3D shape, mandatory for LSTM (num_samples, time_steps, num_features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[540]:


# Hyperparameters
basic_dropout_rate = 0.1
weight_decay = 0.001
num_classes = 1 # Update depending on your problem

# Model
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=basic_dropout_rate + 0.2, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1_l2(l1=weight_decay, l2=weight_decay) )))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=basic_dropout_rate + 0.2, recurrent_dropout=0.1, kernel_regularizer=regularizers.l1_l2(l1=weight_decay, l2=weight_decay) )))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(64, kernel_regularizer=regularizers.l1_l2(l1=weight_decay, l2=weight_decay)))
model.add(Activation('relu'))
model.add(Dropout(basic_dropout_rate + 0.2))
model.add(BatchNormalization())

# model.add(Dense(64, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

# Compile the model (Depends on your problem)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[metrics.MeanSquaredError(name='mse')])


# In[559]:


# training parameters
batch_size = [256, 128, 64, 32]
maxepoches = [75, 25, 15, 10]
learning_rate = [0.0001, 0.00001, 0.000001, 0.000001] 

lr_decay = 1e-6

lr_drop = 5

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


# In[560]:


# Train the model
for batch_size, maxepoches, learning_rate in zip(batch_size, maxepoches, learning_rate):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=maxepoches, callbacks=[reduce_lr, early_stopping], validation_data=(X_test, y_test))


# In[561]:


model.save('models/bilstm_26_nov__016.h5')
model.save_weights('models/weights_bilstm_26_nov__016.h5', overwrite=True)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


y_test_unscaled = y_scaler.inverse_transform(y_test)
y_pred_unscaled = y_scaler.inverse_transform(y_pred)


# In[ ]:


y_pred_unscaled = np.maximum(0, y_pred_unscaled)
decimals = y_pred_unscaled % 1
y_pred_unscaled = np.where(decimals > 0.75, np.ceil(y_pred_unscaled), np.floor(y_pred_unscaled))


# In[562]:


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


# Now, let's evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss on test set: {test_loss}")


# In[ ]:




