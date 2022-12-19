#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# ## Get Stock Data from Yahoo Finance API

# In[2]:


import yfinance as yf

start_time = "2018-01-02"
end_time = "2022-05-10"

def get_stock_data(company, start_time, end_time):
    data = yf.download(company, start=start_time, end=end_time, interval="1d")
    data['Date'] = data.index
    return data


# In[3]:


company_name = ["Google", "Amazon", "Microsoft", "Apple"]


# In[4]:


GOOG = get_stock_data("GOOG", start_time, end_time)

GOOG.tail(10)


# In[5]:


AMZN = get_stock_data("AMZN", start_time, end_time)

AMZN.tail(10)


# In[6]:


MSFT = get_stock_data("MSFT", start_time, end_time)

MSFT.tail(10)


# In[7]:


AAPL = get_stock_data("AAPL", start_time, end_time)

AAPL.tail(10)


# In[8]:


company_list = [GOOG, AMZN, MSFT, AAPL]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
    
df = pd.concat(company_list, axis=0)


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


ma_day = [10, 20, 50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()


# Moving Averages and Adjusted Closes

# In[12]:


fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('Apple')

GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
axes[0,1].set_title('Google')

MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
axes[1,0].set_title('Microsoft')

AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
axes[1,1].set_title('Amazon')

fig.tight_layout()


# ## Sentiment Analyis Results

# In[13]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analysis(company): #gets company name
    data = pd.read_csv('./data/tweets/cleaned{}.csv'.format(company))
    data = data[data.columns[1:]]
    data['tweets'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)), axis=1)

    pos_list, neg_list = [], []
    sia = SentimentIntensityAnalyzer()
    for i in range(len(data["tweetAlsos"])):
        pos, neg = 0, 0
        for text in data["tweets"][i].split('.'):
            res = sia.polarity_scores(text)
            if res['compound'] >= 0.05:
                pos += 1
            elif res['compound'] <= - 0.05:
                neg += 1

        posneg = pos + neg if pos + neg != 0 else 1
        pos_list.append(pos / posneg)
        neg_list.append(neg / posneg)

    df = pd.DataFrame(data={'date': data.date, 'pos': pos_list, 'neg': neg_list})
    return df 
  


# In[14]:


import nltk
nltk.download('vader_lexicon')


# In[15]:


google_rates = sentiment_analysis("GOOG")
google_rates.head(10)


# In[16]:


amazon_rates = sentiment_analysis("AMZN")
amazon_rates.head(10)


# In[17]:


microsoft_rates = sentiment_analysis("MSFT")
microsoft_rates.head(10)


# In[18]:


apple_rates = sentiment_analysis("AAPL")
apple_rates.head(10)


# In[19]:


from sklearn.preprocessing import MinMaxScaler

global scaler

def scale(company_data):
    n = len(company_data)
    train_data = company_data[(n // 20) * 10:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data['Open'].values.reshape(-1, 1))

    prediction_days = 30
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data) - 5):  ######
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])  ###### predict 5 days after

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return(x_train, y_train, scaled_data, train_data)


# ## Scaling Operation for 4 companies

# In[20]:


apple_x_train, apple_y_train, apple_scaled, apple_td = scale(AAPL)

microsoft_x_train, microsoft_y_train, microsoft_scaled, microsoft_td = scale(MSFT)

amazon_x_train, amazon_y_train, amazon_scaled, amazon_td = scale(AMZN)

google_x_train, google_y_train, google_scaled, google_td = scale(GOOG)


# ## LSTM Model Implementation

# In[21]:


from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

def LSTM_trend_model(dim):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(dim, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model


# In[71]:


import datetime

def predict_trend(company, x_train, y_train, scaled_data, train_data):
    model = LSTM_trend_model(x_train.shape[1])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    x_test = []
    pred_period = 15
    prediction_days = 30
    end_time = "2022-05-12"
    for day in range(prediction_days, prediction_days - pred_period, -1):
        x_test.append(scaled_data[-day:, 0])

    np.append(x_test[-1], [1])

    y_pred = [company['Open'].values[-1]]
    for i in range(pred_period):
        x_test[i] = np.array(x_test[i])
        x_test[i] = np.reshape(x_test[i], (1, 30, 1))
        predicted_prices = model.predict(x_test[i])
        for j in range(i + 1, pred_period):
            x_test[j] = np.append(x_test[j], [predicted_prices])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(train_data['Open'].values.reshape(-1, 1))
        y_pred.append(scaler.inverse_transform(predicted_prices)[0][0])

    model.
    now = datetime.datetime(int(end_time[:4]), int(end_time[5:7]), int(end_time[8:10]))
    date = [str(now - datetime.timedelta(days=1))[:10]]
    for i in range(pred_period):
        date.append(str(now + datetime.timedelta(days=i))[:10])

    res = {'Date': date, 'Pred': y_pred}
    res = pd.DataFrame(data=res)
    res = res.set_index(['Date'])
    res['Date'] = res.index
    
    return model, res


# ## Getting LSTM Results

# In[74]:


import mlflow
alpha = 1
l1_ratio = 1

with mlflow.start_run():

    model_LSTM_apple, res_LSTM_apple = predict_trend(AAPL, apple_x_train, apple_y_train, apple_scaled, apple_td)

    model_LSTM_microsoft, res_LSTM_microsoft = predict_trend(MSFT, microsoft_x_train, microsoft_y_train, microsoft_scaled, microsoft_td)

    model_LSTM_amazon, res_LSTM_amazon = predict_trend(AMZN, amazon_x_train, amazon_y_train, amazon_scaled, amazon_td)

    model_LSTM_google, res_LSTM_google = predict_trend(GOOG, google_x_train, google_y_train, google_scaled, google_td)

    mlflow.sklearn.log_model(model_LSTM_apple, 'model')
    mlflow.sklearn.log_model(model_LSTM_microsoft, 'model')
    mlflow.sklearn.log_model(model_LSTM_amazon, 'model')
    mlflow.sklearn.log_model(model_LSTM_google, 'model')


# In[76]:


res_LSTM_apple


# In[77]:


res_LSTM_microsoft


# In[79]:


res_LSTM_amazon


# In[81]:


res_LSTM_google


# In[44]:


# import matplotlib.pyplot
# import matplotlib.dates

# dates = matplotlib.dates.date2num(res_LSTM_google['Date'])
# matplotlib.pyplot.plot_date(dates, res_LSTM_google['Pred'])


# In[82]:


# def showModelPlot(x_train, x_res):
#     plt.figure(figsize=(16,6))
#     plt.title('LSTM Model')
    
#     plt.xlabel('Date', fontsize=18)
#     plt.ylabel('Close Price USD ($)', fontsize=18)
#     plt.plot(x_train['Close'])
#     plt.plot(x_res['Pred'])
#     plt.legend(['Train Set', 'Prediction'])
#     plt.show()


# In[83]:


# showModelPlot(GOOG, res_LSTM_google)


# In[ ]:


# def displayTrendPrediction(train_data, res):
#     fig = plotly.tools.make_subplots(specs=[[{"secondary_y":False}]])
#     fig.add_trace(go.Scatter(x=train_data['Date'], y=train_data['Open'], name="History"), secondary_y=False, )
#     fig.add_trace(go.Scatter(x=res['Date'], y=res['Pred'], name="Prediction", mode="lines"), secondary_y=False, )
#     fig.update_layout(
#         autosize=False, width=900, height=500,
#         title_text=company,
#         # template="plotly_white"
#     )
#     fig.update_xaxes(title_text="year")
#     fig.update_yaxes(title_text="prices", secondary_y=False)

#     return fig


# In[ ]:


# fig_apple = displayTrendPrediction(apple_x_train, res_LSTM_apple)


# In[47]:


start_time_test = "2022-05-11"
end_time_test = "2022-05-27"

test_google = get_stock_data("GOOG", start_time_test, end_time_test)
test_amazon = get_stock_data("AMZN", start_time_test, end_time_test)
test_microsoft = get_stock_data("MSFT", start_time_test, end_time_test)
test_apple = get_stock_data("AAPL", start_time_test, end_time_test)


# In[48]:


test_google


# In[49]:


test_amazon


# In[84]:


test_microsoft


# In[105]:


test_apple


# In[124]:






